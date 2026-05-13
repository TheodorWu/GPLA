import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from eval.logger import MetricLogger
from eval.metrics import Evaluator
from einops import rearrange

from training.augmentation import Augmentator

class SFT():
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 scheduler=None,
                 checkpoint_dir=None,
                 **kwargs
                 ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = cfg.training.get("clip_grad_norm", False)
        self.max_norm = cfg.training.get("max_norm", 1.0)
        if self.clip_grad_norm:
            print(f"Gradient clipping active with max_norm={self.max_norm}")
        self.loss_fn = None

        self.epochs = cfg.training.epochs
        self.current_epoch = -1

        self.global_train_step = 0
        self.global_val_step = 0
        self.global_test_step = 0
        self.action_step = 0

        self.table_batches_per_epoch = self.cfg.wandb.get(
            "table_batches_per_epoch", 5)
        self.logged_batches = 0

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.checkpoint_dir = checkpoint_dir

        self.accumulated_loss = 0.0
        self.accumulation_count = 0

        self.evaluator = Evaluator()
        self.logger = MetricLogger(log_frequency=self.cfg.wandb.log_frequency)

        if self.cfg.training.get("apply_augmentation", True):
            self.augment_batch = Augmentator().augment_batch

    def end_of_epoch(self):
        self.checkpointing()
        self.end_of_epoch_callbacks()

    def end_of_epoch_callbacks(self):
        # Overwrite this if I want to execute something after an epoch after validation
        pass

    def after_test_callback(self):
        pass

    def checkpointing(self):
        if self.checkpoint_dir:
            torch.save(self.model.state_dict(),
                       f"{self.checkpoint_dir}/{wandb.run.name}-sft.pt")
            torch.save(self.optimizer.state_dict(),
                       f"{self.checkpoint_dir}/{wandb.run.name}-sft-optimizer.pt")
            if self.scheduler:
                torch.save(self.scheduler.state_dict(),
                        f"{self.checkpoint_dir}/{wandb.run.name}-sft-scheduler.pt")

    def cleanup(self):
        # Everything that needs to happen after training or inference is done
        print("\nCleaning up...")
        self.logger.aggregate_and_log_metrics()
        self.checkpointing()
        wandb.finish()
        print("Finished cleaning up...")

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def model_call(self, batch, **kwargs):
        model_inputs = self.model.process_batch(batch)

        if 'image_group_ids' in batch:
            print(f"image_group_ids shape: {model_inputs['image_group_ids'].shape}")
            print(f"image_group_ids dtype: {model_inputs['image_group_ids'].dtype}")
            print(f"image_group_ids min/max: {model_inputs['image_group_ids'].min()}/{model_inputs['image_group_ids'].max()}")
            print(f"image_group_ids unique values: {model_inputs['image_group_ids'].unique()}")
            print(f"input_ids shape: {model_inputs['input_ids'].shape}")
        outputs = self.model(**model_inputs, **kwargs)

        return model_inputs, outputs

    def inference_step(self, batch, stage):
        if stage == "val":
            step = self.global_val_step
        elif stage == "test":
            step = self.global_test_step
        else:
            step = -1

        _, outputs = self.model_call(batch)

        self.logger.log_metric(
            outputs.loss, "Loss", step=step, epoch=self.current_epoch, stage=stage)

        if hasattr(outputs, 'logits') and outputs.logits is not None:
            confidence = self.evaluator.confidence(outputs.logits)
            self.logger.log_metric(confidence, "Confidence", step=step, epoch=self.current_epoch, stage=stage)

        decoded_outputs, _ = self.model.generate_and_decode(batch)
        if hasattr(self.model, 'action_only_output') and self.model.action_only_output:
            metrics = self.evaluator.evaluate_vla_actions(decoded_outputs, batch)
        else:
            metrics = self.evaluator.evaluate_vla_all(decoded_outputs, batch)

        for k, v in metrics.items():
            self.logger.log_metric(
                value=v, name=k, step=step, epoch=self.current_epoch, stage=stage)

        if stage == "test" and hasattr(decoded_outputs, "actions"):
            action_pred = decoded_outputs["actions"]
            action_pred = rearrange(action_pred, "b h d -> b (h d)")
            action_gt = batch["action"]
            action_gt = rearrange(action_gt, "b h d -> b (h d)")
            # Get target length
            target_len = action_gt.shape[1]  # (h d)
            gen_len = action_pred.shape[1]

            if gen_len < target_len:
                # Pad with zeros
                pad_size = target_len - gen_len
                action_pred = F.pad(action_pred, (0, pad_size), value=0)
            elif gen_len > target_len:
                # Truncate
                action_pred = action_pred[:, :target_len]

            for i, _ in enumerate(action_gt):
                for j, _ in enumerate(action_gt[i]):
                    try:
                        self.logger.log_metric(
                            value=action_pred[i][j], name=f"action_pred_dim{j}", step=self.action_step, epoch=self.current_epoch, stage=stage)
                        self.logger.log_metric(
                            value=action_gt[i][j], name=f"action_gt_dim{j}", step=self.action_step, epoch=self.current_epoch, stage=stage)
                    except Exception as e:
                        print(f"Error logging action dimension {j} for sample {i}. Pred shape: {action_pred.shape}, GT shape: {action_gt[i].shape}")
                    self.action_step += 1

        self.inference_log_table(batch, decoded_outputs, step, stage)

    def inference_log_table(self, batch, decoded_outputs, step, stage):
        if self.logged_batches < self.table_batches_per_epoch:
            table_data = {k: v for k, v in batch.items()}

            for k, v in decoded_outputs.items():
                table_data[f"output {k}"] = v

            self.logger.log_table(data=table_data, name="Inference", step=step, epoch=self.current_epoch,
                                  stage=stage, str_keys=["action", "output actions"],
                                  img_keys=["frames"],
                                  ignore_keys=["batch_size", "batch_caption_count", "task", "episode_id", "caption_id"])

            self.logged_batches += 1

    def debug_batch_data(self, input_ids, pixel_values, action, attention_mask=None, step=None):
        input_ids = input_ids.float()
        pixel_values = pixel_values.float()
        action = action.float()
        log_data = {}

        # === Text ===
        log_data["text/shape_0"] = input_ids.shape[0]
        log_data["text/shape_1"] = input_ids.shape[1]
        log_data["text/min"] = input_ids.min().item()
        log_data["text/max"] = input_ids.max().item()

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).tolist()
            log_data["text/seq_len_mean"] = sum(seq_lengths) / len(seq_lengths)
            log_data["text/seq_len_min"] = min(seq_lengths)
            log_data["text/seq_len_max"] = max(seq_lengths)
            wandb.log({"text/seq_len_hist": wandb.Histogram(seq_lengths)}, step=step)

        # === Vision ===
        log_data["vision/shape_0"] = pixel_values.shape[0]
        log_data["vision/shape_1"] = pixel_values.shape[1]
        log_data["vision/min"] = pixel_values.min().item()
        log_data["vision/max"] = pixel_values.max().item()
        log_data["vision/mean"] = pixel_values.mean().item()
        log_data["vision/std"] = pixel_values.std().item()

        wandb.log({"vision/pixels_hist": wandb.Histogram(pixel_values.cpu().numpy().flatten())}, step=step)

        # === Actions ===
        log_data["action/shape_0"] = action.shape[0]
        if action.ndim > 1:
            log_data["action/shape_1"] = action.shape[1]
        log_data["action/min"] = action.min().item()
        log_data["action/max"] = action.max().item()
        log_data["action/mean"] = action.mean().item()
        log_data["action/std"] = action.std().item()

        wandb.log({"action/values_hist": wandb.Histogram(action.cpu().numpy().flatten())}, step=step)

        # === Checks ===
        if torch.isnan(pixel_values).any():
            log_data["warnings/nan_in_images"] = 1
        if torch.isnan(action).any():
            log_data["warnings/nan_in_actions"] = 1
        if (action == 0).all():
            log_data["warnings/all_actions_zero"] = 1
        if pixel_values.std() < 0.01:
            log_data["warnings/low_variance_images"] = 1

        # Push scalars/warnings to wandb
        wandb.log(log_data, step=step)

    def log_all_gradients(self, model):
        # Total norm
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

        # Component norms
        component_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                component_norms[f"gradient_norms/{name.replace('.', '/')}"] = param.grad.norm().item()

        wandb.log({
            "gradient_norms/total_norm": total_norm.item(),
            **component_norms
        })

    # @torch.compile
    def train_step(self, batch):
        if self.cfg.training.get("apply_augmentation", True):
            batch = self.augment_batch(batch)

        if self.cfg.training.get("autocast", False):
            with torch.autocast(device_type="cuda" if self.cfg.training.device == "gpu" else "cpu"):
                model_inputs, outputs = self.model_call(batch)

                loss = outputs.loss
        else:
            model_inputs, outputs = self.model_call(batch)

            loss = outputs.loss

        scaled_loss = loss / self.cfg.training.optimizer_step_freq
        scaled_loss.backward()

        self.accumulated_loss += loss.item()
        self.accumulation_count += 1

        # Accumulate gradients
        if self.global_train_step % self.cfg.training.optimizer_step_freq == 0:
            self.log_all_gradients(self.model)
            if self.debug_batch_data:
                try:
                    self.debug_batch_data(model_inputs["input_ids"], model_inputs["pixel_values"], model_inputs["action"])
                except KeyError as e:
                    print(f"Couldn't debug batch data with input data of {model_inputs.keys()}")
                    self.debug_batch_data = False
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
            self.optimizer.step()

            update_step = self.global_train_step // self.cfg.training.optimizer_step_freq

            if self.scheduler:
                self.scheduler.step()
                if self.cfg.wandb.get("log_scheduler", False):
                    current_lrs = self.scheduler.get_last_lr() # get_last_lr() returns a list

                    if self.cfg.wandb.get("log_scheduler_separately", False):
                        for i, lr in enumerate(current_lrs):
                            self.logger.log_metric(
                                value=lr,
                                step=update_step,
                                name=f"LearningRate/Group_{i}",
                                epoch=self.current_epoch,
                                stage="train"
                            )
                    else:
                        current_lr = current_lrs[0]
                        self.logger.log_metric(
                            value=current_lr,
                            step=update_step,
                            name="LearningRate",
                            epoch=self.current_epoch,
                            stage="train"
                        )
            self.optimizer.zero_grad()

            avg_loss = self.accumulated_loss / self.accumulation_count
            self.logger.log_metric(value=avg_loss, step=update_step,
                                   name="Loss", epoch=self.current_epoch, stage="train")
            try:
                self.logger.log_metric(value=torch.exp(self.model.logit_scale).item(), step=update_step,
                                      name="Temperature", epoch=self.current_epoch, stage="train")
            except AttributeError:
                pass

            self.accumulated_loss = 0.0
            self.accumulation_count = 0

    def sanity_check(self):
        with torch.no_grad():
            print("\nPerforming Sanity Check.")
            self.model.eval()
            batch = next(iter(self.val_dataset))
            self.inference_step(batch, stage="sanity")
            print("Sanity Check Complete.")

    def train(self):
        self.sanity_check()
        print("\nStarting Training.")
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            ### Training Loop ###
            self.model.train(True)
            self.optimizer.zero_grad()
            total_batches = (self.cfg.training.stop_train_after if self.cfg.training.stop_train_after else len(self.train_dataset)) * self.cfg.training.optimizer_step_freq
            for i, batch in enumerate(tqdm(self.train_dataset,
                                           desc=f"Training Epoch {epoch+1}",
                                           total=total_batches)):
                if self.cfg.training.stop_train_after and i // self.cfg.training.optimizer_step_freq > self.cfg.training.stop_train_after:
                    break
                self.optimizer.zero_grad()

                self.train_step(batch)
                self.global_train_step += 1


            ### Validation Loop ###
            self.model.eval()
            self.logged_batches = 0
            with torch.no_grad():
                total_batches = (self.cfg.training.stop_val_after if self.cfg.training.stop_val_after else len(self.val_dataset)) * self.cfg.training.optimizer_step_freq
                for i, batch in enumerate(tqdm(self.val_dataset,
                                               desc=f"Validation Epoch {epoch+1}",
                                               total=total_batches)):
                    if self.cfg.training.stop_val_after and i // self.cfg.training.optimizer_step_freq > self.cfg.training.stop_val_after:
                        break
                    self.inference_step(batch, stage="val")
                    self.global_val_step += 1


            self.end_of_epoch()
        print("Training finished.")

    def test(self):
        print("\nStarting Testing.")
        ### Test ###
        self.model.eval()
        self.global_test_step = 0
        self.logged_batches = 0
        with torch.no_grad():
            total_batches = (self.cfg.training.stop_test_after if self.cfg.training.stop_test_after else len(self.test_dataset)) * self.cfg.training.optimizer_step_freq
            for i, batch in enumerate(tqdm(self.test_dataset,
                                           desc="Testing",
                                           total=total_batches)):
                if self.cfg.training.stop_test_after and i // self.cfg.training.optimizer_step_freq > self.cfg.training.stop_test_after:
                    break
                self.inference_step(batch, stage="test")
                self.global_test_step += 1
            self.after_test_callback()
        print("Testing Finished.")
