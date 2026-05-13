import torch
from training.loss import DPOLoss, CPOLoss
from training.sft import SFT
from training.preference import PreferenceGenerator

class GPLA(SFT):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 preference_generator: PreferenceGenerator,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 scheduler=None,
                 checkpoint_dir=None):
        super().__init__(cfg,
                 model,
                 optimizer,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 scheduler=scheduler,
                 checkpoint_dir=checkpoint_dir
                 )

        self.preference_generator = preference_generator
        if "cpo" in cfg.training:
            self.loss_fn = CPOLoss(**cfg.training.cpo)
        elif "dpo" in cfg.training:
            self.loss_fn = DPOLoss(**cfg.training.dpo)
        else:
            raise ValueError("Configuration for 'cpo' or 'dpo' not found. Please check your configuration file.")

        self.simpo_regularization = cfg.training.get("simpo_regularization", False)
        self.regularization_weight = cfg.training.get("regularization_weight", 0.1)


    def train_step(self, batch):
        if self.simpo_regularization:
            _, outputs = self.model_call(batch)
            standard_loss = outputs.loss

        # self.model.eval()
        batch = self.preference_generator.generate_preference_from_batch(self.model, batch)
        # self.model.train()

        batch_chosen = batch["chosen"]
        batch_rejected = batch["rejected"]

        chosen_inputs, chosen_outputs = self.model_call(batch_chosen)

        seq_len = chosen_inputs["labels"].shape[1]  # Assuming input_ids is the first key
        chosen_logits = chosen_outputs.logits[:, :seq_len, :]  # Shape: (b, n, vocab_size)
        chosen_logps, _ = self.loss_fn.get_batch_logps(chosen_logits, chosen_inputs["labels"], label_pad_token_id=-100, is_encoder_decoder=False)

        rejected_inputs, rejected_outputs = self.model_call(batch_rejected)
        seq_len = rejected_inputs["labels"].shape[1]  # Assuming input_ids is the first key
        rejected_logits = rejected_outputs.logits[:, :seq_len, :]  # Shape: (b, n, vocab_size)
        rejected_logps, _ = self.loss_fn.get_batch_logps(rejected_logits, rejected_inputs["labels"], label_pad_token_id=-100, is_encoder_decoder=False)

        losses, _, _ = self.loss_fn(chosen_logps, rejected_logps)
        loss = losses.mean()

        if self.simpo_regularization:
            self.logger.log_metric(value=self.regularization_weight * loss.item(), step=self.global_train_step,
                                   name="SIMPO Loss", epoch=self.current_epoch, stage="train")
            self.logger.log_metric(value=standard_loss.item(), step=self.global_train_step,
                                   name="LM Loss", epoch=self.current_epoch, stage="train")
            loss = self.regularization_weight * loss + standard_loss

        scaled_loss = loss / self.cfg.training.optimizer_step_freq
        scaled_loss.backward()

        self.accumulated_loss += loss.item()
        self.accumulation_count += 1

        ## Accumulate gradients
        if self.global_train_step%self.cfg.training.optimizer_step_freq==0:
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()
                if self.cfg.wandb.get("log_scheduler", False):
                    current_lr = self.scheduler.get_last_lr()[0]  # get_last_lr() returns a list
                    self.logger.log_metric(
                        value=current_lr,
                        step=self.global_train_step // self.cfg.training.optimizer_step_freq,
                        name="LearningRate",
                        epoch=self.current_epoch,
                        stage="train"
                    )

            self.optimizer.zero_grad()

            avg_loss = self.accumulated_loss / self.accumulation_count
            self.logger.log_metric(value=avg_loss, step=self.global_train_step // self.cfg.training.optimizer_step_freq,
                                   name="Loss", epoch=self.current_epoch, stage="train")

            self.accumulated_loss = 0.0
            self.accumulation_count = 0

    def visualize_examples(self):
        # during eval, I want to create a figure which creates a tabular image
        # showing image, ground truth text, model text, and similarity scores

        # I want another figure showing image, chosen text, rejected text, similarity scores
        pass
