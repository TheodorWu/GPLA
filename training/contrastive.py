import torch
import torch.nn.functional as F
from training.sft import SFT
import wandb

from utils import debug_batch_data
from eval.retrieval import RetrievalEvaluator

class Contrastive(SFT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.retrieval_evaluator = RetrievalEvaluator(
            max_eval_samples=kwargs.get('max_retrieval_samples', 500),
            k_values=kwargs.get('retrieval_k_values', [1, 5, 10]),
            eval_batch_size=kwargs.get('retrieval_eval_batch_size', 32)
        )
        if self.cfg.training.get("watch_model", False):
            wandb.watch(self.model, log_freq=100)

    def check_batch_retrieval(self, image_features, text_features):
        with torch.no_grad():
            img_norm = F.normalize(image_features, dim=1)
            txt_norm = F.normalize(text_features, dim=1)

            similarities = torch.mm(img_norm, txt_norm.t()) / 0.07


        # Check batch accuracy
        correct = (similarities.argmax(dim=1) == torch.arange(len(similarities)).to(similarities.device)).float().mean()
        return correct.item()

    def check_embedding_magnitude(self, text_embeddings, action_vision_embeddings, logits, similarities, stage):
        with torch.no_grad():
            t_mean = text_embeddings.mean().item()
            av_mean = action_vision_embeddings.mean().item()
            tnorms = text_embeddings.norm(p=2, dim=-1).mean().item()
            avnorms = action_vision_embeddings.norm(p=2, dim=-1).mean().item()
            tstd = text_embeddings.std().item()
            avstd = action_vision_embeddings.std().item()
            self.logger.log_metric(t_mean, "Text Embedding Mean", step=self.global_train_step, epoch=self.current_epoch, stage=stage)
            self.logger.log_metric(av_mean, "Action-Vision Embedding Mean", step=self.global_train_step, epoch=self.current_epoch, stage=stage)
            self.logger.log_metric(tnorms, "Text Embedding Norm", step=self.global_train_step, epoch=self.current_epoch, stage=stage)
            self.logger.log_metric(avnorms, "Action-Vision Embedding Norm", step=self.global_train_step, epoch=self.current_epoch, stage=stage)
            self.logger.log_metric(tstd, "Text Embedding std", step=self.global_train_step, epoch=self.current_epoch, stage=stage)
            self.logger.log_metric(avstd, "Action-Vision Embedding std", step=self.global_train_step, epoch=self.current_epoch, stage=stage)

            # Logits
            logit_range = logits.max() - logits.min()
            self.logger.log_metric(logit_range.item(), "Logit Range", step=self.global_train_step, epoch=self.current_epoch, stage=stage)

            # Similarity statistics (before temperature scaling)
            sim_mean = similarities.mean().item()
            sim_std = similarities.std().item()
            sim_max = similarities.max().item()
            sim_min = similarities.min().item()
            self.logger.log_metric(sim_mean, "Similarity Mean", step=self.global_train_step, epoch=self.current_epoch, stage=stage)
            self.logger.log_metric(sim_std, "Similarity Std", step=self.global_train_step, epoch=self.current_epoch, stage=stage)
            self.logger.log_metric(sim_max, "Similarity Max", step=self.global_train_step, epoch=self.current_epoch, stage=stage)
            self.logger.log_metric(sim_min, "Similarity Min", step=self.global_train_step, epoch=self.current_epoch, stage=stage)

            # Diagonal vs off-diagonal (positive vs negative pairs)
            batch_size = similarities.shape[0]
            diagonal_mean = similarities.diag().mean().item()  # Positive pairs
            off_diag_mask = ~torch.eye(batch_size, dtype=bool, device=similarities.device)
            off_diagonal_mean = similarities[off_diag_mask].mean().item()  # Negative pairs
            self.logger.log_metric(diagonal_mean, "Positive pairs", step=self.global_train_step, epoch=self.current_epoch, stage=stage)
            self.logger.log_metric(off_diagonal_mean, "Negative pairs", step=self.global_train_step, epoch=self.current_epoch, stage=stage)
            self.logger.log_metric(diagonal_mean - off_diagonal_mean, "Separation", step=self.global_train_step, epoch=self.current_epoch, stage=stage)


    def check_dimensional_collapse(self, embeddings: torch.Tensor, step: int, epoch: int, stage: str):
        """
        Computes and logs the singular value spectrum of embeddings using custom logger.

        Args:
            embeddings: Tensor of shape (n_samples, embedding_dim)
            logger: Logger object with log_metric method
            step: Global training step
            epoch: Current epoch
            stage: Stage name (e.g., "train", "val")
        """
        with torch.no_grad():
            # 1. Center the embeddings
            centered_embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
            if centered_embeddings.dtype != torch.float32:
                centered_embeddings = centered_embeddings.float()
            # 2. Compute SVD
            # We only need the singular values 'S'
            _, S, _ = torch.linalg.svd(centered_embeddings, full_matrices=False) # pylint: disable=not-callable

            # 3. Normalize
            normalized_S = S / S.sum()

            # Print top 5 singular values
            print(f"--- {stage} ---")
            print(f"Top 5 normalized singular values: {normalized_S[:5].cpu().numpy()}")

            # 4. Log metrics using custom logger
            self.logger.log_metric(
                normalized_S[0].item(),
                "Top singular value",
                step=step,
                epoch=epoch,
                stage=stage
            )

            self.logger.log_metric(
                normalized_S[:5].sum().item(),
                "Top 5 singular values sum",
                step=step,
                epoch=epoch,
                stage=stage
            )

            # Effective rank: measures how many dimensions are actually being used
            effective_rank = (normalized_S.sum() ** 2 / (normalized_S ** 2).sum()).item()
            self.logger.log_metric(
                effective_rank,
                "Effective rank",
                step=step,
                epoch=epoch,
                stage=stage
            )


    def calculate_similarity_matrix(self, batch):
        captions = batch["captions"]
        instructions = batch["instruction"]
        frames = batch["frames"]
        actions = batch["action"]


        caption_axis_len = min(len(captions), 8)  # Limit to 8 captions
        frame_axis_len = min(len(frames), 8)  # Limit to 8
        similarity_matrix = torch.zeros((caption_axis_len, frame_axis_len), device=self.model.device)

        texts = []

        for i, caption in enumerate(captions[:caption_axis_len]):
            for j, frame in enumerate(frames[:frame_axis_len]):
                # Replace this with your model's similarity calculation

                _, outputs = self.model_call({
                    "captions": [ caption ] ,  # Assuming text is a tensor
                    "instruction": [instructions[i]],  # Assuming text is a tensor
                    "frames": frame.unsqueeze(0),   # Assuming frame is a tensor
                    "action": actions[j].unsqueeze(0)  # Assuming actions is a tensor
                })  # Adjust based on your model's API
                similarity_matrix[i, j] = outputs.similarities
            full_text = self.model.processor.apply_answer_template([instructions[i]] , [caption], [actions[j]])
            texts.append(full_text[0])

        return similarity_matrix, texts, frames[:frame_axis_len], actions[:frame_axis_len]

    def top_k_retrieval(self):
        """Delegate to retrieval evaluator"""
        return self.retrieval_evaluator.evaluate_retrieval(
            self.logger, self.global_test_step, self.current_epoch, apply_normalization=True #self.cfg.model.get("final_layer_norm", False)  # Embeddings are already normalized in the model
        )

    def after_test_callback(self):
        """Called after all test batches are processed"""
        self.top_k_retrieval()
        self.retrieval_evaluator.reset_cache()

    def model_call(self, batch, **kwargs):
        model_inputs = self.model.process_batch(batch)
        # debug_batch_data(model_inputs)  # Debugging function to check input data
        outputs = self.model(**model_inputs, **kwargs)

        if hasattr(outputs, 'text_embeddings') and hasattr(outputs, 'action_vision_embeddings') and hasattr(outputs, 'logits') and hasattr(outputs, 'similarities'):
            self.check_embedding_magnitude(outputs.text_embeddings, outputs.action_vision_embeddings, outputs.logits, outputs.similarities, stage="debug")
        return model_inputs, outputs

    def inference_step(self, batch, stage):
        if stage=="val":
            step = self.global_val_step
        elif stage=="test":
            step = self.global_test_step
        else:
            step=-1

        _, outputs = self.model_call(batch)

        if step % 100 == 0:
            action_vision_embeddings = outputs.action_vision_embeddings
            text_embeddings = outputs.text_embeddings
            batch_acc = self.check_batch_retrieval(action_vision_embeddings, text_embeddings)
            self.logger.log_metric(batch_acc, "Batch Retrieval Accuracy", step=step, epoch=self.current_epoch, stage=stage)
            self.check_dimensional_collapse(action_vision_embeddings, step, self.current_epoch, stage + " Action-Vision")
            self.check_dimensional_collapse(text_embeddings, step, self.current_epoch, stage + " Text")

        # Accumulate data for retrieval evaluation during test
        if stage == "test" or stage == "sanity":
            self.retrieval_evaluator.accumulate_data(self, batch, step)

        self.logger.log_metric(outputs.loss, "Loss", step=step, epoch=self.current_epoch, stage=stage)

        if self.global_test_step <= 1 and stage == "test":
            sim_matrix, texts, imgs, actions = self.calculate_similarity_matrix(batch)
            self.logger.create_confusion_matrix_plot(sim_matrix, texts, imgs, actions, title="Text-Image+Action Similarity Matrix", figsize=(12, 8), thumbnail_size=(64, 64))

            self.logger.create_hard_negative_figures(outputs.similarities)

