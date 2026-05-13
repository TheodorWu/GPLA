from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DPOLoss(nn.Module):
    """
    Taken from https://pytorch.org/torchtune/0.3/_modules/torchtune/rlhf/loss/dpo.html.

    Direct Preference Optimization (DPO) Loss module: https://arxiv.org/abs/2305.18290
    Simply stated from the paper:

    Intuitively, the DPO update increases the relative log probability of preferred to dispreferred responses,
    but it incorporates a dynamic, per-example importance weight that prevents
    the model degeneration that we find occurs with a naive probability ratio objective.

    Based on the implementation in HF's TRL library:
    https://github.com/huggingface/trl/blob/5d1deb1445828cfd0e947cb3a7925b1c03a283fc/trl/trainer/dpo_trainer.py#L844

    DPO retains similarities to PPO (https://arxiv.org/abs/2009.01325), where it optimizes a policy
    (language) model to align with human preferences, and regularizes the loss function using a baseline
    reference (the frozen, initial language model) to prevent over-fitting to the preference dataset.
    It differs from PPO by optimizing the policy model directly using labelled preference data, rather
    than using an additional reward model to provide feedback.
    This significantly simplifies training and reduces compute overhead.

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Compute the log probabilities of the given labels under the given logits.
        Taken from: https://github.com/aiming-lab/GRAPE/blob/main/TPO-Train/finetune.py

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id
        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps = (per_token_logps * loss_mask).sum(-1)
        size_completion = loss_mask.sum(-1) # how many tokens were included in calculation
        return per_token_logps, size_completion

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps (torch.Tensor): Log probabilities of the policy model
                for the chosen responses. Shape: (batch_size)
            policy_rejected_logps (torch.Tensor): Log probabilities of the policy model
                for the rejected responses. Shape: (batch_size)
            reference_chosen_logps (torch.Tensor): Log probabilities of the reference model
                for the chosen responses. Shape: (batch_size)
            reference_rejected_logps (torch.Tensor): Log probabilities of the reference model
                for the rejected responses. Shape: (batch_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors:
                - losses: The DPO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.

        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss,
        # typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter
        # encodes our uncertainty about  the labels and calculates a conservative DPO loss.
        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) # pylint: disable=not-callable
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing # pylint: disable=not-callable
        )

        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps -
                         reference_rejected_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards


class CPOLoss(DPOLoss):
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0, loss_type: str = "simpo", simpo_gamma: float = 0.5):
        super().__init__(beta, label_smoothing)
        self.loss_type = loss_type
        self.simpo_gamma = simpo_gamma

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor = None, # added these parameters for compatibility with DPO
        reference_rejected_logps: torch.Tensor = None # added these parameters for compatibility with DPO
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CPO loss for a batch of policy and reference model log probabilities.

            Args:
                policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
                policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

            Returns:
                A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
                The losses tensor contains the CPO loss for each example in the batch.
                The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        logits = policy_chosen_logps - policy_rejected_logps

            # The beta is a temperature parameter for the CPO loss, typically something in the range of 0.1 to 0.5.
            # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
            # calculates a conservative CPO loss.

        if self.loss_type == "simpo":
            gamma_logratios = self.simpo_gamma / self.beta
            logits = logits - gamma_logratios
            # This reduces to Equation 3 from the CPO paper when label_smoothing -> 0.
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "sigmoid":
            # This reduces to Equation 3 from the CPO paper when label_smoothing -> 0.
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'simpo']"
            )

        chosen_rewards = self.beta * (policy_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards
