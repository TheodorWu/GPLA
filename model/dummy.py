import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

class DummyVLA(nn.Module):
    def __init__(self, vocab_size, tokenizer, dtype, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.device = device
        self.tokenizer = tokenizer

        self.dummylayer = nn.Linear(1,1, device=self.device, dtype=self.dtype)
        self.dummyloss = nn.MSELoss()

    def forward(self, input_ids, attention_mask, pixel_values, labels, **kwargs): # pylint: disable=unused-argument
        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]

        x = self.dummylayer(torch.rand(1, dtype=self.dtype, device=self.device))
        loss = self.dummyloss(x, torch.rand(1, dtype=self.dtype, device=self.device))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=torch.rand(batch_size, sequence_length, self.vocab_size, dtype=self.dtype, device=self.device),
            past_key_values=None,
            hidden_states=None,
            attentions=None
        )

    def predict_action(self, **inputs):
        return inputs.get("input_ids")

    def generate(self, **inputs):
        n = 10
        input_ids = self.tokenizer(inputs.get("prompt_text"), return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
        indices = torch.randint(0, input_ids.size(1), (n,))
        samples = input_ids[:, indices]

        out = torch.cat([input_ids, samples] , dim=-1)

        return out
