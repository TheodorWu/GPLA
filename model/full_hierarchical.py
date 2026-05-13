import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature
from utils import printable_params

@printable_params
class FullHierarchicalModel(nn.Module):
    def __init__(self, high_level_vlm, vla, cfg, device=torch.device("cuda"), dtype=torch.bfloat16):
        super(FullHierarchicalModel, self).__init__()
        self.high_level_vlm = high_level_vlm
        self.vla = vla
        # Freeze VLA parameters
        for param in self.vla.parameters():
            param.requires_grad = False
        self.vla.eval()  # Set to eval mode once

        self.cfg = cfg
        self.device = device
        self.dtype = dtype

    def train(self, mode=True):
        # Only set high_level_vlm to training mode
        self.high_level_vlm.train(mode)
        self.vla.eval()  # Keep VLA in eval mode
        self.training = mode  # Update the module's training flag
        return self

    def eval(self):
        self.high_level_vlm.eval()
        self.vla.eval()
        self.training = False
        return self

    def forward(self, **vlm_inputs):
        # Process the batch through the high-level VLM
        vlm_outputs = self.high_level_vlm(**vlm_inputs)

        return vlm_outputs

    def process_batch(self, batch, captions=None, vla=False):
        batch = batch.copy()  # Create a shallow copy
        if vla:
            if captions is not None:
                batch["captions"] = captions
            return { "batch": batch }
        else:
            # Process the batch through the high-level VLM
            return self.high_level_vlm.process_batch(batch)

    def generate_and_decode(self, batch) -> BatchFeature:
        try:
            high_level_text, generated_ids = self.high_level_vlm.generate_and_decode(batch)
            vla_inputs = self.process_batch(batch, captions=high_level_text["tokens"], vla=True)
            actions, _ = self.vla.generate_and_decode(**vla_inputs)
            return BatchFeature(data={"actions": actions["actions"], "tokens": high_level_text["tokens"], "generated_ids": generated_ids}), None
        except Exception as e:
            print(f"Couldn't generate from batch: {batch}\n\n {e}")
