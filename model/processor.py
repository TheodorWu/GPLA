import os
from einops import rearrange
from transformers import AutoProcessor, CLIPProcessor, AutoImageProcessor, AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import ToPILImage
import timm
from einops import rearrange
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import get_token

from utils import batch_of_dicts_to_outer_dict

from model.action_tokenizer import ActionTokenizer


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class ContrastiveProcessorWrapper:
    def __init__(self, cfg, horizon, action_dimension, target_vla, dtype, device, custom_prompt_template=False, dino_version="v2", text_backbone="clip", vision_backbone="dino", start_action_token="") -> None:
        self.cfg = cfg
        self.target_vla = target_vla
        self.device = device
        self.dtype = dtype
        self.custom_prompt_template = custom_prompt_template
        self.dino_version = dino_version
        self.text_backbone = text_backbone
        self.vision_backbone = vision_backbone
        self._init_processor()

        self.prompt_template = cfg.prompt_template
        self.answer_template = cfg.answer_template
        if self.custom_prompt_template:
            self.prompt_template = self.custom_prompt_template
            self.answer_template = self.custom_prompt_template

        self.horizon = horizon
        self.action_dimension = action_dimension
        self.trajectory_length = self.horizon * self.action_dimension

        self.max_length = 77 # all CLIP models have a max length of 77 tokens
        self.truncation = True
        self.start_action_token = start_action_token

    def _init_processor(self):
        if self.text_backbone == "siglip2":
            self.processor = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")
            self.pad_token_id = self.processor.pad_token_id
        elif self.text_backbone == "clip":
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                                                device=self.device,
                                                                torch_dtype=self.dtype)
            self.pad_token_id = self.processor.tokenizer.pad_token_id # pylint: disable=no-member
        else:
            raise ValueError(f"Unsupported text backbone: {self.text_backbone}")

        if self.vision_backbone == "dino" or self.vision_backbone == "dinosiglip":
            if self.dino_version == "v2":
                self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            elif self.dino_version == "v3":
                self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
            else:
                raise ValueError(f"Unsupported DINO version: {self.dino_version}")
        elif self.vision_backbone == "siglip2":
            self.image_processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
        elif self.vision_backbone == "clip":
            self.image_processor = self.processor
        else:
            raise ValueError(f"Unsupported vision backbone: {self.vision_backbone}")

        # it would be great if I could replace a token with a special token, but that's not possible at the moment. Might be added in future versions of transformers


    @torch.no_grad()
    def process_batch(self, batch):

        actions = batch.get("action", None)

        text = self.apply_answer_template(batch["instruction"], batch["captions"], actions)

        model_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(device=self.device)

        image_inputs = self.image_processor(images=batch["frames"], return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(device=self.device)
        model_inputs["pixel_values"] = image_inputs["pixel_values"]
        return model_inputs

    @torch.no_grad()
    def apply_answer_template(self, instructions, captions, actions=None):
        return [ self.answer_template
                    .replace("<task>", instructions[i])
                    .replace("<caption>", captions[i])
                for i in range(len(instructions))
            ]

class VisionActionGroundedContrastiveProcessorWrapper(ContrastiveProcessorWrapper):
    @torch.no_grad()
    def process_batch(self, batch):

        if isinstance(batch.get("action"), list):
            actions = torch.stack(batch.get("action")).to(device=self.device, dtype=self.dtype)
        else:
            actions = batch.get("action", None).to(device=self.device, dtype=self.dtype)
        # if actions:
        #     actions = self.action_tokenizer(torch.stack(
        #         actions).to(device=self.device, dtype=self.dtype))
        # if actions.shape[1] > 1:
        actions = rearrange(actions, 'b h d -> b 1 (h d)')

        text = [ self.answer_template
                    .replace("<task>", batch["instruction"][i])
                    .replace("<caption>", batch["captions"][i])
                    .replace("<action>", f"{self.start_action_token}{actions[i]}")
                for i in range(len(batch["instruction"]))
            ]

        model_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(device=self.device)

        image_inputs = self.image_processor(images=batch["frames"], return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(device=self.device)
        model_inputs["pixel_values"] = image_inputs["pixel_values"]
        model_inputs["action"] = actions
        return model_inputs
