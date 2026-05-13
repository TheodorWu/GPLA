import torch
import torch.nn as nn

import numpy as np

from lerobot.configs.types import FeatureType, PolicyFeature
from transformers import AutoTokenizer

from utils import printable_params, DotDict

@printable_params
class VLA(nn.Module):
    def __init__(self, cfg, device=torch.device("cuda"), dtype=torch.bfloat16):
        super(VLA, self).__init__()
        self.device = device
        self.dtype = dtype
        self.cfg = cfg
        self.action_only_output = True # Whether this model outputs only actions or also language

        self._init_backbone()

    def _init_backbone(self):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device=self.device, dtype=self.dtype)
        # normalization handled by dataloaders
        self.policy.normalize_inputs = nn.Identity()  # Disable input normalization
        self.policy.normalize_targets = nn.Identity()  # Disable target normalization
        self.policy.unnormalize_outputs = nn.Identity()  # Disable output unnormalization
        # Define the expected input features for the policy
        self.policy.config.input_features["frames"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224)
        )
        self.policy.config.input_features.pop("observation.images.camera1", None)  # Remove old camera feature
        self.policy.config.input_features.pop("observation.images.camera2", None)  # Remove old camera feature
        self.policy.config.input_features.pop("observation.images.camera3", None)  # Remove old camera feature
        self.policy.config.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(2,)  # Assuming state has 2 dimensions; adjust as necessary
        )
        self.policy.config.chunk_size = 8  # Number of steps to predict in one forward pass
        self.policy.config.output_features["action"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(2,)  # Assuming action has 2 dimensions; adjust as necessary
        )

    def process_batch(self, batch):
        batch = { k: v.to(device=self.device, dtype=self.dtype) if torch.is_tensor(v) else v for k, v in batch.items() }
        return {"batch": batch}

    def _project_state_actions(self, batch):
        # Project the state and actions to the expected format of the policy
        batch["observation.state"] = batch.pop("state")
        batch["task"] = batch["captions"]
        return batch

    def forward(self, batch):
        batch = self._project_state_actions(batch)
        # Align vision and language features
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True): # need to run mixed decision. If this doesn't work I have to make sure to create noise in corresponding dtype
            loss, loss_dict = self.policy(batch)
            return DotDict({
                "loss": loss,
                "loss_dict": loss_dict
            })

    def generate_and_decode(self, batch):
        batch = self.process_batch(batch)["batch"]
        batch = self._project_state_actions(batch)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
            generated = self.policy.predict_action_chunk(batch)
            return DotDict({ "actions": generated }), None  # No attention weights returned

class VLAPiZeroFive(VLA):
    def __init__(self, cfg, device=torch.device("cuda"), dtype=torch.bfloat16):
        super(VLAPiZeroFive, self).__init__(cfg, device=device, dtype=dtype)
        self.max_token_len = 512  # Max token length for pi05 tokenizer

    def _init_backbone(self):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        self.policy = PI05Policy.from_pretrained(pretrained_name_or_path="lerobot/pi05_base").to(device=self.device, dtype=self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

        # normalization handled by dataloaders
        self.policy.normalize_inputs = nn.Identity()  # Disable input normalization
        self.policy.normalize_targets = nn.Identity()  # Disable target normalization
        self.policy.unnormalize_outputs = nn.Identity()  # Disable output unnormalization
        # Define the expected input features for the policy
        self.policy.config.input_features["frames"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224)
        )
        self.policy.config.input_features.pop("observation.images.camera1", None)  # Remove old camera feature
        self.policy.config.input_features.pop("observation.images.camera2", None)  # Remove old camera feature
        self.policy.config.input_features.pop("observation.images.camera3", None)  # Remove old camera feature
        self.policy.config.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(2,)  # Assuming state has 2 dimensions; adjust as necessary
        )
        self.policy.config.chunk_size = 8  # Number of steps to predict in one forward pass
        self.policy.config.output_features["action"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(2,)  # Assuming action has 2 dimensions; adjust as necessary
        )

    def prepare_input_tokens(self, batch):
        batch_size = len(batch["captions"])
        # Get task description (pi05 processor handles all text formatting)
        tasks = batch.get("task", ["Pick up the object"] * batch_size)
        if isinstance(tasks, str):
            tasks = [tasks] * batch_size
        elif len(tasks) == 1:
            tasks = tasks * batch_size

        # Use pi05 state and input tokenizer logic (same as Pi05PrepareStateTokenizerProcessorStep)
        state = batch["observation.state"]
        # state = deepcopy(state)

        # Prepare state (pad to max_state_dim)
        # from lerobot.policies.pi05.modeling_pi05 import pad_vector

        # state = pad_vector(state, DUMMY_STATE_DIM)

        # Normalize state to [-1, 1] range if needed (assuming it's already normalized from normalize_inputs)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        if state.dtype in [torch.float16, torch.bfloat16]:
            state = state.float()  # Convert to float for digitization
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Create pi05-formatted prompts that include state information
        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        # Tokenize with max_length padding to match OpenPI's expected format
        tokenized = self.tokenizer(
            full_prompts,
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt",
        )
        lang_tokens = tokenized["input_ids"].to(device=self.device)
        lang_masks = tokenized["attention_mask"].to(device=self.device, dtype=torch.bool)
        return lang_tokens, lang_masks

    def _project_state_actions(self, batch):
        # Project the state and actions to the expected format of the policy
        batch["observation.state"] = batch.pop("state")
        batch["task"] = batch["captions"]

        batch["observation.language.tokens"], batch["observation.language.attention_mask"] = self.prepare_input_tokens(batch)
        return batch

class VLAGR00T(VLA):
    def __init__(self, cfg, device=torch.device("cuda"), dtype=torch.bfloat16):
        super(VLAGR00T, self).__init__(cfg, device=device, dtype=dtype)

    def _init_backbone(self):
        # todo: replace with actual GR00T policy when available
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
        self.policy = GrootPolicy.from_pretrained("nvidia/GR00T-N1.5-3B").to(device=self.device, dtype=self.dtype)

        # normalization handled by dataloaders
        self.policy.normalize_inputs = nn.Identity()  # Disable input normalization
        self.policy.normalize_targets = nn.Identity()  # Disable target normalization
        self.policy.unnormalize_outputs = nn.Identity()  # Disable output unnormalization
        # Define the expected input features for the policy
        self.policy.config.input_features["frames"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224)
        )
        self.policy.config.input_features.pop("observation.images.camera1", None)  # Remove old camera feature
        self.policy.config.input_features.pop("observation.images.camera2", None)  # Remove old camera feature
        self.policy.config.input_features.pop("observation.images.camera3", None)  # Remove old camera feature
        self.policy.config.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(2,)  # Assuming state has 2 dimensions; adjust as necessary
        )
        self.policy.config.chunk_size = 8  # Number of steps to predict in one forward pass
        self.policy.config.output_features["action"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(2,)  # Assuming action has 2 dimensions; adjust as necessary
        )

        self.groot_preprocessor, self.groot_postprocessor =  make_groot_pre_post_processors(self.policy.config)  # Create pre/post processors based on policy config

    def _project_state_actions(self, batch):
        # Project the state and actions to the expected format of the policy
        batch["observation.state"] = batch.pop("state")
        batch["observation.images.ego_view"] = batch.pop("frames")
        batch["task"] = batch["captions"]
        batch = self.groot_preprocessor(batch)  # Apply GR00T-specific preprocessing to align with expected input format
        return batch


if __name__ == "__main__":
    # Example usage
    from data.language_table.dataset import load_language_table
    cfg = {}
    # vla_model = VLAPiZeroFive(cfg)
    vla_model = VLAGR00T(cfg)
    tasks = ["all"]
    dl = load_language_table(
        tasks=tasks,
        mode="train",
        frames_per_episode=50,
        horizon=8,
        threshold=0.1,
        cutoff_type="single",
        batch_size=2,
        shuffle_buffer_size=1000,
        enforce_caption_uniqueness=True,
        initial_buffer_size=500,
        max_buffer_size=2000,
        buffer_growth_factor=1.5,
        num_workers=1
    )
    # Train Test
    # example_batch = next(iter(dl))
    # example_batch = vla_model.process_batch(example_batch)
    # loss_dict = vla_model(**example_batch)
    # print(loss_dict)
    # Generate Test
    example_batch = next(iter(dl))
    example_batch = vla_model.process_batch(example_batch)
    generated, _ = vla_model.generate_and_decode(example_batch["batch"])
    print(generated["actions"].shape)
