from pathlib import Path
import torch
from tqdm import tqdm
from torch import nn
import webdataset as wds
import wandb

from utils import prepare_inputs_for_inference, batch_of_dicts_to_outer_dict
from data.language_table.preprocess import DIR_PATH
from data.language_table.preference_dataset import load_language_table_preferences


class PreferenceScoreInterface(nn.Module):
    def forward(self, trajectory, text, observation, instruction):
        raise NotImplementedError("Subclasses must implement this method.")

class PreferenceScoreDummy(PreferenceScoreInterface):
    def forward(self, trajectory, text, observation, instruction):
        score = torch.rand(1, device=observation.device)
        # Dummy implementation: always return the same score
        return score

class PreferenceScoreCLIP(PreferenceScoreInterface):
    def __init__(self, cfg, contrastive_model):
        super().__init__()
        self.cfg = cfg

        self.contrastive_model = contrastive_model

    def forward(self, trajectory, text, observation, instruction):
        score = self.contrastive_model.score(trajectory, text, observation, instruction)
        return score


class PreferenceGenerator():
    def __init__(self, cfg, contrastive_model) -> None:
        self.cfg = cfg
        if cfg.stage == "local_dev":
            print("Running in local_dev mode. Initializing dummy PreferenceScore...")
            self.preference_score = PreferenceScoreDummy()
        else:
            self.preference_score = PreferenceScoreCLIP(cfg=self.cfg, contrastive_model=contrastive_model)
        self.K = self.cfg.training.K # pylint: disable=invalid-name

    def get_preferred(self, trajectories, observation, instruction, wds=False):
        scores = torch.zeros(len(trajectories))
        for i, _ in enumerate(trajectories):
            actions = trajectories[i]["actions"]
            if isinstance(trajectories[i]["tokens"], list):
                # If tokens is a list, take the first one
                generated_text = trajectories[i]["tokens"][0]
            else:
                # If tokens is a single string, use it directly
                generated_text = trajectories[i]["tokens"]

            scores[i] = self.preference_score(
                actions, generated_text, observation, instruction)

        best = torch.argmax(scores)
        worst = torch.argmin(scores)

        if wds:
            return  {
                "observation.npy": observation.numpy(),
                "instruction.txt": instruction,
                "preferred_action.npy": trajectories[best]["actions"].to(torch.float32).numpy(force=True),
                "preferred_caption.txt": trajectories[best]["tokens"][0],
                "rejected_action.npy": trajectories[worst]["actions"].to(torch.float32).numpy(force=True),
                "rejected_caption.txt": trajectories[worst]["tokens"][0]
            }
        else:

            preference_dict = {
                "chosen": {
                    "action": trajectories[best]["actions"],
                    "captions": trajectories[best]["tokens"],
                    "frames": observation,
                    "instruction": instruction
                },
                "rejected": {
                    "action": trajectories[worst]["actions"],
                    "captions": trajectories[worst]["tokens"],
                    "frames": observation,
                    "instruction": instruction
                }
            }

            if isinstance(preference_dict["chosen"]["captions"], list):
                preference_dict["chosen"]["captions"] = preference_dict["chosen"]["captions"][0]
            if isinstance(preference_dict["rejected"]["captions"], list):
                preference_dict["rejected"]["captions"] = preference_dict["rejected"]["captions"][0]

            return preference_dict

    def generate_preference_dataset(self, model, dataset, epoch=0):
        mode = dataset.subset
        horizon = dataset.horizon
        base_dir = f"{DIR_PATH}/preference/{wandb.run.id}/epoch_{epoch}_h_{horizon}/{mode}/"
        Path(base_dir).mkdir(exist_ok=True, parents=True)
        tar_path = base_dir + "data-%06d.tar"
        sink = wds.ShardWriter(tar_path, maxcount=1000)

        c = 0
        for batch in tqdm(dataset):
            model_inputs = model.process_batch(batch)
            b = model_inputs["input_ids"].shape[0]
            model_inputs = prepare_inputs_for_inference(model_inputs)
            for bi in range(b):
                sample = {k: (v[bi].unsqueeze(0) if torch.is_tensor(v[bi]) else [v[bi]]) for k, v in model_inputs.items()}
                trajectories = []
                for _ in range(self.K):
                    decoded_outputs, _ = model.generate_and_decode(**sample)
                    trajectories.append(decoded_outputs)
                preference_sample = self.get_preferred(trajectories, batch["frames"][bi], batch["instruction"][bi], wds=True)
                preference_sample["__key__"] = f"epoch_{epoch}_sample_{c:08d}"
                sink.write(preference_sample)
                c += 1
        sink.close()
        # load updated preference dataset
        dataset = load_language_table_preferences(root=base_dir, mode=mode, frames_per_episode=dataset.frames_per_episode, horizon=dataset.horizon, buffer_size=dataset.shuffle_buffer_size)
        return dataset

    @torch.no_grad()
    def generate_preference_from_batch(self, model, batch):
        b = len(batch["frames"])
        chosen = []
        rejected = []
        for bi in range(b):
            sample = self.get_sample_from_batch(batch, bi)
            trajectories = []
            for _ in range(self.K):
                decoded_outputs, _ = model.generate_and_decode(sample)
                trajectories.append(decoded_outputs)
            preference_sample = self.get_preferred(trajectories, batch["frames"][bi], batch["instruction"][bi])

            chosen.append(preference_sample["chosen"])
            rejected.append(preference_sample["rejected"])

        return {
            "chosen": batch_of_dicts_to_outer_dict(chosen),
            "rejected": batch_of_dicts_to_outer_dict(rejected)
        }

    def get_sample_from_batch(self, batch_inputs, index):
        """
        Get a single sample from the batch at the specified index.
        """
        sample = {}
        for key, value in batch_inputs.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value[index].unsqueeze(0)
            elif isinstance(value, list):
                sample[key] = [value[index]]
            elif isinstance(value, dict):
                sample[key] = {k: (v[index].unsqueeze(0) if torch.is_tensor(v[index]) else [v[index]]) for k, v in value.items()} # Handle nested dictionaries, could probably be recursive
        return sample
