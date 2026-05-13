import io
from glob import glob
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import webdataset as wds
from einops import rearrange
import numpy as np

from utils import ResetAwareDataLoader

class CaptionAwareLanguageTable(IterableDataset):
    def __init__(self,
                 # Core parameters
                 shuffle_buffer_size=None, shuffle=False, mode="train", tasks=None,
                 frames_per_episode=-1, horizon=1, threshold=0.05, root=None, cutoff_type="single",
                 batch_size=64,
                 # Caption deduplication parameters
                 enforce_caption_uniqueness=True,
                 # Dynamic buffer parameters
                 initial_buffer_size=500, max_buffer_size=5000, buffer_growth_factor=1.5, caption_buffer_target=100, **kwargs):

        super().__init__()
        self.mode = mode
        self.frames_per_episode = frames_per_episode
        self.horizon = horizon
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.threshold = threshold
        self.cutoff_type = cutoff_type
        self.enforce_caption_uniqueness = enforce_caption_uniqueness
        self.batch_size = batch_size

        # Dynamic buffer management
        self.initial_buffer_size = initial_buffer_size
        self.max_buffer_size = max_buffer_size
        self.buffer_growth_factor = buffer_growth_factor
        self.current_buffer_target = initial_buffer_size
        self.caption_buffer_target = caption_buffer_target  # Minimum unique captions to maintain

        self.data_root = root or f"{Path(__file__).parent.resolve()}/preference"
        self.tar_paths = []

        self.norm_keys = {
            "min": torch.tensor([-0.21989956, -0.23478234]).unsqueeze(0),
            "max": torch.tensor([0.23207834, 0.24496803]).unsqueeze(0)
        }

        if tasks is None:
            raise ValueError("You must specify `tasks`.")

        self.tasks = tasks

        # Caption-based buffer management
        self.caption_buffers = {}  # caption -> list of samples
        self.used_captions_in_batch = set()  # Track captions used in current batch
        self.current_episode_iter = None
        self.episodes_processed = 0

        # Build tar paths
        for task in tasks:
            task_path = f"{self.data_root}/{task}/{self.mode}/data-*.tar"
            task_files = glob(task_path)
            if task_files:
                self.tar_paths.extend(task_files)

        # Build pipeline
        if self.shuffle and self.shuffle_buffer_size is not None:
            self.pipeline = (
                wds.WebDataset(self.tar_paths)
                .shuffle(self.shuffle_buffer_size)
                .to_tuple("instruction.txt", "caption.txt", "frames.npy", "action.npy", "effector_translation.npy", "__key__")
                .map(self.process_sample)
            )
        else:
            if self.shuffle and self.shuffle_buffer_size is None:
                print("You must specify `shuffle_buffer_size` if you want to shuffle the dataset. "
                      "Falling back to non-shuffled dataset.")
            self.pipeline = (
                wds.WebDataset(self.tar_paths)
                .to_tuple("instruction.txt", "caption.txt", "frames.npy", "action.npy", "effector_translation.npy", "__key__")
                .map(self.process_sample)
            )

    def min_max_normalize(self, x):
        return 2 * (x - self.norm_keys["min"]) / (self.norm_keys["max"] - self.norm_keys["min"]) - 1

    def _infer_task_from_key(self, key):
        """Infer task from the webdataset key/path"""
        for task in self.tasks:
            if task in key:
                return task
        return self.tasks[0]

    def process_sample(self, sample):
        instruction, caption, frames, actions, state, key = sample
        instruction = instruction.decode("utf-8")
        caption = caption.decode("utf-8")

        actions = np.load(io.BytesIO(actions))
        actions = rearrange(torch.from_numpy(actions), "(n a) -> n a", a=2)
        actions = self.min_max_normalize(actions)
        steps = actions.shape[0]
        frames = torch.from_numpy(np.load(io.BytesIO(frames)))
        state = np.load(io.BytesIO(state))
        state = rearrange(torch.from_numpy(state), "(n a) -> n a", a=2)

        if self.frames_per_episode > 0:
            number_to_sample = min(steps, self.frames_per_episode)
            indices = torch.linspace(start=0, end=steps-2, steps=number_to_sample, dtype=torch.int)
            captions = [caption] * number_to_sample
            instructions = [instruction] * number_to_sample
            frames = torch.index_select(frames, 0, indices)
            state = torch.index_select(state, 0, indices)

            padded_actions = F.pad(input=actions, pad=(0, 0, 0, self.horizon), mode="constant", value=0)
            indices_with_horizon = torch.stack([indices + h for h in range(self.horizon)], dim=-1)
            actions = torch.stack([torch.index_select(padded_actions, 0, idx_h) for idx_h in indices_with_horizon])
        else:
            captions = [caption] * steps
            instructions = [instruction] * steps

        episode_id = hash(caption)

        # Base sample data
        sample_data = {
            "captions": captions,
            "instruction": instructions,
            "frames": frames,
            "action": actions,
            "episode_id": episode_id,
            "caption_id": caption,  # Store the actual caption for deduplication
            "task": self._infer_task_from_key(key),
            "state": state
        }

        return sample_data

    def _extract_valid_samples_from_episode(self, episode_batch):
        """Extract all valid samples from an episode"""
        actions = episode_batch["action"]
        # Only check the first timestep of each horizon window
        if len(actions.shape) == 3:  # (n, horizon, action_dim)
            first_timestep_actions = actions[:, 0, :]  # (n, action_dim)
        else:  # (n, action_dim)
            first_timestep_actions = actions

        legal_mask = torch.abs(first_timestep_actions) >= self.threshold

        if self.cutoff_type == "single":
            legal_mask = torch.any(legal_mask, dim=-1)
        else:
            legal_mask = torch.all(legal_mask, dim=-1)

        legal_indices = torch.nonzero(legal_mask, as_tuple=True)[0]

        if len(legal_indices) == 0:
            return []

        valid_samples = []
        for i, idx in enumerate(legal_indices):
            sample = {
                "captions": episode_batch["captions"][idx],
                "instruction": episode_batch["instruction"][idx],
                "frames": episode_batch["frames"][idx],
                "action": episode_batch["action"][idx],
                "state": episode_batch["state"][idx],
                "episode_id": episode_batch["episode_id"],
                "caption_id": episode_batch["caption_id"],
                "task": episode_batch["task"],
            }

            valid_samples.append(sample)

        return valid_samples

    def _refill_caption_buffers(self):
        """Refill caption buffers with new episodes"""
        target_samples = self.current_buffer_target
        current_samples = sum(len(samples) for samples in self.caption_buffers.values())

        current_captions = len(self.caption_buffers)
        target_captions = self.caption_buffer_target
        # this refills the buffer if we have too few samples
        while current_samples < target_samples and self.current_buffer_target <= self.max_buffer_size or current_captions < target_captions:
            try:
                if self.current_episode_iter is None:
                    self.current_episode_iter = iter(self.pipeline)

                episode_batch = next(self.current_episode_iter)
                valid_samples = self._extract_valid_samples_from_episode(episode_batch)

                if valid_samples:
                    # Group samples by episode
                    episode_id = valid_samples[0]["episode_id"]

                    if episode_id not in self.caption_buffers:
                        self.caption_buffers[episode_id] = []

                    self.caption_buffers[episode_id].extend(valid_samples)
                    current_samples += len(valid_samples)
                    self.episodes_processed += 1
                    current_captions = len(self.caption_buffers)

            except StopIteration:
                self.current_episode_iter = None
                break

        if current_samples > self.current_buffer_target:
        # # Dynamically grow sample buffer if caption diversity is high
        # if len(self.caption_buffers) >= self.caption_buffer_target:
        #     old_target = self.current_buffer_target
            self.current_buffer_target = min(
                int(self.current_buffer_target * self.buffer_growth_factor),
                self.max_buffer_size
            )
        #     if self.current_buffer_target > old_target:
        #         print(
        #             f"Grew sample buffer target from {old_target} to {self.current_buffer_target} "
        #             f"(unique captions: {len(self.caption_buffers)}, "
        #             f"min required: {self.caption_buffer_target})"
        #         )

    def _sample_batch_with_unique_captions(self, batch_size):
        """Sample a batch ensuring each caption appears only once"""
        if not self.enforce_caption_uniqueness:
            # Fallback to simple random sampling
            all_samples = []
            for caption_samples in self.caption_buffers.values():
                all_samples.extend(caption_samples)

            if len(all_samples) < batch_size:
                return all_samples

            if self.shuffle:
                indices = torch.randperm(len(all_samples))[:batch_size]
                return [all_samples[i] for i in indices]
            else:
                return all_samples[:batch_size]

        # Sample with caption uniqueness
        available_captions = [caption for caption, samples in self.caption_buffers.items()
                            if len(samples) > 0 and caption not in self.used_captions_in_batch]

        if len(available_captions) == 0:
            # No more unique captions available, reset and continue
            self.used_captions_in_batch.clear()
            available_captions = [caption for caption, samples in self.caption_buffers.items()
                                if len(samples) > 0]

        # Limit batch size to available unique captions
        actual_batch_size = min(batch_size, len(available_captions))

        if actual_batch_size == 0:
            return []

        # Sample captions for this batch
        if self.shuffle:
            caption_indices = torch.randperm(len(available_captions))[:actual_batch_size]
            selected_captions = [available_captions[i] for i in caption_indices]
        else:
            selected_captions = available_captions[:actual_batch_size]

        # Sample one sample per selected caption
        batch_samples = []
        for caption in selected_captions:
            caption_samples = self.caption_buffers[caption]

            if self.shuffle and len(caption_samples) > 1:
                # Random sample from this caption
                sample_idx = torch.randint(0, len(caption_samples), (1,)).item()
                sample = caption_samples.pop(sample_idx)
            else:
                # Take first sample (FIFO)
                sample = caption_samples.pop(0)

            batch_samples.append(sample)
            self.used_captions_in_batch.add(caption)

            # Remove empty caption buffers
            if not caption_samples:
                del self.caption_buffers[caption]

        return batch_samples

    def get_buffer_stats(self):
        """Get statistics about current buffer state"""
        total_samples = sum(len(samples) for samples in self.caption_buffers.values())
        return {
            "unique_captions": len(self.caption_buffers),
            "total_samples": total_samples,
            "avg_samples_per_caption": total_samples / max(len(self.caption_buffers), 1),
            "current_buffer_target": self.current_buffer_target,
            "episodes_processed": self.episodes_processed,
            "used_captions_in_current_batch": len(self.used_captions_in_batch)
        }

    def reset_episodes(self):
        # Reset state
        self.caption_buffers = {}
        self.used_captions_in_batch = set()
        self.current_buffer_target = self.initial_buffer_size
        self.episodes_processed = 0

    def __iter__(self):
        self.reset_episodes()
        self.current_episode_iter = None
        sample_count = 0
        while True:
            # Refill buffers
            self._refill_caption_buffers()

            # Use batch sampling to get samples with unique captions
            # TODO: issue, this only samples a single sample per caption
            batch_samples = self._sample_batch_with_unique_captions(batch_size=self.batch_size)

            if not batch_samples:
                break

            # Yield individual samples from the batch
            if self.shuffle:
                indices = torch.randperm(len(batch_samples))
                for idx in indices:
                    yield batch_samples[idx]
                    sample_count += 1
            else:
                for sample in batch_samples:
                    yield sample
                    sample_count += 1

            # Print stats periodically
            if sample_count % 1000 == 0:
                stats = self.get_buffer_stats()
                print(f"Yielded {sample_count} samples. Buffer stats: {stats}")

class CaptionUniqueCollate:
    """Collate function that ensures caption uniqueness within batches"""

    def __init__(self, enforce_uniqueness=True):
        self.enforce_uniqueness = enforce_uniqueness

    def __call__(self, batch):
        if not batch:
            return {}

        # Ensure caption uniqueness if requested
        if self.enforce_uniqueness:
            seen_captions = set()
            unique_batch = []

            for item in batch:
                caption = item["caption_id"]
                if caption not in seen_captions:
                    seen_captions.add(caption)
                    unique_batch.append(item)

            batch = unique_batch

        if not batch:
            return {}

        # Standard collation
        collated = {}
        for key in batch[0].keys():
            if key in ["frames", "action", "state"]:
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]

        # Add minimal batch-level metadata
        collated["batch_caption_count"] = len(set(collated["caption_id"]))
        collated["batch_size"] = len(batch)

        return collated

class CaptionCollate:
    """Simple collate function for caption datasets."""

    def __call__(self, batch):
        if not batch:
            return {}

        collated = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(values)
                except Exception as e:
                    raise ValueError(
                        f"Error stacking '{key}': {[v.shape for v in values]}"
                    ) from e
            else:
                collated[key] = values
        # Add metadata
        collated["batch_size"] = len(batch)
        collated["batch_caption_count"] = (
            len(set(collated["caption_id"])) if "caption_id" in collated else 0
        )

        return collated

def load_caption_aware_language_table(enforce_caption_uniqueness=True, **kwargs):
    """Convenience function to create caption-aware dataloader"""

    # Create dataset
    ds = CaptionAwareLanguageTable(
        enforce_caption_uniqueness=enforce_caption_uniqueness,
        **kwargs
    )

    # Create collate function
    # collate_fn = CaptionUniqueCollate(enforce_uniqueness=enforce_caption_uniqueness)
    collate_fn = CaptionCollate()

    return ds, collate_fn

def load_language_table(tasks=None, enforce_caption_uniqueness=True, **kwargs):
    """Convenience function to create dataloader for specific learning mode"""
    batch_size = kwargs.get("batch_size", 64)
    # Create dataset
    ds, collate_fn = load_caption_aware_language_table(tasks=tasks, enforce_caption_uniqueness=enforce_caption_uniqueness, caption_buffer_target=batch_size*2, **kwargs)

        # Create dataloader with appropriate collate function
    dl = ResetAwareDataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=kwargs.get("num_workers", 4),
        pin_memory=kwargs.get("pin_memory", True),
        persistent_workers=kwargs.get("persistent_workers", True),
        shuffle=kwargs.get("shuffle", False),
    )
    return dl

if __name__ == "__main__":
    # Example usage
    tasks = ["all"]
    dl = load_language_table(
        tasks=tasks,
        mode="train",
        frames_per_episode=50,
        horizon=8,
        threshold=0.1,
        cutoff_type="single",
        batch_size=64,
        shuffle_buffer_size=1000,
        enforce_caption_uniqueness=True,
        initial_buffer_size=500,
        max_buffer_size=2000,
        buffer_growth_factor=1.5,
        num_workers=1
    )

    for i, batch in enumerate(dl):
        if i % 10 == 0:
            print(f"Batch {i}:")
            print(f"  Batch size: {batch['batch_size']}")
            print(f"  Unique captions in batch: {batch['batch_caption_count']}")
            print(f"  Frames shape: {batch['frames'].shape}")
            print(f"  Action shape: {batch['action'].shape}")
        if i >= 50:
            break
