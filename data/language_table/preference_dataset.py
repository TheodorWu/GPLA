import io
from glob import glob
import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import numpy as np
import webdataset as wds

from data.language_table.dataset import load_language_table
from tqdm import tqdm

DIR_PATH = Path(__file__).parent.resolve()

class LanguageTablePreferences(IterableDataset): # pylint: disable=abstract-method
    def __init__(self, shuffle_buffer_size=None, shuffle=False, subset="train", frames_per_episode=-1, horizon=1, root=None):
        super().__init__()
        self.subset = subset
        self.frames_per_episode = frames_per_episode
        self.horizon = horizon
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        self.data_root = root
        self.tar_paths = []
        task_path = f"{self.data_root}/{self.subset}/data-*.tar"
        for p in glob(task_path):
            self.tar_paths.append(p)

        if self.shuffle and not self.shuffle_buffer_size is None:
            self.pipeline = (
                wds.WebDataset(self.tar_paths)
                .shuffle(self.shuffle_buffer_size)
                .to_tuple("observation.npy", "instruction.txt", "preferred_action.npy", "preferred_caption.txt", "rejected_action.npy", "rejected_caption.txt")
                .map(self.process_sample)
            )
        else:
            if self.shuffle and self.shuffle_buffer_size is None:
                print("You must specify `shuffle_buffer_size` if you want to shuffle the dataset. Falling back to non-shuffled dataset.")
            self.pipeline = (
                wds.WebDataset(self.tar_paths)
                .to_tuple("observation.npy", "instruction.txt", "preferred_action.npy", "preferred_caption.txt", "rejected_action.npy", "rejected_caption.txt")
                .map(self.process_sample)
            )

    def process_sample(self, sample):
        observation, instruction, preferred_action, preferred_caption, rejected_action, rejected_caption = sample
        instruction = instruction.decode("utf-8")
        preferred_caption = preferred_caption.decode("utf-8")
        rejected_caption = rejected_caption.decode("utf-8")
        preferred_action = torch.from_numpy(np.load(io.BytesIO(preferred_action)))
        rejected_action = torch.from_numpy(np.load(io.BytesIO(rejected_action)))
        frames = torch.from_numpy(np.load(io.BytesIO(observation)))

        sample = {
            "observation": frames,
            "instruction": instruction,
            "preferred_action": preferred_action,
            "preferred_caption": preferred_caption,
            "rejected_action": rejected_action,
            "rejected_caption": rejected_caption
        }
        return sample

    def __iter__(self):
        return iter(self.pipeline)


def language_table_collate(data):
    outer = {}
    first_dictionary = data[0]

    for k in first_dictionary.keys():
        outer[k] = [dictionary[k] for dictionary in data]
    return outer


def load_language_table_preferences(root, mode, frames_per_episode=-1, horizon=1, batch_size=16, buffer_size=100, num_workers=4):
    shuffle = mode == "train"
    ds = LanguageTablePreferences(shuffle_buffer_size=buffer_size, shuffle=shuffle, subset=mode, frames_per_episode=frames_per_episode, horizon=horizon, root=root)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=language_table_collate, pin_memory=True, num_workers=num_workers)
    return dl

def generate_initial_preferences(task, mode, horizon):
    og_dataset = load_language_table(tasks=task, mode=mode, frames_per_episode=-1, horizon=8, buffer_size=100, batch_size=128)
    base_dir = f"{DIR_PATH}/preference/{task}/init_h_{horizon}/{mode}/"

    sink = wds.ShardWriter(base_dir + "data-%06d.tar", maxcount=1000)
    for batch in tqdm(og_dataset):
        b = batch["frames"].shape[0]
        for i in range(b):
            r_i = np.random.randint(0, b)
            if i == r_i:
                r_i = (r_i + 1) % b
            sample = {
                "observation.npy": batch["frames"][i].numpy(),
                "instruction.txt": batch["instruction"][i],
                "preferred_action.npy": batch["actions"][i].to(torch.float32).numpy(force=True),
                "preferred_caption.txt": batch["captions"][i],
                "rejected_action.npy": batch["actions"][r_i].to(torch.float32).numpy(force=True),
                "rejected_caption.txt": batch["captions"][r_i],
            }
            sink.write(sample)
    sink.close()
