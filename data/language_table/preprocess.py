import json
import os
from pathlib import Path
import h5py
import webdataset as wds
from tqdm import tqdm
import torch
from torchvision.io import decode_image
import numpy as np

DIR_PATH = Path(__file__).parent.resolve()


def decode_image_sequence(images):
    decoded_images = torch.stack([ decode_image(torch.tensor(np.frombuffer(img, dtype=np.uint8))) for img in images ]).numpy()
    return decoded_images

def extract_single_task(tasks=None, hdf5_files=None, target_dir=None):
    if not (tasks and hdf5_files):
        print("Please specify target tasks and data files.")
        return

    for mode in ["train", "validation", "test"]:
        base_dir = f"{target_dir}/{mode}/"
        Path(base_dir).mkdir(exist_ok=True, parents=True)

        tar_path = base_dir + "data-%06d.tar"
        sink = wds.ShardWriter(tar_path, maxcount=1000)

        sample_index = 0
        for hdf5_name in hdf5_files:
            hdf5_path = os.path.join(DIR_PATH, hdf5_name)
            with h5py.File(hdf5_path, "r") as hdf5file:

                for cap_id in tqdm(hdf5file[mode].keys()):
                    episode = hdf5file[mode][cap_id]
                    instruction = episode["instruction"][()].decode("utf-8")

                    if instruction not in tasks:
                        continue

                    episode_id = episode["episode_id"][()].decode("utf-8")
                    # Prepare sample components
                    sample = {"__key__": f"{episode_id}_{sample_index:06d}"}

                    sample["instruction.txt"] = instruction

                    if "captions" in episode:
                        captions = episode["captions"][()].decode("utf-8")
                        sample["caption.txt"] = captions

                    if "frames" in episode:
                        image_data = decode_image_sequence(episode["frames"][()])
                        sample["frames.npy"] = image_data

                    if "effector_translation" in episode:
                        sample["effector_translation.npy"] = episode["effector_translation"][()]

                    if "effector_target_translation" in episode:
                        sample["effector_target_translation.npy"] = episode["effector_target_translation"][()]

                    if "action" in episode:
                        sample["action.npy"] = episode["action"][()]

                    sink.write(sample)
                    sample_index += 1

        sink.close()

def extract_n_tasks(n, hdf5_files):
    if n == "all":
        target_dir = f"{DIR_PATH}/preference/all"
    else:
        target_dir = f"{DIR_PATH}/preference/first_{n}"

    with open(f"{DIR_PATH}/captions.json", "r", encoding="utf-8") as f:
        d  = json.load(f)
        episodes = d["episodes"]
        instructions = list(set(map(lambda x: x["long_horizon_instructions"], episodes)))

        if n == "all":
            chosen_tasks = instructions
        else:
            chosen_tasks = instructions[:n]

        extract_single_task(tasks=chosen_tasks, hdf5_files=hdf5_files, target_dir=target_dir)

        with open(f"{target_dir}/tasks.txt", "w", encoding="utf-8") as taskfile:
            for task in chosen_tasks:
                taskfile.write(task + "\n\n")


if __name__=="__main__":
    source_files = [
        "language_table_part1.hdf5",
        "language_table_part2.hdf5",
        "language_table_part3.hdf5",
        "language_table_part4.hdf5"
    ]

    extract_n_tasks("all", source_files)
