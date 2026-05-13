import sys, os
from pathlib import Path
import pandas as pd
import wandb

import torch
from tqdm import tqdm
submodule_path = Path(__file__).parent / "openvla_mini"
sys.path.insert(0, str(submodule_path))
# needed only to avoid error for unhandled case of PRISMATIC_DATA_ROOT not being set
os.environ["PRISMATIC_DATA_ROOT"] = "/tmp/data"
os.environ["USE_TORCH"] = "1"
import hydra
from omegaconf import DictConfig #, OmegaConf

import transformers
if hasattr(transformers, 'video_utils'):
    transformers.image_utils.VideoInput = transformers.video_utils.VideoInput
    transformers.image_utils.make_batched_videos = transformers.video_utils.make_batched_videos

from utils import seed_all, BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS#, DIR_PATH
transformers.image_processing_utils_fast.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = BASE_IMAGE_PROCESSOR_FAST_DOCSTRING
transformers.image_processing_utils_fast.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS


from model.initializer import Initializer

from transformers import CLIPProcessor, CLIPModel, AutoModel, Siglip2Processor


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # visualize_model = "Contrastive Grounding Model"
    # visualize_model = "SigLIP 2"
    visualize_model = cfg.visualization.get("model", "CLIP")
    initializer = Initializer(cfg=cfg)
    initializer.init_wandb()
    dtype = initializer.dtype
    device = initializer.device

    # seed after randomly generating run name
    seed_all(cfg.training.get("seed", 42))

    # Loop through test set and then run t-SNE visualization
    test_ds = initializer.init_language_table("test")

    if visualize_model == "CLIP":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=dtype).to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif visualize_model == "SigLIP 2":
        model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", device_map="auto", torch_dtype=dtype).to(device)
        processor = Siglip2Processor.from_pretrained("google/siglip2-base-patch16-224")
    elif visualize_model == "Contrastive Grounding Model":
        model = initializer.init_clip_model()
        processor = None
    else:
        raise ValueError(f"Unknown visualize_model: {visualize_model}")

    model.eval()
    vision_embeddings = []
    text_embeddings = []
    stop_after = cfg.visualization.get("stop_after", 100)
    current_vision_embedding = None
    current_text_embedding = None
    captions = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_ds,
                                        desc=f"Visualizing Embeddings with {visualize_model}",
                                        total=stop_after)):
            if i >= stop_after:
                break
            if not processor:
                inputs = model.process_batch(batch)
                outputs = model(**inputs)

                current_vision_embedding = outputs.action_vision_embeddings
                current_text_embedding = outputs.text_embeddings
            else:
                images = batch["frames"]  # list of PIL images
                texts = batch["captions"]    # list of strings

                inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device=device)
                outputs = model(**inputs)
                current_vision_embedding = outputs.image_embeds
                current_text_embedding = outputs.text_embeds
                captions.extend(texts)
            if current_vision_embedding.dtype != torch.float32:
                current_vision_embedding = current_vision_embedding.float()
            if current_text_embedding.dtype != torch.float32:
                current_text_embedding = current_text_embedding.float()
            vision_embeddings.append(current_text_embedding)
            text_embeddings.append(current_vision_embedding)
    # For example, using t-SNE or another dimensionality reduction technique
    print("Collected vision and text embeddings for visualization.")
        # Prepare embeddings for t-SNE
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Concatenate all embeddings
    vision_embs = torch.cat(vision_embeddings, dim=0).cpu().numpy()
    text_embs = torch.cat(text_embeddings, dim=0).cpu().numpy()

    # Combine vision and text embeddings
    all_embeddings = np.vstack([vision_embs, text_embs])

    # Create labels for coloring
    labels = ['vision'] * len(vision_embs) + ['text'] * len(text_embs)

    # Apply t-SNE
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=cfg.training.get("seed", 42),
                perplexity=min(30, len(all_embeddings) - 1))
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Store IDs and coordinates
    num_vision = len(vision_embs)
    # num_text = len(text_embs)

    # Create a structured array or dictionary to store the data
    datapoint_info = []
    for i in range(len(embeddings_2d)):
        if i < num_vision:
            datapoint_type = 'vision'
            idx_within_type = i
        else:
            datapoint_type = 'text'
            idx_within_type = i - num_vision

        datapoint_info.append({
            'id': i,
            'type': datapoint_type,
            'idx_within_type': idx_within_type,
            'x': embeddings_2d[i, 0],
            'y': embeddings_2d[i, 1],
            'caption': captions[idx_within_type]
        })

    df = pd.DataFrame(datapoint_info)
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    coords_output_path = output_dir / f'tsne_coordinates_{visualize_model.lower()}.csv'
    df.to_csv(coords_output_path, index=False)
    print(f"Saved t-SNE coordinates to {coords_output_path}")

    # Visualize
    plt.figure(figsize=(12, 8))

    # Plot vision embeddings
    vision_mask = np.array(labels) == 'vision'
    vision_label = "vision"
    if visualize_model == "Contrastive Grounding Model":
        vision_label = "action-conditioned Vision"

    plt.scatter(embeddings_2d[vision_mask, 0], embeddings_2d[vision_mask, 1],
                c='blue', label=vision_label, alpha=0.6, s=50)

    # Plot text embeddings
    text_mask = np.array(labels) == 'text'
    plt.scatter(embeddings_2d[text_mask, 0], embeddings_2d[text_mask, 1],
                c='red', label='Text', alpha=0.6, s=50)

    plt.title(f't-SNE Visualization of {visualize_model} Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save figure
    output_path = output_dir / f'tsne_{visualize_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE visualization to {output_path}")

    # Upload to wandb
    wandb.log({"tsne_visualization": wandb.Image(str(output_path))})



if __name__=="__main__":
    main() # pylint: disable=no-value-for-parameter
