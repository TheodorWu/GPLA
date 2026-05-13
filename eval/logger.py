import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from utils import preprocess_image_for_display

class MetricLogger():
    def __init__(self, log_frequency) -> None:
        self.log_frequency = log_frequency
        self.collected_metrics = {}

    def collect_metric(self, value, name, epoch, stage):
        key = f"{stage}/epoch-{epoch}/{name}"
        if key not in self.collected_metrics:
            self.collected_metrics[key] = []
        if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
            value = value.float()
        self.collected_metrics[key].append(value)

    def aggregate_collected_metrics(self):
        aggregated_metrics = {}
        for key, values in self.collected_metrics.items():
            if all(isinstance(v, (int, float, np.number)) for v in values):
                aggregated_metrics[f"{key}_mean"] = np.mean(values)
                aggregated_metrics[f"{key}_std"] = np.std(values)
            elif all(isinstance(v, torch.Tensor) for v in values):
                stacked = torch.stack([v.float() if v.dtype == torch.bfloat16 else v for v in values])
                aggregated_metrics[f"{key}_mean"] = torch.mean(stacked, dim=0)
                aggregated_metrics[f"{key}_std"] = torch.std(stacked, dim=0)
            else:
                aggregated_metrics[key] = values
        self.collected_metrics = {}
        return aggregated_metrics

    def aggregate_and_log_metrics(self):
        print("Aggregating and logging collected metrics...")
        aggregated_metrics = self.aggregate_collected_metrics()
        for key, value in aggregated_metrics.items():
            wandb.log({key: value})
            print(f"Logged {key}: {value}")
        print("Finished aggregating and logging metrics.")

    def log_metric(self, value, name, step, epoch, stage):
        if torch.is_tensor(value) and value.dtype == torch.bfloat16:
            value = value.float()
        if self.log_frequency > 0 and step % self.log_frequency == 0:
            print(f"{stage}/{name} after {step} steps: {value}", file=sys.stdout)
        if step >= 0:
            wandb.log({f"{stage}/{name}": value,
                        f"{stage}/step": step, f"{stage}/epoch": epoch})
        if step >= 0 and stage != "train":
            value = value.item() if torch.is_tensor(value) else value
            self.collect_metric(value, name, epoch, stage)

    def log_table(self, data, name, step, epoch, stage, img_keys=None, str_keys=None, ignore_keys=None):
        cols = list(data.keys())
        if ignore_keys:
            cols = [col for col in cols if col not in ignore_keys]

        rows = []
        for i, _ in enumerate(data[cols[0]]):
            row = []
            for col in cols:
                v = data[col][i]

                if img_keys and col in img_keys:
                    v = wandb.Image((v).to(dtype=torch.uint8))

                if str_keys and col in str_keys:
                    if torch.is_tensor(v):
                        v = v.tolist()
                    v = str(v)

                row.append(v)

            rows.append(row)

        table = wandb.Table(columns=cols, data=rows)
        wandb.log({f"stage_{stage}/epoch_{epoch}/step_{step}/{name}": table})

    def create_confusion_matrix_plot(self,
                               similarity_matrix,
                               text_labels,
                               images,
                               actions,
                               title = "Text-Image+Action Similarity Matrix",
                               figsize = (12, 8),
                               thumbnail_size = (64, 64)) -> None:
        """
        Create a confusion matrix-style plot for text-image similarities.

        Args:
            similarity_matrix: 2D array with similarity scores
            text_labels: Labels for text samples (y-axis)
            image_labels: Labels for image samples (x-axis)
            title: Plot title
            figsize: Figure size

        Returns:
            matplotlib figure object
        """
                # Create a figure with extra space for images
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 2))

        # Create main heatmap subplot
        ax_heatmap = plt.subplot2grid((10, 1), (2, 0), rowspan=8)

        # Create heatmap without x-axis labels initially
        sns.heatmap(similarity_matrix.cpu(),
                    annot=True,
                    fmt='.3f',
                    cmap='viridis',
                    xticklabels=False,  # We'll add images instead
                    yticklabels=text_labels,
                    ax=ax_heatmap,
                    cbar_kws={'label': 'Similarity Score'})

        ax_heatmap.set_ylabel('Text', fontsize=12)
        ax_heatmap.set_xlabel('')

        # Create subplot for image thumbnails
        ax_images = plt.subplot2grid((10, 1), (0, 0), rowspan=1)
        ax_images.set_xlim(0, len(images))
        ax_images.set_ylim(0, 1)
        ax_images.axis('off')

        # Add image thumbnails
        for i, image in enumerate(images):
            action_text = ', '.join([ f"{a:.4f}" for a in actions[i].cpu().numpy().flatten()])
            try:
                # Preprocess image for display
                processed_image = preprocess_image_for_display(image, thumbnail_size)

                # Calculate position
                x_pos = i + 0.5
                y_pos = 0.5

                # Create OffsetImage and AnnotationBbox
                imagebox = OffsetImage(processed_image, zoom=0.5)
                ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False)
                ax_images.add_artist(ab)

                # Add image label below thumbnail
                ax_images.text(x_pos, 0.1, action_text,
                             ha='center', va='top', fontsize=8, rotation=45)

            except Exception as e:
                print(f"Error processing image {i}: {e}")
                # Fall back to text label
                ax_images.text(i + 0.5, 0.5, action_text,
                             ha='center', va='center', fontsize=8, rotation=45)

        # Add title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

        # Add xlabel to bottom
        fig.text(0.5, 0.02, 'Images', ha='center', fontsize=12)

        # Log the plot as an image
        wandb.log({"similarity_matrix": wandb.Image(fig)})

        # Log individual images if provided
        if images is not None:
            wandb_images = []
            for i, image in enumerate(images):
                action_text = ', '.join([ f"{a:.4f}" for a in actions[i].cpu().numpy().flatten()])
                try:
                    processed_image = preprocess_image_for_display(image, (128, 128))
                    wandb_images.append(wandb.Image(processed_image, caption=action_text))
                except Exception as e:
                    print(f"Error logging image {i}: {e}")

            if len(wandb_images) > 0:
                wandb.log({"input_images": wandb_images})

        # Log raw similarity matrix as a table for interactive exploration
        # Convert matrix to wandb Table format
        table_data = []
        for i, text_label in enumerate(text_labels):
            row_data = [text_label] + similarity_matrix[i].tolist()
            table_data.append(row_data)

        columns = ["Text"] + [ str(a.tolist()) for a in actions]
        table = wandb.Table(data=table_data, columns=columns)
        wandb.log({"similarity_table": table})

        # Log summary statistics
        wandb.log({
            "mean_similarity": torch.mean(similarity_matrix).cpu().numpy(),
            "max_similarity": torch.max(similarity_matrix).cpu().numpy(),
            "min_similarity": torch.min(similarity_matrix).cpu().numpy(),
            "std_similarity": torch.std(similarity_matrix).cpu().numpy()
        })

        # Log the matrix as a histogram
        wandb.log({"similarity_distribution": wandb.Histogram(similarity_matrix.cpu().numpy().flatten())})

        # plt.show()  # Show the plot
        plt.close(fig)  # Close the figure to free memory

    def create_hard_negative_figures(self, similarities):
        """Generate figures for hard negative mining"""

        # Extract positive similarities (diagonal elements)
        pos_sims = torch.diagonal(similarities).cpu().numpy()

        # Extract negative similarities (off-diagonal elements)
        mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
        neg_sims = similarities[mask].cpu().numpy()

        # Figure 1: Similarity distribution showing overlap
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot histograms
        axes[0].hist(pos_sims, bins=50, alpha=0.5, label='Positive pairs')
        axes[0].hist(neg_sims, bins=50, alpha=0.5, label='Negative pairs')
        axes[0].axvline(pos_sims.mean(), color='blue', linestyle='--', label='Pos mean')
        axes[0].axvline(neg_sims.mean(), color='red', linestyle='--', label='Neg mean')
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Similarity Distribution (Showing Hard Negatives)')
        axes[0].legend()

        # Plot CDF to show overlap
        axes[1].hist(pos_sims, bins=50, cumulative=True, density=True, alpha=0.5, label='Positive')
        axes[1].hist(neg_sims, bins=50, cumulative=True, density=True, alpha=0.5, label='Negative')
        axes[1].set_xlabel('Cosine Similarity')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('Cumulative Distribution')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('hard_negatives_analysis.pdf')
        wandb.log({"hard_negative_figure": wandb.Image(fig)})
