# utils/attention_visualization.py

"""
Attention visualization utilities for Vision Transformers in continual learning.
This module provides functions to visualize and analyze attention maps from ViT models.
"""
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


class AttentionAnalyzer:
    """
    Analyzes attention patterns in a Vision Transformer model by leveraging its
    native ability to return attention scores.
    """

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        # Ensure model is on the correct device
        self.model.to(device)

    def extract_attention_maps(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """
        Extracts attention maps for given inputs by calling the ViT backbone's
        forward pass with `return_attention_scores=True`.

        Args:
            inputs: Input tensor of shape [B, C, H, W].

        Returns:
            A list of attention map tensors, one from each block.
            Each tensor has shape [B, num_heads, N, N].
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)

            # Navigate through the model hierarchy to reach the ViT backbone
            try:
                # For ContinualModel -> backbone structure
                if hasattr(self.model, "net") and hasattr(
                    self.model.net, "backbone"
                ):
                    backbone = self.model.net.backbone
                elif hasattr(self.model, "net"):
                    backbone = self.model.net
                elif hasattr(self.model, "backbone"):
                    backbone = self.model.backbone
                else:
                    backbone = self.model

                # Check if the backbone supports attention score extraction
                if (
                    hasattr(backbone, "forward")
                    and "return_attention_scores"
                    in backbone.forward.__code__.co_varnames
                ):
                    output, attn_maps = backbone(
                        inputs, return_attention_scores=True
                    )
                    return attn_maps
                else:
                    print(
                        "Warning: Backbone does not support return_attention_scores"
                    )
                    return []

            except Exception as e:
                print(f"Warning: Could not extract attention maps: {e}")
                return []


def visualize_attention_map(
    attention_map: torch.Tensor,
    input_image: torch.Tensor,
    head_idx: int = 0,
    layer_name: str = "",
    save_path: Optional[str] = None,
    patch_size: int = 16,
) -> None:
    """
    Visualizes the attention from the CLS token to all patch tokens for a
    specific head, overlaid on the input image.
    """
    # Take first sample in batch and specified head
    attn = attention_map[0, head_idx].cpu().numpy()  # Shape: (N, N)
    # Get attention from CLS token (row 0) to all patch tokens (cols 1:)
    cls_attention = attn[0, 1:]

    # Reshape attention to spatial dimensions
    h, w = input_image.shape[2], input_image.shape[3]
    num_patches_h, num_patches_w = h // patch_size, w // patch_size
    attention_spatial = cls_attention.reshape(num_patches_h, num_patches_w)

    # Prepare image for display (denormalize)
    img_np = input_image[0].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2615])
    img_np = np.clip(img_np * std + mean, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"CLS Token Attention: {layer_name}, Head {head_idx}", fontsize=16
    )

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(attention_spatial, cmap="hot")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(attention_spatial, cmap="hot", alpha=0.6)
    axes[2].set_title("Attention Overlay")
    axes[2].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def analyze_task_attention(
    model,
    test_loader: DataLoader,
    class_names: List[str],
    device="cuda",
    save_dir: Optional[str] = None,
    samples_per_class: int = 3,
) -> Dict[int, Dict]:
    """
    Analyzes and visualizes attention for a sample of images from a task's
    test set.

    Returns:
        A dictionary mapping class index to a dictionary of sample inputs
        and their extracted attention maps.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    analyzer = AttentionAnalyzer(model, device)
    # The test_loader is now passed in directly, already configured for the correct task.

    # Collect samples for each class in the current task
    class_samples = {}
    for inputs, labels, _ in test_loader:
        for i in range(inputs.shape[0]):
            label = labels[i].item()
            if label not in class_samples:
                class_samples[label] = []
            if len(class_samples[label]) < samples_per_class:
                class_samples[label].append(inputs[i])
        # Check if we have enough samples
        if all(
            len(v) == samples_per_class
            for v in class_samples.values()
            if v is not None
        ):
            break

    # Analyze and visualize attention for collected samples
    analyzed_data = {}
    for class_idx, samples in class_samples.items():
        class_name = class_names[class_idx]
        analyzed_data[class_idx] = {"inputs": [], "maps": []}
        for i, sample_input in enumerate(samples):
            sample_input_batch = sample_input.unsqueeze(0)
            attention_maps = analyzer.extract_attention_maps(sample_input_batch)
            analyzed_data[class_idx]["inputs"].append(sample_input)
            analyzed_data[class_idx]["maps"].append(attention_maps)

            if save_dir and attention_maps:
                for block_idx, attn_map in enumerate(attention_maps):
                    # Visualize first 4 heads
                    for head_idx in range(min(4, attn_map.shape[1])):
                        save_path = os.path.join(
                            save_dir,
                            f"class_{class_name}_sample_{i}",
                            f"block_{block_idx}_head_{head_idx}.png",
                        )
                        visualize_attention_map(
                            attn_map,
                            sample_input_batch,
                            head_idx=head_idx,
                            layer_name=f"Block {block_idx}",
                            save_path=save_path,
                        )
    return analyzed_data