# utils/network_flow_visualization.py

"""
Utilities for extracting and visualizing intermediate activations (network flow)
from models, particularly for transformer-based architectures.
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


class ActivationExtractor:
    """
    Extracts intermediate activations from specified layers of a model using
    forward hooks.

    Usage:
        extractor = ActivationExtractor(model, ['block.0', 'block.5'])
        activations = extractor.extract(inputs)
        extractor.remove_hooks()
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.activations = {}
        self.hooks = []

    def _get_hook(self, name: str):
        def hook(model, input, output):
            # Detach and move to CPU to avoid holding GPU memory
            self.activations[name] = output.detach().cpu()

        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on all transformer blocks."""
        self.remove_hooks()  # Ensure no old hooks are present
        if hasattr(self.model, "net") and hasattr(self.model.net, "backbone"):
            backbone = self.model.net.backbone
            if hasattr(backbone, "blocks"):
                for i, block in enumerate(backbone.blocks):
                    hook = block.register_forward_hook(
                        self._get_hook(f"block_{i}")
                    )
                    self.hooks.append(hook)

    def extract_activations(
        self, inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass and returns the captured activations.
        Hooks are automatically registered and removed.
        """
        self.register_hooks()
        self.activations = {}
        self.model.eval()
        with torch.no_grad():
            _ = self.model(inputs.to(self.device))
        self.remove_hooks()
        return self.activations

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def visualize_activations(
    activations: Dict[str, torch.Tensor],
    save_path: str,
    num_patches_to_show: int = 50,
) -> None:
    """
    Visualizes the mean activation magnitude across patches for each block.

    Args:
        activations: Dictionary of activations from ActivationExtractor.
        save_path: Path to save the visualization.
        num_patches_to_show: Number of patch tokens to visualize.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    block_names = sorted(activations.keys(), key=lambda x: int(x.split("_")[1]))
    mean_activations = []

    for name in block_names:
        # Shape: [B, N, D], where N is num_patches + 1 (for CLS)
        act = activations[name]
        # Take first item in batch, ignore CLS token, take mean over feature dim
        mean_act_per_patch = act[0, 1:, :].abs().mean(dim=-1).numpy()
        mean_activations.append(mean_act_per_patch)

    # Create a heatmap
    activation_matrix = np.array(mean_activations)
    num_patches_total = activation_matrix.shape[1]
    step = max(1, num_patches_total // num_patches_to_show)
    subset_indices = np.arange(0, num_patches_total, step)
    im = ax.imshow(
        activation_matrix[:, subset_indices],
        cmap="viridis",
        aspect="auto",
    )

    ax.set_yticks(np.arange(len(block_names)))
    ax.set_yticklabels(block_names)
    ax.set_xlabel("Patch Index (Subsampled)")
    ax.set_ylabel("Transformer Block")
    ax.set_title("Mean Activation Magnitude Across Patches")

    fig.colorbar(im, ax=ax, label="Mean Absolute Activation")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()