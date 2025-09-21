# utils/attention_visualization.py

"""
Attention visualization utilities for Vision Transformers in continual learning.
This module provides functions to visualize and analyze attention maps from ViT models.

OPTIMIZED VERSION: Reduces memory usage and computation time to prevent ViT timeouts.
"""
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


class AttentionAnalyzer:
    """
    OPTIMIZED Attention analyzer for Vision Transformers.

    Key optimizations:
    - Selective layer analysis (only key layers)
    - Memory-efficient batch processing
    - Configurable frequency (not every epoch)
    - Early stopping for large datasets
    """

    def __init__(self, model, device="cuda", max_samples_per_analysis=16,
                 extract_layers=[0, 5, 11], extract_heads=[0, 3, 7]):
        """
        Initialize optimized attention analyzer.

        Args:
            model: The Vision Transformer model
            device: Device for computation
            max_samples_per_analysis: Maximum samples to process per analysis
            extract_layers: Which transformer layers to extract (default: first, middle, last)
            extract_heads: Which attention heads to extract (default: subset)
        """
        self.model = model
        self.device = device
        self.max_samples = max_samples_per_analysis
        self.extract_layers = extract_layers
        self.extract_heads = extract_heads

        # Ensure model is on the correct device
        self.model.to(device)

        # Logging
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 OPTIMIZED AttentionAnalyzer initialized:")
        self.logger.info(f"   - Max samples per analysis: {max_samples_per_analysis}")
        self.logger.info(f"   - Extract layers: {extract_layers}")
        self.logger.info(f"   - Extract heads: {extract_heads}")

    def extract_attention_maps(self, inputs: torch.Tensor,
                             skip_if_large_batch: bool = True) -> List[torch.Tensor]:
        """
        OPTIMIZED: Extract attention maps with memory and computation optimizations.

        Args:
            inputs: Input tensor of shape [B, C, H, W]
            skip_if_large_batch: Skip processing if batch is too large

        Returns:
            A list of attention map tensors from selected layers only.
            Each tensor has shape [B, selected_heads, N, N].
        """
        # OPTIMIZATION: Early exit for large batches to prevent memory issues
        if skip_if_large_batch and inputs.shape[0] > self.max_samples:
            self.logger.debug(f"Skipping attention extraction - batch too large: {inputs.shape[0]} > {self.max_samples}")
            return []

        # OPTIMIZATION: Limit batch size
        if inputs.shape[0] > self.max_samples:
            inputs = inputs[:self.max_samples]
            self.logger.debug(f"Truncated batch to {self.max_samples} samples")

        self.model.eval()
        extracted_maps = []

        with torch.no_grad():
            try:
                inputs = inputs.to(self.device)

                # Navigate through the model hierarchy to reach the ViT backbone
                backbone = self._get_vit_backbone()
                if backbone is None:
                    self.logger.warning("Could not find ViT backbone - skipping attention extraction")
                    return []

                # Inform backbone which layers to extract attention from (optimization)
                try:
                    backbone.attention_extract_layers = self.extract_layers
                except Exception:
                    pass

                # Check if the backbone supports attention score extraction
                if not self._supports_attention_extraction(backbone):
                    self.logger.warning("Backbone does not support attention extraction")
                    return []

                # OPTIMIZED: Extract attention only from selected layers
                self.logger.debug(f"Extracting attention from {len(self.extract_layers)} layers...")

                # Get all attention maps
                output, all_attn_maps = backbone(inputs, return_attention_scores=True)

                # OPTIMIZATION: Extract only selected layers and heads
                for layer_idx in self.extract_layers:
                    if layer_idx < len(all_attn_maps):
                        layer_attn = all_attn_maps[layer_idx]  # [B, num_heads, N, N]

                        # Skip layers where attention was not extracted
                        if layer_attn is None:
                            continue

                        # OPTIMIZATION: Extract only selected heads
                        if len(self.extract_heads) < layer_attn.shape[1]:
                            selected_heads = []
                            for head_idx in self.extract_heads:
                                if head_idx < layer_attn.shape[1]:
                                    selected_heads.append(layer_attn[:, head_idx:head_idx+1, :, :])

                            if selected_heads:
                                layer_attn = torch.cat(selected_heads, dim=1)

                        extracted_maps.append(layer_attn.cpu())  # Move to CPU immediately

                # Clear GPU memory
                del inputs, output, all_attn_maps
                # Remove the temporary attribute if it was set
                if hasattr(backbone, 'attention_extract_layers'):
                    delattr(backbone, 'attention_extract_layers')
                torch.cuda.empty_cache()

                self.logger.debug(f"Successfully extracted attention from {len(extracted_maps)} layers")
                return extracted_maps

            except Exception as e:
                self.logger.error(f"Error extracting attention maps: {e}")
                # Ensure cleanup on error
                torch.cuda.empty_cache()
                return []

    def _get_vit_backbone(self):
        """Navigate model hierarchy to find ViT backbone."""
        try:
            # For ContinualModel -> backbone structure
            if hasattr(self.model, "net") and hasattr(self.model.net, "backbone"):
                return self.model.net.backbone
            elif hasattr(self.model, "net"):
                return self.model.net
            elif hasattr(self.model, "backbone"):
                return self.model.backbone
            else:
                return self.model
        except Exception as e:
            self.logger.debug(f"Error navigating model hierarchy: {e}")
            return None

    def _supports_attention_extraction(self, backbone) -> bool:
        """Check if backbone supports attention extraction."""
        try:
            return (hasattr(backbone, "forward") and
                   "return_attention_scores" in backbone.forward.__code__.co_varnames)
        except:
            return False


def visualize_attention_map(
    attention_map: torch.Tensor,
    input_image: torch.Tensor,
    head_idx: int = 0,
    layer_name: str = "",
    save_path: Optional[str] = None,
    patch_size: int = 16,
) -> None:
    """
    OPTIMIZED: Visualizes attention map with memory-efficient processing.
    """
    try:
        # Take first sample in batch and specified head
        attn = attention_map[0, head_idx].cpu().numpy()  # Shape: (N, N)
        # Get attention from CLS token (row 0) to all patch tokens (cols 1:)
        cls_attention = attn[0, 1:]

        # Reshape attention to spatial dimensions
        h, w = input_image.shape[2], input_image.shape[3]
        num_patches_h, num_patches_w = h // patch_size, w // patch_size

        # OPTIMIZATION: Check if reshaping is valid
        expected_patches = num_patches_h * num_patches_w
        if len(cls_attention) != expected_patches:
            print(f"Warning: Attention size mismatch. Expected {expected_patches}, got {len(cls_attention)}")
            return

        attention_spatial = cls_attention.reshape(num_patches_h, num_patches_w)

        # Prepare image for display (denormalize)
        img_np = input_image[0].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2615])
        img_np = np.clip(img_np * std + mean, 0, 1)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"CLS Token Attention: {layer_name}, Head {head_idx}", fontsize=16)

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
            plt.savefig(save_path, dpi=150, bbox_inches="tight")  # Reduced DPI for speed
            plt.close()
        else:
            plt.show()

    except Exception as e:
        print(f"Error visualizing attention: {e}")
        if 'fig' in locals():
            plt.close(fig)


def analyze_task_attention(
    model,
    test_loader: DataLoader,
    class_names: List[str],
    device="cuda",
    save_dir: Optional[str] = None,
    samples_per_class: int = 2,  # REDUCED from 3
    max_total_samples: int = 32,  # OPTIMIZATION: Cap total samples
    analysis_frequency: int = 5   # OPTIMIZATION: Only analyze every 5th epoch
) -> Dict[int, Dict]:
    """
    OPTIMIZED: Analyze and visualize attention with strict limits to prevent timeouts.
    """
    import logging
    logger = logging.getLogger(__name__)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # OPTIMIZATION: Use the optimized analyzer
    analyzer = AttentionAnalyzer(model, device, max_samples_per_analysis=16)

    # OPTIMIZATION: Collect samples more efficiently
    class_samples = {}
    total_samples_collected = 0

    logger.debug(f"Starting attention analysis - max {max_total_samples} samples")

    try:
        for inputs, labels in test_loader:
            if total_samples_collected >= max_total_samples:
                logger.debug(f"Reached sample limit ({max_total_samples}) - stopping collection")
                break

            for i in range(min(inputs.shape[0], max_total_samples - total_samples_collected)):
                label = labels[i].item()
                if label not in class_samples:
                    class_samples[label] = []
                if len(class_samples[label]) < samples_per_class:
                    class_samples[label].append(inputs[i])
                    total_samples_collected += 1

                    if total_samples_collected >= max_total_samples:
                        break

            # Early exit if we have enough samples
            if all(len(v) >= samples_per_class for v in class_samples.values()) or \
               total_samples_collected >= max_total_samples:
                break

        logger.debug(f"Collected {total_samples_collected} samples from {len(class_samples)} classes")

        # OPTIMIZATION: Analyze only a subset and only key layers/heads
        analyzed_data = {}
        for class_idx, samples in list(class_samples.items())[:min(4, len(class_samples))]:  # Max 4 classes
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            analyzed_data[class_idx] = {"inputs": [], "maps": []}

            for i, sample_input in enumerate(samples[:samples_per_class]):  # Respect limit
                sample_input_batch = sample_input.unsqueeze(0)

                # OPTIMIZATION: Extract attention with early exit if needed
                attention_maps = analyzer.extract_attention_maps(sample_input_batch)

                if not attention_maps:
                    logger.debug(f"No attention maps extracted for class {class_name} sample {i}")
                    continue

                analyzed_data[class_idx]["inputs"].append(sample_input)
                analyzed_data[class_idx]["maps"].append(attention_maps)

                # OPTIMIZATION: Save only key visualizations
                if save_dir and attention_maps:
                    for layer_idx, attn_map in enumerate(attention_maps[:2]):  # Only first 2 layers
                        for head_idx in range(min(2, attn_map.shape[1])):  # Only first 2 heads
                            save_path = os.path.join(
                                save_dir,
                                f"class_{class_name}_sample_{i}",
                                f"layer_{layer_idx}_head_{head_idx}.png",
                            )
                            visualize_attention_map(
                                attn_map,
                                sample_input_batch,
                                head_idx=head_idx,
                                layer_name=f"Layer {layer_idx}",
                                save_path=save_path,
                            )

        logger.debug(f"Attention analysis completed for {len(analyzed_data)} classes")
        return analyzed_data

    except Exception as e:
        logger.error(f"Error in analyze_task_attention: {e}")
        return {}
