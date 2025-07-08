# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Einstellung Effect Attention Analysis

This module extends the existing attention visualization system with
Einstellung-specific metrics for analyzing attention patterns related to
shortcut learning and cognitive rigidity.

Extended Metrics:
- Attention Spread (AS): How concentrated attention is on shortcut patches
- Shortcut Attention Gain (SAG): Increase in attention to shortcut regions
- Cross-Task Attention Similarity (CTAS): Attention pattern similarity across tasks
- Attention Diversity Index (ADI): Diversity of attention across different regions
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

from utils.attention_visualization import AttentionAnalyzer
from utils.einstellung_metrics import EinstellungTimelineData


class EinstellungAttentionAnalyzer(AttentionAnalyzer):
    """
    Extended attention analyzer for Einstellung Effect experiments.

    Provides specialized metrics for analyzing attention patterns related to
    shortcut learning and cognitive rigidity in continual learning.
    """

    def __init__(self, model, device="cuda", patch_size=16):
        super().__init__(model, device)
        self.patch_size = patch_size
        self.attention_timeline = []

    def calculate_attention_spread(self,
                                 attention_map: torch.Tensor,
                                 shortcut_coords: Tuple[int, int, int, int] = None,
                                 head_idx: int = 0) -> float:
        """
        Calculate Attention Spread (AS): concentration of attention on shortcut patches.

        Args:
            attention_map: Attention tensor [B, num_heads, N, N]
            shortcut_coords: (x, y, width, height) of shortcut region in patch coordinates
            head_idx: Which attention head to analyze

        Returns:
            AS score (higher = more concentrated on shortcuts)
        """
        if attention_map.size(0) == 0:
            return 0.0

        # Get CLS token attention to patch tokens
        attn = attention_map[0, head_idx].cpu().numpy()  # [N, N]
        cls_attention = attn[0, 1:]  # Attention from CLS to patches

        if shortcut_coords is None:
            # If no shortcut coordinates provided, calculate general spread (entropy)
            # Normalize attention scores
            cls_attention = cls_attention / (cls_attention.sum() + 1e-8)
            # Higher entropy = more spread out (lower AS)
            spread = -entropy(cls_attention + 1e-8)
            return float(spread)

        # Calculate attention concentration on shortcut region
        x, y, w, h = shortcut_coords
        h_patches = int(np.sqrt(len(cls_attention)))
        w_patches = h_patches

        # Reshape to spatial layout
        attention_spatial = cls_attention.reshape(h_patches, w_patches)

        # Extract attention on shortcut region
        x_end = min(x + w, w_patches)
        y_end = min(y + h, h_patches)
        shortcut_attention = attention_spatial[y:y_end, x:x_end].sum()
        total_attention = attention_spatial.sum()

        # AS = proportion of attention on shortcut region
        as_score = shortcut_attention / (total_attention + 1e-8)
        return float(as_score)

    def calculate_shortcut_attention_gain(self,
                                        attention_normal: torch.Tensor,
                                        attention_with_shortcut: torch.Tensor,
                                        shortcut_coords: Tuple[int, int, int, int],
                                        head_idx: int = 0) -> float:
        """
        Calculate Shortcut Attention Gain (SAG): increase in attention to shortcut regions.

        Args:
            attention_normal: Attention map without shortcuts [B, num_heads, N, N]
            attention_with_shortcut: Attention map with shortcuts [B, num_heads, N, N]
            shortcut_coords: (x, y, width, height) of shortcut region
            head_idx: Which attention head to analyze

        Returns:
            SAG score (higher = more gain from shortcuts)
        """
        as_normal = self.calculate_attention_spread(attention_normal, shortcut_coords, head_idx)
        as_shortcut = self.calculate_attention_spread(attention_with_shortcut, shortcut_coords, head_idx)

        # SAG = relative increase in attention concentration
        sag_score = (as_shortcut - as_normal) / (as_normal + 1e-8)
        return float(sag_score)

    def calculate_cross_task_attention_similarity(self,
                                                attention_t1: torch.Tensor,
                                                attention_t2: torch.Tensor,
                                                head_idx: int = 0) -> float:
        """
        Calculate Cross-Task Attention Similarity (CTAS): similarity of attention patterns.

        Args:
            attention_t1: Task 1 attention maps [B, num_heads, N, N]
            attention_t2: Task 2 attention maps [B, num_heads, N, N]
            head_idx: Which attention head to analyze

        Returns:
            CTAS score (0-1, higher = more similar)
        """
        if attention_t1.size(0) == 0 or attention_t2.size(0) == 0:
            return 0.0

        # Extract CLS attention patterns
        attn_t1 = attention_t1[0, head_idx, 0, 1:].cpu().numpy()  # T1 CLS->patches
        attn_t2 = attention_t2[0, head_idx, 0, 1:].cpu().numpy()  # T2 CLS->patches

        # Normalize attention vectors
        attn_t1 = attn_t1 / (np.linalg.norm(attn_t1) + 1e-8)
        attn_t2 = attn_t2 / (np.linalg.norm(attn_t2) + 1e-8)

        # Calculate cosine similarity
        similarity = np.dot(attn_t1, attn_t2)
        return float(similarity)

    def calculate_attention_diversity_index(self,
                                          attention_maps: List[torch.Tensor],
                                          head_idx: int = 0) -> float:
        """
        Calculate Attention Diversity Index (ADI): diversity of attention across samples.

        Args:
            attention_maps: List of attention tensors from different samples
            head_idx: Which attention head to analyze

        Returns:
            ADI score (higher = more diverse attention patterns)
        """
        if not attention_maps:
            return 0.0

        attention_vectors = []
        for attn_map in attention_maps:
            if attn_map.size(0) > 0:
                attn_vec = attn_map[0, head_idx, 0, 1:].cpu().numpy()
                attn_vec = attn_vec / (np.linalg.norm(attn_vec) + 1e-8)
                attention_vectors.append(attn_vec)

        if len(attention_vectors) < 2:
            return 0.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(attention_vectors)):
            for j in range(i + 1, len(attention_vectors)):
                sim = np.dot(attention_vectors[i], attention_vectors[j])
                similarities.append(sim)

        # ADI = 1 - mean similarity (higher diversity = lower similarity)
        mean_similarity = np.mean(similarities)
        adi_score = 1.0 - mean_similarity
        return float(adi_score)

    def extract_shortcut_region_coords(self,
                                     input_tensor: torch.Tensor,
                                     patch_size: int = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract shortcut region coordinates from an input with magenta patches.

        Args:
            input_tensor: Input image tensor [B, C, H, W]
            patch_size: Size of patches for coordinate conversion

        Returns:
            (x, y, width, height) in patch coordinates, or None if no shortcut found
        """
        if patch_size is None:
            patch_size = self.patch_size

        # Convert to numpy for processing
        img = input_tensor[0].cpu().numpy()  # [C, H, W]
        img = np.transpose(img, (1, 2, 0))  # [H, W, C]

        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
        img = img * std + mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Look for magenta patches (RGB: 255, 0, 255)
        magenta_mask = ((img[:, :, 0] > 200) &
                       (img[:, :, 1] < 50) &
                       (img[:, :, 2] > 200))

        if not np.any(magenta_mask):
            return None

        # Find bounding box of magenta region
        rows, cols = np.where(magenta_mask)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()

        # Convert to patch coordinates
        patch_x = x_min // patch_size
        patch_y = y_min // patch_size
        patch_w = max(1, (x_max - x_min) // patch_size)
        patch_h = max(1, (y_max - y_min) // patch_size)

        return (patch_x, patch_y, patch_w, patch_h)

    def analyze_einstellung_attention_batch(self,
                                          inputs: torch.Tensor,
                                          subset_name: str,
                                          epoch: int = 0) -> Dict[str, float]:
        """
        Analyze attention patterns for a batch of Einstellung data.

        Args:
            inputs: Batch of input tensors [B, C, H, W]
            subset_name: Name of evaluation subset (e.g., 'T2_shortcut_normal')
            epoch: Current training epoch

        Returns:
            Dictionary of attention metrics
        """
        attention_maps = self.extract_attention_maps(inputs)
        if not attention_maps:
            return {}

        metrics = {}

        # Analyze attention for each layer
        for layer_idx, attn_map in enumerate(attention_maps):
            layer_metrics = {}

            # Calculate metrics for multiple heads
            num_heads = attn_map.size(1)
            for head_idx in range(min(4, num_heads)):  # Analyze first 4 heads

                # Basic attention spread
                as_score = self.calculate_attention_spread(attn_map, head_idx=head_idx)
                layer_metrics[f'attention_spread_head_{head_idx}'] = as_score

                # If this is shortcut data, analyze shortcut-specific metrics
                if 'shortcut' in subset_name:
                    shortcut_coords = self.extract_shortcut_region_coords(inputs)
                    if shortcut_coords:
                        as_shortcut = self.calculate_attention_spread(
                            attn_map, shortcut_coords, head_idx
                        )
                        layer_metrics[f'shortcut_attention_spread_head_{head_idx}'] = as_shortcut

            # Store layer metrics
            for metric_name, value in layer_metrics.items():
                metrics[f'layer_{layer_idx}_{metric_name}'] = value

        # Store timeline data
        self.attention_timeline.append({
            'epoch': epoch,
            'subset_name': subset_name,
            'metrics': metrics
        })

        return metrics

    def visualize_einstellung_attention_comparison(self,
                                                 normal_inputs: torch.Tensor,
                                                 shortcut_inputs: torch.Tensor,
                                                 masked_inputs: torch.Tensor,
                                                 save_path: str = None) -> None:
        """
        Create comparative visualization of attention patterns across different conditions.

        Args:
            normal_inputs: Inputs without shortcuts
            shortcut_inputs: Inputs with shortcuts
            masked_inputs: Inputs with masked shortcuts
            save_path: Path to save visualization
        """
        # Extract attention maps for each condition
        attn_normal = self.extract_attention_maps(normal_inputs)
        attn_shortcut = self.extract_attention_maps(shortcut_inputs)
        attn_masked = self.extract_attention_maps(masked_inputs)

        if not (attn_normal and attn_shortcut and attn_masked):
            print("Warning: Could not extract attention maps for comparison")
            return

        # Use last layer attention for visualization
        attn_normal_last = attn_normal[-1][0, 0]  # [N, N]
        attn_shortcut_last = attn_shortcut[-1][0, 0]
        attn_masked_last = attn_masked[-1][0, 0]

        # Get CLS attention to patches
        cls_normal = attn_normal_last[0, 1:].cpu().numpy()
        cls_shortcut = attn_shortcut_last[0, 1:].cpu().numpy()
        cls_masked = attn_masked_last[0, 1:].cpu().numpy()

        # Reshape to spatial layout
        h_patches = int(np.sqrt(len(cls_normal)))
        w_patches = h_patches

        attn_normal_spatial = cls_normal.reshape(h_patches, w_patches)
        attn_shortcut_spatial = cls_shortcut.reshape(h_patches, w_patches)
        attn_masked_spatial = cls_masked.reshape(h_patches, w_patches)

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Einstellung Effect: Attention Pattern Comparison', fontsize=16)

        # Top row: input images
        for i, (inputs, title) in enumerate([(normal_inputs, 'Normal'),
                                           (shortcut_inputs, 'With Shortcuts'),
                                           (masked_inputs, 'Masked Shortcuts')]):
            img = self._denormalize_image(inputs[0])
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'{title} Input')
            axes[0, i].axis('off')

        # Bottom row: attention maps
        attention_data = [(attn_normal_spatial, 'Normal'),
                         (attn_shortcut_spatial, 'With Shortcuts'),
                         (attn_masked_spatial, 'Masked')]

        vmin = min(data.min() for data, _ in attention_data)
        vmax = max(data.max() for data, _ in attention_data)

        for i, (attn_data, title) in enumerate(attention_data):
            im = axes[1, i].imshow(attn_data, cmap='hot', vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f'{title} Attention')
            axes[1, i].axis('off')

        # Add colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _denormalize_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Denormalize image tensor for visualization."""
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
        img = np.clip(img * std + mean, 0, 1)
        return img

    def get_attention_timeline_summary(self) -> Dict[str, List[float]]:
        """
        Get summary of attention metrics over training timeline.

        Returns:
            Dictionary mapping metric names to lists of values over time
        """
        summary = {}

        for entry in self.attention_timeline:
            epoch = entry['epoch']
            subset_name = entry['subset_name']

            for metric_name, value in entry['metrics'].items():
                full_metric_name = f"{subset_name}_{metric_name}"
                if full_metric_name not in summary:
                    summary[full_metric_name] = []
                summary[full_metric_name].append(value)

        return summary

    def export_attention_analysis(self, filepath: str) -> None:
        """
        Export comprehensive attention analysis to file.

        Args:
            filepath: Path to save analysis results
        """
        analysis_data = {
            'timeline_data': self.attention_timeline,
            'summary_statistics': self.get_attention_timeline_summary()
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
