"""
Attention visualization utilities for Vision Transformers in continual learning experiments.
This module provides functions to visualize and analyze attention maps from ViT models.
"""

import os
from typing import Dict, List, Tuple, Optional, Union
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionAnalyzer:
    """
    Class for analyzing attention patterns in Vision Transformer models.
    Uses the model's built-in attention score extraction rather than hooks.
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize the attention analyzer.
        
        Args:
            model: The Vision Transformer model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.class_token_idx = 0  # Index of CLS token attention
    
    def extract_attention_maps(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for given inputs using the model's built-in attention score extraction.
        
        Args:
            inputs: Input tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary with attention maps from each transformer block
        """
        with torch.no_grad():
            self.model.eval()
            # Forward pass with only necessary arguments
            outputs = self.model(inputs.to(self.device), return_attention=True)
            
            # Extract attention maps based on model output format
            if isinstance(outputs, tuple):
                # Handle case where model returns (logits, attention_maps)
                _, attn_maps = outputs
            elif isinstance(outputs, dict):
                # Handle case where model returns a dictionary
                attn_maps = outputs.get('attention_maps', [])
            else:
                raise ValueError("Model output format not supported for attention visualization")
            
        # Ensure attention maps are in the expected format
        if not isinstance(attn_maps, (list, tuple)):
            attn_maps = [attn_maps]
            
        return {f'block_{i}': maps.cpu() for i, maps in enumerate(attn_maps)}

def visualize_attention_map(attention_map: torch.Tensor, 
                          input_image: torch.Tensor,
                          head_idx: int = 0,
                          layer_name: str = "",
                          save_path: Optional[str] = None,
                          patch_size: int = 16,
                          class_token_idx: int = 0) -> None:
    """
    Visualize attention map from a specific head overlaid on the input image.
    
    Args:
        attention_map: Attention tensor of shape [B, num_heads, N, N]
        input_image: Input image tensor of shape [B, C, H, W]
        head_idx: Which attention head to visualize
        layer_name: Name of the layer for title
        save_path: Path to save the visualization
        patch_size: Size of patches in the ViT
        class_token_idx: Index of the CLS token (usually 0)
    """
    # Take first sample in batch
    attn = attention_map[0, head_idx]  # Shape: (N, N)
    img = input_image[0]  # Shape: (C, H, W)
    
    # Convert image to numpy and denormalize
    img_np = img.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2615])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Get attention from CLS token to patches (first row, excluding CLS token)
    cls_attention = attn[class_token_idx, 1:].cpu().numpy()  # Shape: (num_patches,)
    
    # Reshape attention to spatial dimensions
    H, W = img.shape[1], img.shape[2]
    num_patches_per_side = H // patch_size
    attention_spatial = cls_attention.reshape(num_patches_per_side, num_patches_per_side)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    im1 = axes[1].imshow(attention_spatial, cmap='hot')
    axes[1].set_title(f'Attention Map - {layer_name} Head {head_idx}')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Overlay
    axes[2].imshow(img_np)
    axes[2].imshow(attention_spatial, cmap='hot', alpha=0.6)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_attention_patterns(attention_maps: Dict[str, torch.Tensor],
                             class_names: List[str],
                             save_dir: str,
                             class_token_idx: int = 0) -> Dict[str, Dict[str, float]]:
    """
    Analyze attention patterns across different layers and heads.
    
    Args:
        attention_maps: Dictionary of attention maps from different layers
        class_names: Names of the classes
        save_dir: Directory to save analysis results
        class_token_idx: Index of the CLS token
        
    Returns:
        Dictionary with attention statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    stats = {}
    
    for layer_name, attn in attention_maps.items():
        # attn shape: (B, num_heads, N, N)
        B, num_heads, N, _ = attn.shape
        
        # Analyze CLS token attention (attention from CLS to other tokens)
        cls_attention = attn[:, :, class_token_idx, 1:]  # (B, num_heads, num_patches)
        
        # Compute statistics
        mean_attention = cls_attention.mean(dim=0)  # (num_heads, num_patches)
        std_attention = cls_attention.std(dim=0)
        max_attention = cls_attention.max(dim=0)[0]
        
        # Attention entropy (measure of attention spread)
        attention_entropy = -(cls_attention * torch.log(cls_attention + 1e-8)).sum(dim=-1)
        
        stats[layer_name] = {
            'mean': mean_attention.mean().item(),
            'std': std_attention.mean().item(),
            'max': max_attention.mean().item(),
            'entropy': attention_entropy.mean().item()
        }
        
        # Create heatmap of attention patterns across heads
        plt.figure(figsize=(12, 8))
        sns.heatmap(mean_attention.cpu().numpy(), 
                   xticklabels=False, 
                   yticklabels=[f'Head {i}' for i in range(num_heads)],
                   cmap='viridis')
        plt.title(f'Mean Attention Patterns - {layer_name}')
        plt.xlabel('Patch Index')
        plt.ylabel('Attention Head')
        plt.savefig(os.path.join(save_dir, f'attention_heatmap_{layer_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    return stats

def compare_attention_across_tasks(task1_maps: Dict[str, torch.Tensor],
                                 task2_maps: Dict[str, torch.Tensor],
                                 save_dir: str,
                                 class_token_idx: int = 0) -> Dict[str, Dict[str, float]]:
    """
    Compare attention patterns between two tasks.
    
    Args:
        task1_maps: Attention maps from task 1
        task2_maps: Attention maps from task 2
        save_dir: Directory to save comparison results
        class_token_idx: Index of the CLS token
        
    Returns:
        Dictionary with comparison statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    comparison_stats = {}
    
    for layer_name in task1_maps.keys():
        if layer_name not in task2_maps:
            continue
            
        attn1 = task1_maps[layer_name]
        attn2 = task2_maps[layer_name]
        
        # Extract CLS attention for both tasks
        cls_attn1 = attn1[:, :, class_token_idx, 1:].mean(dim=0)  # Average over batch
        cls_attn2 = attn2[:, :, class_token_idx, 1:].mean(dim=0)
        
        # Compute differences and metrics
        attention_diff = (cls_attn2 - cls_attn1).cpu().numpy()
        correlation = torch.corrcoef(torch.stack([
            cls_attn1.flatten(), 
            cls_attn2.flatten()
        ]))[0, 1].item()
        
        kl_div = torch.nn.functional.kl_div(
            torch.log(cls_attn2 + 1e-8), 
            cls_attn1,
            reduction='batchmean'
        ).item()
        
        comparison_stats[layer_name] = {
            'correlation': correlation,
            'kl_divergence': kl_div,
            'mean_diff': float(attention_diff.mean()),
            'max_diff': float(np.abs(attention_diff).max())
        }
        
        # Visualize attention differences
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_diff, 
                   center=0,
                   cmap='RdBu_r',
                   xticklabels=False,
                   yticklabels=[f'Head {i}' for i in range(attention_diff.shape[0])])
        plt.title(f'Attention Difference (Task2 - Task1) - {layer_name}')
        plt.xlabel('Patch Index')
        plt.ylabel('Attention Head')
        plt.savefig(os.path.join(save_dir, f'attention_diff_{layer_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    return comparison_stats
