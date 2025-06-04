"""
Attention visualization utilities for Vision Transformers in continual learning experiments.
This module provides functions to extract and visualize attention maps from ViT models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from PIL import Image
import cv2


class AttentionExtractor:
    """
    Class to extract attention maps from Vision Transformer models.
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize the attention extractor.
        
        Args:
            model: The Vision Transformer model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.attention_maps = {}
        self.hooks = []
        
        # Register hooks to extract attention maps
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract attention maps from each transformer block."""
        def get_attention_hook(layer_name):
            def hook(module, input, output):
                # For ViT, we need to extract attention from the attention layer
                if hasattr(module, 'attn') and hasattr(module.attn, 'qkv'):
                    # Extract attention weights during forward pass
                    x = input[0]
                    B, N, C = x.shape
                    
                    # Get q, k, v
                    qkv = module.attn.qkv(module.norm1(x))
                    qkv = qkv.reshape(B, N, 3, module.attn.num_heads, C // module.attn.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    
                    # Compute attention weights
                    attn = (q @ k.transpose(-2, -1)) * module.attn.scale
                    attn = attn.softmax(dim=-1)
                    
                    # Store attention maps
                    self.attention_maps[layer_name] = attn.detach().cpu()
            return hook
        
        # Register hooks for each transformer block
        for i, block in enumerate(self.model.net.blocks):
            hook = block.register_forward_hook(get_attention_hook(f'block_{i}'))
            self.hooks.append(hook)
    
    def extract_attention(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for given inputs.
        
        Args:
            inputs: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary mapping layer names to attention maps
        """
        self.attention_maps = {}
        
        with torch.no_grad():
            self.model.eval()
            _ = self.model(inputs.to(self.device))
        
        return self.attention_maps.copy()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def visualize_attention_map(attention_map: torch.Tensor, 
                          input_image: torch.Tensor,
                          head_idx: int = 0,
                          layer_name: str = "",
                          save_path: Optional[str] = None,
                          patch_size: int = 16) -> None:
    """
    Visualize attention map overlaid on the input image.
    
    Args:
        attention_map: Attention tensor of shape (B, num_heads, N, N)
        input_image: Input image tensor of shape (B, C, H, W)
        head_idx: Which attention head to visualize
        layer_name: Name of the layer for title
        save_path: Path to save the visualization
        patch_size: Size of patches in the ViT
    """
    # Take first sample in batch
    attn = attention_map[0, head_idx]  # Shape: (N, N)
    img = input_image[0]  # Shape: (C, H, W)
    
    # Convert image to numpy and denormalize
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize CIFAR-10 image
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2615])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Get attention from CLS token to patches (first row, excluding CLS token)
    cls_attention = attn[0, 1:].cpu().numpy()  # Shape: (num_patches,)
    
    # Reshape attention to spatial dimensions
    H, W = img.shape[1], img.shape[2]
    num_patches_per_side = H // patch_size
    attention_spatial = cls_attention.reshape(num_patches_per_side, num_patches_per_side)
    
    # Resize attention map to match image size
    attention_resized = cv2.resize(attention_spatial, (W, H), interpolation=cv2.INTER_CUBIC)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    im1 = axes[1].imshow(attention_resized, cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Attention Map - {layer_name} Head {head_idx}')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Overlay
    axes[2].imshow(img_np)
    axes[2].imshow(attention_resized, cmap='hot', alpha=0.6, interpolation='nearest')
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
                             save_dir: str) -> Dict[str, float]:
    """
    Analyze attention patterns across different layers and heads.
    
    Args:
        attention_maps: Dictionary of attention maps from different layers
        class_names: Names of the classes
        save_dir: Directory to save analysis results
        
    Returns:
        Dictionary with attention statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    stats = {}
    
    for layer_name, attn in attention_maps.items():
        # attn shape: (B, num_heads, N, N)
        B, num_heads, N, _ = attn.shape
        
        # Analyze CLS token attention (first token)
        cls_attention = attn[:, :, 0, 1:]  # (B, num_heads, num_patches)
        
        # Compute statistics
        mean_attention = cls_attention.mean(dim=0)  # (num_heads, num_patches)
        std_attention = cls_attention.std(dim=0)
        max_attention = cls_attention.max(dim=0)[0]
        
        # Attention entropy (measure of attention spread)
        attention_entropy = -(cls_attention * torch.log(cls_attention + 1e-8)).sum(dim=-1)
        
        stats[layer_name] = {
            'mean_attention': mean_attention.cpu().numpy(),
            'std_attention': std_attention.cpu().numpy(),
            'max_attention': max_attention.cpu().numpy(),
            'attention_entropy': attention_entropy.mean().item()
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


def compare_attention_across_tasks(attention_maps_task1: Dict[str, torch.Tensor],
                                 attention_maps_task2: Dict[str, torch.Tensor],
                                 save_dir: str) -> Dict[str, float]:
    """
    Compare attention patterns between two tasks to identify changes.
    
    Args:
        attention_maps_task1: Attention maps from task 1
        attention_maps_task2: Attention maps from task 2
        save_dir: Directory to save comparison results
        
    Returns:
        Dictionary with comparison statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    comparison_stats = {}
    
    for layer_name in attention_maps_task1.keys():
        if layer_name not in attention_maps_task2:
            continue
            
        attn1 = attention_maps_task1[layer_name]
        attn2 = attention_maps_task2[layer_name]
        
        # Extract CLS attention for both tasks
        cls_attn1 = attn1[:, :, 0, 1:].mean(dim=0)  # Average over batch
        cls_attn2 = attn2[:, :, 0, 1:].mean(dim=0)
        
        # Compute differences
        attention_diff = (cls_attn2 - cls_attn1).cpu().numpy()
        
        # Compute correlation between attention patterns
        correlation = torch.corrcoef(torch.stack([
            cls_attn1.flatten(), 
            cls_attn2.flatten()
        ]))[0, 1].item()
        
        # Compute KL divergence
        kl_div = torch.nn.functional.kl_div(
            torch.log(cls_attn2 + 1e-8), 
            cls_attn1, 
            reduction='batchmean'
        ).item()
        
        comparison_stats[layer_name] = {
            'correlation': correlation,
            'kl_divergence': kl_div,
            'mean_abs_diff': np.abs(attention_diff).mean(),
            'max_abs_diff': np.abs(attention_diff).max()
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
