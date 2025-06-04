"""
Network flow visualization utilities for analyzing activation patterns in continual learning.
This module provides functions to extract and visualize network activations and flow.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ActivationExtractor:
    """
    Class to extract activations from different layers of a neural network.
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize the activation extractor.
        
        Args:
            model: The neural network model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.activations = {}
        self.hooks = []
        
        # Register hooks to extract activations
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract activations from different layers."""
        def get_activation_hook(layer_name):
            def hook(module, input, output):
                # Store activations
                if isinstance(output, torch.Tensor):
                    self.activations[layer_name] = output.detach().cpu()
                elif isinstance(output, (list, tuple)):
                    # Handle cases where output is a tuple/list
                    self.activations[layer_name] = output[0].detach().cpu()
            return hook
        
        # Register hooks for different types of layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
                hook = module.register_forward_hook(get_activation_hook(name))
                self.hooks.append(hook)
            elif 'block' in name and hasattr(module, 'forward'):
                # For transformer blocks
                hook = module.register_forward_hook(get_activation_hook(name))
                self.hooks.append(hook)
    
    def extract_activations(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract activations for given inputs.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Dictionary mapping layer names to activations
        """
        self.activations = {}
        
        with torch.no_grad():
            self.model.eval()
            _ = self.model(inputs.to(self.device))
        
        return self.activations.copy()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def compute_activation_statistics(activations: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for activations from different layers.
    
    Args:
        activations: Dictionary of activations from different layers
        
    Returns:
        Dictionary with activation statistics
    """
    stats = {}
    
    for layer_name, activation in activations.items():
        # Flatten activation for statistics
        flat_activation = activation.flatten()
        
        stats[layer_name] = {
            'mean': float(flat_activation.mean()),
            'std': float(flat_activation.std()),
            'min': float(flat_activation.min()),
            'max': float(flat_activation.max()),
            'sparsity': float((flat_activation == 0).sum() / len(flat_activation)),
            'l1_norm': float(flat_activation.abs().mean()),
            'l2_norm': float(torch.norm(flat_activation)),
        }
    
    return stats


def visualize_activation_flow(activations: Dict[str, torch.Tensor], 
                            save_path: Optional[str] = None) -> None:
    """
    Visualize the flow of activations through the network.
    
    Args:
        activations: Dictionary of activations from different layers
        save_path: Path to save the visualization
    """
    # Extract statistics for visualization
    layer_names = []
    means = []
    stds = []
    sparsities = []
    l2_norms = []
    
    for layer_name, activation in activations.items():
        if 'block' in layer_name or 'head' in layer_name:  # Focus on main layers
            layer_names.append(layer_name)
            flat_activation = activation.flatten()
            means.append(float(flat_activation.mean()))
            stds.append(float(flat_activation.std()))
            sparsities.append(float((flat_activation == 0).sum() / len(flat_activation)))
            l2_norms.append(float(torch.norm(flat_activation)))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot mean activations
    axes[0, 0].plot(range(len(layer_names)), means, 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_title('Mean Activation Values')
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Mean Activation')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot standard deviations
    axes[0, 1].plot(range(len(layer_names)), stds, 'o-', color='orange', linewidth=2, markersize=6)
    axes[0, 1].set_title('Activation Standard Deviation')
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Std Deviation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot sparsity
    axes[1, 0].plot(range(len(layer_names)), sparsities, 'o-', color='green', linewidth=2, markersize=6)
    axes[1, 0].set_title('Activation Sparsity')
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Sparsity (fraction of zeros)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot L2 norms
    axes[1, 1].plot(range(len(layer_names)), l2_norms, 'o-', color='red', linewidth=2, markersize=6)
    axes[1, 1].set_title('L2 Norm of Activations')
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_feature_representations(activations: Dict[str, torch.Tensor],
                                  labels: torch.Tensor,
                                  class_names: List[str],
                                  save_dir: str,
                                  layer_name: str = None) -> None:
    """
    Analyze feature representations using dimensionality reduction.
    
    Args:
        activations: Dictionary of activations from different layers
        labels: Ground truth labels
        class_names: Names of the classes
        save_dir: Directory to save visualizations
        layer_name: Specific layer to analyze (if None, analyze all)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    layers_to_analyze = [layer_name] if layer_name else list(activations.keys())
    
    for layer in layers_to_analyze:
        if layer not in activations:
            continue
            
        activation = activations[layer]
        
        # Reshape activation for analysis
        if len(activation.shape) > 2:
            # For conv layers or transformer outputs, flatten spatial dimensions
            batch_size = activation.shape[0]
            activation_flat = activation.view(batch_size, -1)
        else:
            activation_flat = activation
        
        # Skip if too few samples
        if activation_flat.shape[0] < 2:
            continue
        
        # Apply PCA
        if activation_flat.shape[1] > 50:  # Only apply PCA if high dimensional
            pca = PCA(n_components=min(50, activation_flat.shape[0]-1))
            activation_pca = pca.fit_transform(activation_flat.numpy())
        else:
            activation_pca = activation_flat.numpy()
        
        # Apply t-SNE for 2D visualization
        if activation_pca.shape[0] > 1:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, activation_pca.shape[0]-1))
            activation_tsne = tsne.fit_transform(activation_pca)
            
            # Create t-SNE plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(activation_tsne[:, 0], activation_tsne[:, 1], 
                                c=labels.numpy(), cmap='tab10', alpha=0.7)
            plt.colorbar(scatter, ticks=range(len(class_names)), 
                        label='Class')
            plt.title(f't-SNE Visualization - {layer}')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            # Add legend
            for i, class_name in enumerate(class_names):
                plt.scatter([], [], c=plt.cm.tab10(i), label=class_name)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'tsne_{layer.replace(".", "_")}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()


def compare_activation_patterns(activations_task1: Dict[str, torch.Tensor],
                              activations_task2: Dict[str, torch.Tensor],
                              save_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Compare activation patterns between two tasks.
    
    Args:
        activations_task1: Activations from task 1
        activations_task2: Activations from task 2
        save_dir: Directory to save comparison results
        
    Returns:
        Dictionary with comparison statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    comparison_stats = {}
    
    for layer_name in activations_task1.keys():
        if layer_name not in activations_task2:
            continue
        
        act1 = activations_task1[layer_name].flatten()
        act2 = activations_task2[layer_name].flatten()
        
        # Ensure same size for comparison
        min_size = min(len(act1), len(act2))
        act1 = act1[:min_size]
        act2 = act2[:min_size]
        
        # Compute statistics
        correlation = torch.corrcoef(torch.stack([act1, act2]))[0, 1].item()
        mse = torch.nn.functional.mse_loss(act1, act2).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            act1.unsqueeze(0), act2.unsqueeze(0)
        ).item()
        
        comparison_stats[layer_name] = {
            'correlation': correlation,
            'mse': mse,
            'cosine_similarity': cosine_sim,
            'mean_diff': float((act2 - act1).mean()),
            'std_diff': float((act2 - act1).std())
        }
    
    # Create comparison visualization
    layer_names = list(comparison_stats.keys())
    correlations = [comparison_stats[layer]['correlation'] for layer in layer_names]
    cosine_sims = [comparison_stats[layer]['cosine_similarity'] for layer in layer_names]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(layer_names)), correlations, 'o-', linewidth=2, markersize=6)
    plt.title('Activation Correlation Between Tasks')
    plt.xlabel('Layer Index')
    plt.ylabel('Correlation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(layer_names)), cosine_sims, 'o-', color='orange', linewidth=2, markersize=6)
    plt.title('Cosine Similarity Between Tasks')
    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'activation_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_stats
