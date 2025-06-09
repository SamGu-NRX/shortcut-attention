"""Attention visualization support for ViT backbone without modifying models."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

class AttentionExtractor:
    """Extract attention maps from ViT backbone without modifying model code."""
    
    def __init__(self):
        self.attention_maps = []
        self.hooks = []

    def _attention_hook(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        """Hook for capturing attention maps during forward pass."""
        if isinstance(output, tuple):
            # Some attention implementations return (output, attention)
            self.attention_maps.append(output[1].detach())
        else:
            # For standard attention output
            self.attention_maps.append(output.detach())

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on attention blocks."""
        # Clear any existing hooks
        self.remove_hooks()
        
        # Find attention modules in backbone
        for name, module in model.net.backbone.blocks.named_children():
            if hasattr(module, 'attn'):
                hook = module.attn.register_forward_hook(self._attention_hook)
                self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_attention_maps(self, model: nn.Module, inputs: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention maps for given inputs.
        
        Args:
            model: The continual learning model (e.g., DerPP)
            inputs: Input tensor of shape [B, C, H, W]
            
        Returns:
            List of attention tensors from each transformer block
        """
        self.attention_maps = []  # Reset stored maps
        
        # Register hooks if needed
        if not self.hooks:
            self.register_hooks(model)
            
        # Forward pass will trigger hooks
        with torch.no_grad():
            model.net(inputs)
            
        return self.attention_maps
