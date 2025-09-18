"""
Deep Generative Replay (DGR) for Mammoth

This module provides the DGR continual learning method, which uses a Variational
Autoencoder (VAE) to generate synthetic samples from previous tasks for replay-based
continual learning.

The implementation adapts the original DGR approach to work seamlessly with the
Mammoth framework while preserving the original VAE architecture and training procedures.
"""

# Import the main DGR implementation
from models.dgr_mammoth_adapter import DGRMammothAdapter as Dgr

# Ensure the class is available for Mammoth's model discovery
__all__ = ['Dgr']
