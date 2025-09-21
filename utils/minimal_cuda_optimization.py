"""
Minimal CUDA Performance Optimization
Uses PyTorch defaults with only essential optimizations that actually improve performance.
"""

import torch
import logging
import os

logger = logging.getLogger(__name__)

def apply_minimal_cuda_optimizations(optimization_level: int = 1) -> dict:
    """
    Apply minimal, proven CUDA optimizations without overriding PyTorch defaults.

    Args:
        optimization_level: 0=none, 1=essential, 2=+precision, 3=+compile

    Returns:
        dict: Applied optimizations
    """
    optimizations = {}

    if optimization_level == 0:
        logger.info("CUDA optimizations disabled")
        return optimizations

    if not torch.cuda.is_available():
        logger.info("CUDA not available - no optimizations applied")
        return optimizations

    # Get GPU info
    device_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    logger.info(f"Optimizing for: {device_name} (compute {compute_capability[0]}.{compute_capability[1]})")

    # Level 1: Essential optimizations only
    if optimization_level >= 1:
        # Enable TF32 for Ampere+ GPUs (RTX A4500 is Ampere)
        if compute_capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            optimizations['tf32'] = True
            logger.info("✓ Enabled TensorFloat-32 (TF32)")

        # Enable cuDNN benchmark (auto-tune algorithms)
        torch.backends.cudnn.benchmark = True
        optimizations['cudnn_benchmark'] = True
        logger.info("✓ Enabled cuDNN benchmark")

        # Disable deterministic for speed
        torch.backends.cudnn.deterministic = False
        optimizations['deterministic'] = False
        logger.info("✓ Disabled deterministic mode for speed")

    # Level 2: Add precision optimizations
    if optimization_level >= 2:
        # Enable mixed precision if supported
        if compute_capability[0] >= 7:  # Volta+ supports mixed precision
            optimizations['mixed_precision'] = True
            logger.info("✓ Mixed precision available (use with autocast)")

    # Level 3: Add compilation (experimental)
    if optimization_level >= 3:
        if compute_capability[0] >= 7:
            optimizations['compile_ready'] = True
            logger.info("✓ torch.compile ready (will be applied to model)")
        else:
            logger.warning("⚠️  torch.compile requires compute capability 7.0+")

    # Environment optimizations (always safe)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async operations

    # Clear cache
    torch.cuda.empty_cache()

    logger.info(f"Applied {len(optimizations)} CUDA optimizations (level {optimization_level})")
    return optimizations

def get_optimal_dataloader_settings(batch_size: int) -> dict:
    """
    Get optimal DataLoader settings without overriding PyTorch defaults.

    Args:
        batch_size: Training batch size

    Returns:
        dict: Optimal settings
    """
    settings = {}

    if torch.cuda.is_available():
        # For CUDA: pin_memory is almost always beneficial
        settings['pin_memory'] = True

        # For num_workers: let user/system decide, but provide guidance
        # Don't override - just suggest
        cpu_count = os.cpu_count() or 4
        if batch_size >= 64:
            suggested_workers = min(4, cpu_count // 2)
        else:
            suggested_workers = 0  # Single-threaded often better for small batches

        settings['suggested_num_workers'] = suggested_workers
        logger.info(f"Suggested num_workers: {suggested_workers} (current batch_size: {batch_size})")

    return settings
