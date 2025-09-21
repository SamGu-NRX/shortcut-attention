#!/usr/bin/env python3
"""
CUDA Performance Optimization Script
Apply these fixes to improve CUDA performance from 3it/s to expected speeds.
"""

import torch
import logging

def apply_cuda_performance_fixes():
    """Apply recommended CUDA performance optimizations"""

    logging.info("Applying CUDA performance optimizations...")

    # 1. Optimize thread count (don't limit to 2!)
    optimal_threads = min(torch.get_num_threads(), 8)  # Usually 4-8 is optimal
    torch.set_num_threads(optimal_threads)
    logging.info(f"Set torch threads to: {optimal_threads}")

    # 2. Enable optimized CUDA operations
    if torch.cuda.is_available():
        # Enable TensorFloat-32 (TF32) for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("Enabled TF32 for faster operations")

        # Enable cuDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        logging.info("Enabled cuDNN benchmarking")

        # Optimize memory allocation
        torch.cuda.empty_cache()
        logging.info("Cleared CUDA cache")

    # 3. Set optimal environment variables
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Disable blocking for async operations

    logging.info("CUDA performance optimizations applied!")

if __name__ == "__main__":
    apply_cuda_performance_fixes()
