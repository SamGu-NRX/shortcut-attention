"""
Automatic Performance Optimization for PyTorch CUDA
Dynamically configures PyTorch for maximum performance based on hardware detection.
"""

import torch
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

def auto_optimize_pytorch_performance(device: Optional[torch.device] = None, optimization_level: int = 1) -> dict:
    """
    Automatically configure PyTorch for maximum performance.

    Args:
        device: CUDA device to optimize for
        optimization_level: 0=none, 1=TF32+cuDNN, 2=+BF16, 3=+torch.compile

    Returns:
        dict: Applied optimizations summary
    """
    optimizations = {}

    if optimization_level == 0:
        logger.info("Performance optimizations disabled (level 0)")
        return optimizations

    if not torch.cuda.is_available():
        logger.info("CUDA not available - using CPU optimizations only")
        return optimizations

    if device is None:
        device = torch.cuda.current_device()

    # Get GPU information
    gpu_name = torch.cuda.get_device_name(device)
    compute_capability = torch.cuda.get_device_capability(device)
    memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)

    logger.info(f"Auto-optimizing for: {gpu_name}")
    logger.info(f"Compute capability: {compute_capability[0]}.{compute_capability[1]}")
    logger.info(f"Memory: {memory_gb:.1f} GB")

    # 1. Enable TensorFloat-32 for Ampere+ GPUs (compute capability >= 8.0)
    if compute_capability[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        optimizations['tf32'] = True
        logger.info("✓ Enabled TensorFloat-32 (TF32) for Ampere+ GPU")
    else:
        optimizations['tf32'] = False
        logger.info("- TF32 not available (requires Ampere+ GPU)")

    # 2. Auto-optimize cuDNN
    torch.backends.cudnn.benchmark = True  # Auto-tune algorithms
    torch.backends.cudnn.deterministic = False  # Disable for max speed
    torch.backends.cudnn.enabled = True
    optimizations['cudnn_benchmark'] = True
    logger.info("✓ Enabled cuDNN auto-tuning")

    # 3. Enable Flash Attention and other optimized kernels (PyTorch 2.0+)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        optimizations['flash_attention'] = True
        logger.info("✓ Enabled Flash Attention and optimized kernels")
    except AttributeError:
        optimizations['flash_attention'] = False
        logger.info("- Flash Attention not available (requires PyTorch 2.0+)")

    # 4. Use PyTorch's default thread optimization (don't override!)
    current_threads = torch.get_num_threads()
    optimizations['threads'] = current_threads
    logger.info(f"✓ Using PyTorch default thread count: {current_threads} (not overriding)")

    # 5. Set environment variables for maximum performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async operations
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Use latest cuDNN API

    # 6. Memory optimization
    torch.cuda.empty_cache()

    # 7. Enable JIT compilation for supported operations
    try:
        torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
        optimizations['jit_fusion'] = True
        logger.info("✓ Enabled JIT fusion optimization")
    except:
        optimizations['jit_fusion'] = False

    # 8. Set optimal memory allocation strategy
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        # Use 90% of GPU memory to avoid fragmentation
        torch.cuda.set_per_process_memory_fraction(0.9, device)
        optimizations['memory_fraction'] = 0.9
        logger.info("✓ Set optimal memory allocation (90%)")

    logger.info("🚀 Auto-optimization complete - maximum performance enabled!")
    return optimizations

def get_optimal_dataloader_config(batch_size: int, dataset_size: int) -> dict:
    """
    Get optimal DataLoader configuration based on hardware and data characteristics.

    Args:
        batch_size: Batch size for training
        dataset_size: Total number of samples in dataset

    Returns:
        dict: Optimal DataLoader configuration
    """
    config = {}

    if torch.cuda.is_available():
        # Auto-determine optimal number of workers
        cpu_cores = os.cpu_count() or 4

        if batch_size >= 128:
            # Large batches can benefit from parallel data loading
            num_workers = min(4, cpu_cores // 2)
        elif batch_size >= 64:
            num_workers = min(2, cpu_cores // 4)
        else:
            # Small batches: single-threaded is often faster
            num_workers = 0

        config.update({
            'num_workers': num_workers,
            'pin_memory': True,  # Faster CPU-GPU transfers
            'persistent_workers': num_workers > 0,  # Reuse workers
            'prefetch_factor': 2 if num_workers > 0 else None,  # Prefetch batches
        })

        logger.info(f"Optimal DataLoader config: {num_workers} workers, pin_memory=True")
    else:
        # CPU-only configuration
        config.update({
            'num_workers': min(4, os.cpu_count() or 4),
            'pin_memory': False,
        })

    return config

def benchmark_dataloader_performance(dataset, batch_size: int = 64, test_configs: Optional[list] = None) -> dict:
    """
    Benchmark different DataLoader configurations to find the optimal one.

    Args:
        dataset: PyTorch dataset to benchmark
        batch_size: Batch size for testing
        test_configs: List of configurations to test (optional)

    Returns:
        dict: Best configuration and performance metrics
    """
    import time
    from torch.utils.data import DataLoader

    if test_configs is None:
        test_configs = [
            {'num_workers': 0, 'pin_memory': False},
            {'num_workers': 0, 'pin_memory': True},
            {'num_workers': 2, 'pin_memory': True},
            {'num_workers': 4, 'pin_memory': True},
        ]

    best_config = None
    best_throughput = 0
    results = {}

    logger.info("Benchmarking DataLoader configurations...")

    for config in test_configs:
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **config)

            # Warmup
            for i, batch in enumerate(dataloader):
                if i >= 3:
                    break

            # Benchmark
            start_time = time.time()
            batch_count = 0

            for batch in dataloader:
                batch_count += 1
                if batch_count >= 20:  # Test 20 batches
                    break

            elapsed = time.time() - start_time
            throughput = batch_count / elapsed

            results[str(config)] = throughput

            if throughput > best_throughput:
                best_throughput = throughput
                best_config = config

            logger.info(f"Config {config}: {throughput:.1f} batches/s")

        except Exception as e:
            logger.warning(f"Config {config} failed: {e}")

    logger.info(f"Best config: {best_config} ({best_throughput:.1f} batches/s)")
    return {'best_config': best_config, 'throughput': best_throughput, 'all_results': results}
