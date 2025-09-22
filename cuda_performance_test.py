#!/usr/bin/env python3
"""
CUDA Performance Diagnostic Script
This script tests various CUDA performance aspects to identify bottlenecks.
"""

import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_cuda_performance():
    """Test basic CUDA operations performance"""
    logger.info("=== Basic CUDA Performance Test ===")

    device = torch.device('cuda:0')
    logger.info(f"Using device: {device}")
    logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"Device capability: {torch.cuda.get_device_capability(0)}")

    # Test matrix multiplication
    size = 4096
    logger.info(f"Testing matrix multiplication ({size}x{size})...")

    # CPU test
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    start_time = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    logger.info(f"CPU time: {cpu_time:.4f}s")

    # GPU test
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)

    # Warm up
    for _ in range(3):
        _ = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()

    start_time = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    logger.info(f"GPU time: {gpu_time:.4f}s")
    logger.info(f"Speedup: {cpu_time/gpu_time:.2f}x")

    return gpu_time, cpu_time

def test_dataloader_performance():
    """Test DataLoader performance with different configurations"""
    logger.info("\n=== DataLoader Performance Test ===")

    device = torch.device('cuda:0')
    batch_size = 256
    num_samples = 10000

    # Create dummy data
    data = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 1000, (num_samples,))
    dataset = TensorDataset(data, labels)

    configs = [
        {"num_workers": 0, "pin_memory": False, "name": "0 workers, no pin_memory"},
        {"num_workers": 0, "pin_memory": True, "name": "0 workers, pin_memory"},
        {"num_workers": 4, "pin_memory": False, "name": "4 workers, no pin_memory"},
        {"num_workers": 4, "pin_memory": True, "name": "4 workers, pin_memory"},
        {"num_workers": 8, "pin_memory": True, "name": "8 workers, pin_memory"},
        {"num_workers": 16, "pin_memory": True, "name": "16 pin"}
    ]

    for config in configs:
        logger.info(f"\nTesting: {config['name']}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )

        start_time = time.time()
        total_batches = 0

        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            # Simulate some computation
            _ = torch.sum(batch_data)
            total_batches += 1

            if total_batches >= 500:  # Test first 50 batches
                break

        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        throughput = total_batches / elapsed_time

        logger.info(f"  Time: {elapsed_time:.4f}s, Throughput: {throughput:.2f} batches/s")

def test_thread_settings():
    """Test different thread settings"""
    logger.info("\n=== Thread Settings Test ===")

    logger.info(f"Current torch.get_num_threads(): {torch.get_num_threads()}")
    logger.info(f"Current torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")

    # Test different thread counts
    thread_counts = [1, 2, 4, 8, 16]
    device = torch.device('cuda:0')

    for num_threads in thread_counts:
        torch.set_num_threads(num_threads)
        logger.info(f"\nTesting with {num_threads} threads...")

        # Test CPU-GPU data transfer
        data = torch.randn(1000, 3, 224, 224)

        start_time = time.time()
        for _ in range(10):
            gpu_data = data.to(device, non_blocking=True)
            torch.cuda.synchronize()
        transfer_time = time.time() - start_time

        logger.info(f"  Data transfer time: {transfer_time:.4f}s")

def test_memory_usage():
    """Test memory allocation patterns"""
    logger.info("\n=== Memory Usage Test ===")

    device = torch.device('cuda:0')

    logger.info(f"Initial memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    logger.info(f"Initial memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # Allocate some tensors
    tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000).to(device)
        tensors.append(tensor)

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"After tensor {i+1}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # Clear memory
    del tensors
    torch.cuda.empty_cache()

    final_allocated = torch.cuda.memory_allocated(0) / 1024**3
    final_reserved = torch.cuda.memory_reserved(0) / 1024**3
    logger.info(f"After cleanup: Allocated: {final_allocated:.2f} GB, Reserved: {final_reserved:.2f} GB")

def test_cuda_synchronization():
    """Test impact of CUDA synchronization"""
    logger.info("\n=== CUDA Synchronization Test ===")

    device = torch.device('cuda:0')
    data = torch.randn(1000, 1000).to(device)

    # Test without synchronization
    start_time = time.time()
    for _ in range(100):
        result = torch.mm(data, data)
    no_sync_time = time.time() - start_time
    logger.info(f"Without synchronization: {no_sync_time:.4f}s")

    # Test with synchronization
    start_time = time.time()
    for _ in range(100):
        result = torch.mm(data, data)
        torch.cuda.synchronize()
    sync_time = time.time() - start_time
    logger.info(f"With synchronization: {sync_time:.4f}s")
    logger.info(f"Synchronization overhead: {sync_time - no_sync_time:.4f}s")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        exit(1)

    logger.info("Starting CUDA performance diagnostics...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")

    test_basic_cuda_performance()
    test_dataloader_performance()
    test_thread_settings()
    test_memory_usage()
    test_cuda_synchronization()

    logger.info("\n=== Diagnostics Complete ===")
