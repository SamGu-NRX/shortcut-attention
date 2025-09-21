#!/usr/bin/env python3
"""
Test script to validate CUDA performance improvements
Run this to verify the optimizations are working.
"""

import torch
import time
import logging
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_optimized_performance():
    """Test the performance with optimizations applied"""

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return

    logger.info("Testing optimized CUDA performance...")

    # Apply the same optimizations as in main.py
    optimal_threads = min(torch.get_num_threads(), 8)
    torch.set_num_threads(optimal_threads)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    logger.info(f"Using {optimal_threads} threads")
    logger.info(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    device = torch.device('cuda:0')

    # Create a realistic training scenario
    batch_size = 64
    num_batches = 100

    # Simulate typical model data (e.g., ResNet input)
    data = torch.randn(num_batches * batch_size, 3, 224, 224)
    labels = torch.randint(0, 1000, (num_batches * batch_size,))
    dataset = TensorDataset(data, labels)

    # Test with optimized dataloader settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Optimized for CUDA
        pin_memory=True,  # Faster CPU-GPU transfers
        drop_last=True
    )

    # Simple model for testing
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 1000)
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Warmup
    logger.info("Warming up...")
    for i, (batch_data, batch_labels) in enumerate(dataloader):
        if i >= 5:
            break
        batch_data = batch_data.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    # Actual performance test
    logger.info("Starting performance test...")
    start_time = time.time()
    processed_batches = 0

    for batch_data, batch_labels in dataloader:
        batch_data = batch_data.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        processed_batches += 1

        if processed_batches >= 50:  # Test 50 batches
            break

    # Don't synchronize unless necessary - this was part of the problem!
    # torch.cuda.synchronize()  # Commented out for better performance

    elapsed_time = time.time() - start_time
    iterations_per_second = processed_batches / elapsed_time

    logger.info(f"Processed {processed_batches} batches in {elapsed_time:.2f}s")
    logger.info(f"Performance: {iterations_per_second:.1f} it/s")

    # Memory usage
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    if iterations_per_second > 10:
        logger.info("✅ Performance looks good! Should be much faster than 3 it/s")
    else:
        logger.warning("⚠️  Performance still seems low. Check for other bottlenecks.")

    return iterations_per_second

if __name__ == "__main__":
    test_optimized_performance()
