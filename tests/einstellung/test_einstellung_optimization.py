#!/usr/bin/env python3
"""
Test the optimized Einstellung dataset performance
"""

import sys
import time
import torch
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_performance(dataset_class, name, **kwargs):
    """Test performance of a dataset class"""

    logger.info(f"\n{'='*50}")
    logger.info(f"Testing: {name}")
    logger.info(f"{'='*50}")

    try:
        # Create dataset
        dataset = dataset_class(
            root='./data/CIFAR100',
            train=True,
            download=True,
            apply_shortcut=True,  # Enable Einstellung processing
            patch_size=4,
            **kwargs
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,  # Single-threaded for fair comparison
            pin_memory=True
        )

        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Batch size: 32")

        # Benchmark data loading
        start_time = time.time()
        batch_count = 0

        for batch in dataloader:
            batch_count += 1
            if batch_count >= 20:  # Test 20 batches
                break

        elapsed = time.time() - start_time
        throughput = batch_count / elapsed

        logger.info(f"Performance: {throughput:.1f} batches/s")
        logger.info(f"Estimated full epoch time: {len(dataloader) / throughput:.1f}s")

        return throughput

    except Exception as e:
        logger.error(f"Failed to test {name}: {e}")
        return 0

def main():
    """Compare original vs optimized Einstellung dataset"""

    logger.info("Comparing Einstellung dataset performance...")

    # Test original dataset
    try:
        from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung
        original_throughput = test_dataset_performance(
            MyCIFAR100Einstellung,
            "Original Einstellung Dataset"
        )
    except Exception as e:
        logger.error(f"Could not test original dataset: {e}")
        original_throughput = 0

    # Test optimized dataset
    try:
        from datasets.seq_cifar100_einstellung_optimized import OptimizedMyCIFAR100Einstellung
        optimized_throughput = test_dataset_performance(
            OptimizedMyCIFAR100Einstellung,
            "Optimized Einstellung Dataset"
        )
    except Exception as e:
        logger.error(f"Could not test optimized dataset: {e}")
        optimized_throughput = 0

    # Compare results
    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE COMPARISON")
    logger.info(f"{'='*60}")

    if original_throughput > 0 and optimized_throughput > 0:
        speedup = optimized_throughput / original_throughput
        logger.info(f"Original dataset:  {original_throughput:.1f} batches/s")
        logger.info(f"Optimized dataset: {optimized_throughput:.1f} batches/s")
        logger.info(f"Speedup: {speedup:.1f}x faster")

        if speedup > 2:
            logger.info("✅ Significant performance improvement!")
        elif speedup > 1.2:
            logger.info("✅ Good performance improvement!")
        else:
            logger.info("⚠️  Modest improvement - may need further optimization")
    else:
        logger.warning("Could not complete comparison")

    # Provide usage instructions
    logger.info(f"\n📖 USAGE INSTRUCTIONS:")
    logger.info(f"To use the optimized dataset in your experiments:")
    logger.info(f"1. The optimized dataset will build a cache on first use")
    logger.info(f"2. Subsequent runs will be much faster")
    logger.info(f"3. Cache is stored in ./data/CIFAR100/einstellung_cache/")
    logger.info(f"4. Delete cache if you change patch parameters")

if __name__ == "__main__":
    main()
