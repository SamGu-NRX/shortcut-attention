#!/usr/bin/env python3
"""
Test the automatic performance optimizations
"""

import torch
import logging
import time
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_auto_optimizations():
    """Test automatic performance optimizations"""

    logger.info("Testing automatic performance optimizations...")

    # Import and apply auto-optimizations
    try:
        from utils.auto_performance import auto_optimize_pytorch_performance, get_optimal_dataloader_config

        # Apply optimizations
        optimizations = auto_optimize_pytorch_performance()
        logger.info(f"Applied optimizations: {optimizations}")

        # Test with a realistic training scenario
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Create test data
        batch_size = 64
        data = torch.randn(1000, 3, 224, 224)
        labels = torch.randint(0, 1000, (1000,))
        dataset = TensorDataset(data, labels)

        # Get optimal dataloader config
        optimal_config = get_optimal_dataloader_config(batch_size, len(dataset))
        logger.info(f"Optimal DataLoader config: {optimal_config}")

        # Create optimized dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **optimal_config)

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

        # Performance test
        logger.info("Running performance test...")
        start_time = time.time()

        for i, (batch_data, batch_labels) in enumerate(dataloader):
            if i >= 20:  # Test 20 batches
                break

            batch_data = batch_data.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        elapsed = time.time() - start_time
        throughput = 20 / elapsed

        logger.info(f"Performance: {throughput:.1f} it/s")

        if throughput > 15:
            logger.info("✅ Excellent performance! Auto-optimizations working well.")
        elif throughput > 10:
            logger.info("✅ Good performance! Significant improvement expected.")
        else:
            logger.warning("⚠️  Performance still low. Check for other bottlenecks.")

        return throughput

    except ImportError as e:
        logger.error(f"Could not import auto_performance module: {e}")
        return None

if __name__ == "__main__":
    test_auto_optimizations()
