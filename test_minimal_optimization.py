#!/usr/bin/env python3
"""
Test minimal CUDA optimizations vs PyTorch defaults
"""

import torch
import time
import logging
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_configuration(config_name: str, setup_func=None):
    """Benchmark a specific configuration"""

    logger.info(f"\n{'='*50}")
    logger.info(f"Testing: {config_name}")
    logger.info(f"{'='*50}")

    # Reset PyTorch state
    torch.cuda.empty_cache()

    # Apply configuration
    if setup_func:
        setup_func()

    # Show current settings
    logger.info(f"torch.get_num_threads(): {torch.get_num_threads()}")
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")

    device = torch.device('cuda:0')

    # Create test scenario
    batch_size = 64
    data = torch.randn(1000, 3, 224, 224)
    labels = torch.randint(0, 1000, (1000,))
    dataset = TensorDataset(data, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep consistent
        pin_memory=True
    )

    # Simple model
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
    for i, (batch_data, batch_labels) in enumerate(dataloader):
        if i >= 3:
            break
        batch_data = batch_data.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    # Benchmark
    start_time = time.time()
    batch_count = 0

    for batch_data, batch_labels in dataloader:
        batch_data = batch_data.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        batch_count += 1
        if batch_count >= 15:  # Test 15 batches
            break

    elapsed = time.time() - start_time
    throughput = batch_count / elapsed

    logger.info(f"Performance: {throughput:.1f} it/s")
    logger.info(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    return throughput

def setup_pytorch_defaults():
    """Reset to PyTorch defaults"""
    # Don't change thread count - use whatever PyTorch set
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def setup_minimal_optimizations():
    """Apply minimal optimizations"""
    from utils.minimal_cuda_optimization import apply_minimal_cuda_optimizations
    apply_minimal_cuda_optimizations(1)

def setup_aggressive_optimizations():
    """Apply the previous aggressive optimizations"""
    torch.set_num_threads(16)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        exit(1)

    logger.info("Benchmarking different CUDA optimization approaches...")

    results = {}

    # Test configurations
    configs = [
        ("PyTorch Defaults", setup_pytorch_defaults),
        ("Minimal Optimizations", setup_minimal_optimizations),
        ("Aggressive Optimizations", setup_aggressive_optimizations),
    ]

    for name, setup_func in configs:
        try:
            throughput = benchmark_configuration(name, setup_func)
            results[name] = throughput
        except Exception as e:
            logger.error(f"Failed to test {name}: {e}")
            results[name] = 0

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE COMPARISON")
    logger.info(f"{'='*60}")

    best_throughput = max(results.values())
    for name, throughput in results.items():
        improvement = f"({throughput/best_throughput*100:.0f}%)" if throughput > 0 else "(FAILED)"
        logger.info(f"{name:<25}: {throughput:>6.1f} it/s {improvement}")

    # Recommendation
    best_config = max(results.items(), key=lambda x: x[1])
    logger.info(f"\n🏆 Best configuration: {best_config[0]} ({best_config[1]:.1f} it/s)")

    if best_config[1] > 10:
        logger.info("✅ Good performance achieved!")
    else:
        logger.warning("⚠️  Performance still low - may need different approach")
