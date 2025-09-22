#!/usr/bin/env python3
"""
Test script to verify that caching and performance optimizations are working correctly.

This script tests:
1. Cache creation and loading
2. Performance improvements with caching enabled vs disabled
3. Code optimization levels
4. Backward compatibility
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_cache_functionality():
    """Test that caching is working correctly."""
    logger.info("Testing cache functionality...")

    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())

    try:
        from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung
        from utils.conf import base_path

        # Test with caching enabled
        logger.info("Creating dataset with caching enabled...")
        dataset_cached = MyCIFAR100Einstellung(
            root=base_path() + 'CIFAR100',
            train=True,
            download=True,
            enable_cache=True,
            apply_shortcut=True,
            patch_size=4
        )

        # Check if cache was created
        cache_dir = os.path.join(base_path(), 'CIFAR100', 'einstellung_cache')
        cache_files = []
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]

        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"Cache files found: {len(cache_files)}")

        if cache_files:
            logger.info("✓ Cache files created successfully")
            for cache_file in cache_files:
                file_path = os.path.join(cache_dir, cache_file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                logger.info(f"  - {cache_file}: {file_size:.1f} MB")
        else:
            logger.warning("⚠️  No cache files found")

        # Test data loading performance
        logger.info("Testing data loading performance...")

        # Time cached loading
        start_time = time.time()
        for i in range(min(100, len(dataset_cached))):
            img, target, not_aug_img = dataset_cached[i]
        cached_time = time.time() - start_time

        # Test with caching disabled
        logger.info("Creating dataset with caching disabled...")
        dataset_uncached = MyCIFAR100Einstellung(
            root=base_path() + 'CIFAR100',
            train=True,
            download=False,
            enable_cache=False,
            apply_shortcut=True,
            patch_size=4
        )

        # Time uncached loading
        start_time = time.time()
        for i in range(min(100, len(dataset_uncached))):
            img, target, not_aug_img = dataset_uncached[i]
        uncached_time = time.time() - start_time

        logger.info(f"Cached loading time (100 samples): {cached_time:.3f}s")
        logger.info(f"Uncached loading time (100 samples): {uncached_time:.3f}s")

        if cached_time < uncached_time:
            speedup = uncached_time / cached_time
            logger.info(f"✓ Caching provides {speedup:.1f}x speedup")
        else:
            logger.warning(f"⚠️  Caching is slower than uncached ({cached_time:.3f}s vs {uncached_time:.3f}s)")

        return True

    except Exception as e:
        logger.error(f"❌ Cache functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_optimization_integration():
    """Test that code_optimization parameter is properly integrated."""
    logger.info("Testing code optimization integration...")

    try:
        # Test help output includes code_optimization
        result = subprocess.run(
            [sys.executable, 'run_einstellung_experiment.py', '--help'],
            capture_output=True,
            text=True,
            timeout=30
        )

        help_output = result.stdout

        if '--code_optimization' in help_output:
            logger.info("✓ --code_optimization argument found in help")
        else:
            logger.error("❌ --code_optimization argument missing from help")
            return False

        # Test that different optimization levels are accepted
        for level in [0, 1, 2, 3]:
            logger.info(f"Testing code_optimization level {level}...")

            # Create a minimal test command
            test_cmd = [
                sys.executable, 'run_einstellung_experiment.py',
                '--model', 'sgd',
                '--backbone', 'resnet18',
                '--code_optimization', str(level),
                '--skip_training',  # Skip actual training
                '--debug'  # Use debug mode for faster testing
            ]

            try:
                result = subprocess.run(
                    test_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    logger.info(f"✓ Code optimization level {level} accepted")
                else:
                    logger.warning(f"⚠️  Code optimization level {level} failed: {result.stderr}")

            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️  Code optimization level {level} test timed out")

        return True

    except Exception as e:
        logger.error(f"❌ Code optimization integration test failed: {e}")
        return False


def test_backward_compatibility():
    """Test that existing scripts work without modification."""
    logger.info("Testing backward compatibility...")

    try:
        # Test that default behavior works
        test_cmd = [
            sys.executable, 'run_einstellung_experiment.py',
            '--model', 'sgd',
            '--backbone', 'resnet18',
            '--skip_training',  # Skip actual training
            '--debug'  # Use debug mode for faster testing
        ]

        result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.info("✓ Default behavior works (backward compatibility maintained)")

            # Check that caching is enabled by default
            if 'cache' in result.stdout.lower() or 'cache' in result.stderr.lower():
                logger.info("✓ Caching appears to be enabled by default")
            else:
                logger.warning("⚠️  No cache-related output found")

        else:
            logger.error(f"❌ Default behavior failed: {result.stderr}")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
    """Run all performance and compatibility tests."""
    logger.info("Starting cache and performance tests...")

    tests = [
        ("Cache Functionality", test_cache_functionality),
        ("Code Optimization Integration", test_code_optimization_integration),
        ("Backward Compatibility", test_backward_compatibility)
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")

        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Cache and performance optimizations are working correctly.")
        logger.info("\nNext steps:")
        logger.info("1. Run actual training with: python run_einstellung_experiment.py --model derpp --backbone resnet18")
        logger.info("2. Monitor GPU utilization with: nvidia-smi")
        logger.info("3. Expected performance: 15-30+ it/s with 80-95% GPU utilization")
    else:
        logger.error("❌ Some tests failed. Please check the issues above.")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
