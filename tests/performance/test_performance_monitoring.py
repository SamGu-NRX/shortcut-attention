#!/usr/bin/env python3
"""
Test script for Einstellung Dataset Performance Monitoring

This script validates the performance monitoring and optimization features
implemented for the Einstellung dataset caching system.
"""

import os
import sys
import time
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.einstellung_performance_monitor import (
    PerformanceMonitor,
    OptimizedCacheLoader,
    benchmark_cache_performance,
    get_performance_monitor
)
from utils.robust_einstellung_cache import (
    RobustEinstellungCache,
    RobustEinstellungDatasetMixin,
    validate_robust_cache_consistency
)
import numpy as np
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_performance_monitor():
    """Test basic performance monitoring functionality."""
    print("🔧 Testing Performance Monitor")
    print("-" * 50)

    # Create test monitor
    monitor = PerformanceMonitor("test_dataset", enable_detailed_logging=True)

    # Test cache operations
    print("Testing cache operations...")
    monitor.record_cache_build(2.5, 150.0)
    monitor.record_cache_load(0.8, 150.0)

    # Test item access tracking
    print("Testing item access tracking...")
    for i in range(100):
        access_time = 0.001 + (i % 10) * 0.0001  # Simulate varying access times
        if i % 10 == 0:
            monitor.record_cache_miss(access_time)
        else:
            monitor.record_cache_hit(access_time)

        # Take memory snapshots periodically
        if i % 20 == 0:
            monitor.take_memory_snapshot()

    # Test batch processing
    print("Testing batch processing...")
    for batch in range(5):
        batch_size = 50 + batch * 10
        batch_time = 0.1 + batch * 0.02
        monitor.record_batch_operation(batch_size, batch_time)

    # Test error recording
    monitor.record_cache_error("test_error", "This is a test error")

    # Generate and display report
    print("\nPerformance Report:")
    monitor.print_performance_summary()

    # Test memory trend analysis
    trend = monitor.get_memory_usage_trend(window_minutes=1)
    print(f"Memory trend: {trend}")

    # Test batch size optimization
    optimal_batch = monitor.optimize_batch_size(10000, 2048)  # 10k items, 2GB memory
    print(f"Optimal batch size: {optimal_batch}")

    print("✅ Performance monitor test completed\n")
    return True


def test_optimized_cache_loader():
    """Test optimized cache loader functionality."""
    print("🚀 Testing Optimized Cache Loader")
    print("-" * 50)

    monitor = PerformanceMonitor("loader_test")
    loader = OptimizedCacheLoader(monitor)

    # Create a dummy cache file for testing
    import pickle
    import numpy as np

    test_cache_path = "test_cache.pkl"
    test_data = {
        'raw_images': np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8),
        'raw_targets': list(range(100)),
        'cache_key': 'test_key',
        'version': '2.0'
    }

    try:
        # Save test cache
        with open(test_cache_path, 'wb') as f:
            pickle.dump(test_data, f)

        print(f"Created test cache: {test_cache_path}")

        # Test optimized loading
        loaded_data = loader.load_cache_optimized(test_cache_path, use_memory_mapping=False)

        if loaded_data is not None:
            print(f"✅ Successfully loaded cache with {len(loaded_data['raw_images'])} items")

            # Test batch processing
            def dummy_processor(images, targets):
                # Simulate some processing
                time.sleep(0.001)
                return [f"processed_{i}" for i in range(len(images))]

            results = loader.process_cache_in_batches(
                loaded_data, dummy_processor, batch_size=25
            )

            print(f"✅ Batch processing completed: {len(results)} results")
        else:
            print("❌ Failed to load test cache")
            return False

    finally:
        # Clean up test file
        if os.path.exists(test_cache_path):
            os.remove(test_cache_path)
            print(f"Cleaned up test cache: {test_cache_path}")

    print("✅ Optimized cache loader test completed\n")
    return True


def test_robust_cache_with_monitoring():
    """Test robust cache system with performance monitoring."""
    print("🎯 Testing Robust Cache with Performance Monitoring")
    print("-" * 50)

    # Create test data
    import numpy as np
    test_images = np.random.randint(0, 255, (50, 32, 32, 3), dtype=np.uint8)
    test_targets = list(range(50))

    # Create robust cache instance
    cache = RobustEinstellungCache(
        dataset_name="test_robust_cache",
        train=True,
        apply_shortcut=True,
        mask_shortcut=False,
        patch_size=4,
        patch_color=(255, 0, 255),
        resolution="32x32"
    )

    try:
        # Test cache building with monitoring
        print("Building cache with performance monitoring...")
        success = cache.load_or_build_cache(test_images, test_targets)

        if success:
            print("✅ Cache built successfully")

            # Test item access with monitoring
            print("Testing item access with monitoring...")
            for i in range(min(10, len(test_images))):
                item = cache.get_processed_item(
                    index=i,
                    shortcut_labels={0, 1, 2, 3, 4}  # First 5 classes get shortcuts
                )
                if i == 0:
                    print(f"✅ Successfully accessed item {i}")

            # Get cache info with performance metrics
            cache_info = cache.get_cache_info()
            print(f"Cache info keys: {list(cache_info.keys())}")

            # Print performance summary
            print("\nCache Performance Summary:")
            cache.print_performance_summary()

        else:
            print("❌ Failed to build cache")
            return False

    finally:
        # Clean up cache file
        if hasattr(cache, '_cache_path') and os.path.exists(cache._cache_path):
            os.remove(cache._cache_path)
            print(f"Cleaned up cache file: {cache._cache_path}")

    print("✅ Robust cache with monitoring test completed\n")
    return True


def test_mixin_performance_features():
    """Test performance features in the dataset mixin."""
    print("🔗 Testing Dataset Mixin Performance Features")
    print("-" * 50)

    # Create a test class that uses the mixin
    class TestDataset(RobustEinstellungDatasetMixin):
        def __init__(self):
            self.train = True
            self.apply_shortcut = True
            self.mask_shortcut = False
            self.patch_size = 4
            self.patch_color = np.array([255, 0, 255], dtype=np.uint8)
            self.data = np.random.randint(0, 255, (20, 32, 32, 3), dtype=np.uint8)

            # Initialize robust cache
            self.init_robust_cache()

    try:
        # Create test dataset
        import numpy as np
        dataset = TestDataset()

        # Test cache setup
        test_images = np.random.randint(0, 255, (20, 32, 32, 3), dtype=np.uint8)
        test_targets = list(range(20))

        success = dataset.setup_robust_cache(test_images, test_targets)

        if success:
            print("✅ Mixin cache setup successful")

            # Test performance features
            cache_info = dataset.get_robust_cache_info()
            print(f"✅ Cache info retrieved: {len(cache_info)} keys")

            # Test performance summary
            perf_summary = dataset.get_cache_performance_summary()
            print(f"✅ Performance summary retrieved: {len(perf_summary)} keys")

            # Test optimization
            dataset.optimize_cache_settings()
            print("✅ Cache settings optimized")

            # Print performance summary
            print("\nDataset Performance Summary:")
            dataset.print_cache_performance_summary()

        else:
            print("❌ Failed to setup mixin cache")
            return False

    except Exception as e:
        print(f"❌ Mixin test failed: {e}")
        return False

    print("✅ Dataset mixin performance test completed\n")
    return True


def test_memory_monitoring():
    """Test memory monitoring capabilities."""
    print("💾 Testing Memory Monitoring")
    print("-" * 50)

    monitor = PerformanceMonitor("memory_test", enable_detailed_logging=True)

    # Take initial snapshot
    initial_snapshot = monitor.take_memory_snapshot()
    print(f"Initial memory: {initial_snapshot.rss_mb:.1f}MB RSS, {initial_snapshot.percent:.1f}%")

    # Simulate memory-intensive operations
    data_blocks = []
    for i in range(5):
        # Allocate some memory
        block = np.random.random((1000, 1000))  # ~8MB per block
        data_blocks.append(block)

        # Take snapshot
        snapshot = monitor.take_memory_snapshot()
        print(f"After block {i+1}: {snapshot.rss_mb:.1f}MB RSS, {snapshot.percent:.1f}%")

        time.sleep(0.1)  # Small delay

    # Test memory trend analysis
    trend = monitor.get_memory_usage_trend(window_minutes=1)
    print(f"\nMemory trend analysis:")
    print(f"  Trend: {trend['trend']}")
    print(f"  Change: {trend['change_mb']:+.1f}MB")
    print(f"  Rate: {trend['rate_mb_per_min']:+.1f}MB/min")

    # Clean up
    del data_blocks

    final_snapshot = monitor.take_memory_snapshot()
    print(f"Final memory: {final_snapshot.rss_mb:.1f}MB RSS, {final_snapshot.percent:.1f}%")

    print("✅ Memory monitoring test completed\n")
    return True


def run_comprehensive_test():
    """Run comprehensive test of all performance monitoring features."""
    print("🚀 Comprehensive Performance Monitoring Test")
    print("=" * 60)

    tests = [
        ("Performance Monitor", test_performance_monitor),
        ("Optimized Cache Loader", test_optimized_cache_loader),
        ("Robust Cache with Monitoring", test_robust_cache_with_monitoring),
        ("Dataset Mixin Performance", test_mixin_performance_features),
        ("Memory Monitoring", test_memory_monitoring),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            start_time = time.time()

            success = test_func()

            duration = time.time() - start_time
            results[test_name] = {
                "success": success,
                "duration": duration
            }

            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status} - {test_name} ({duration:.2f}s)")

        except Exception as e:
            results[test_name] = {
                "success": False,
                "duration": 0,
                "error": str(e)
            }
            print(f"❌ FAILED - {test_name}: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = result["duration"]
        error = result.get("error", "")

        print(f"{status} {test_name:<30} ({duration:.2f}s) {error}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All performance monitoring tests PASSED!")
        return True
    else:
        print("⚠️  Some performance monitoring tests FAILED!")
        return False


if __name__ == '__main__':
    # Run comprehensive test
    success = run_comprehensive_test()

    if success:
        print("\n✅ Performance monitoring implementation is ready!")
        print("   - Cache hit/miss tracking: ✅")
        print("   - Memory usage monitoring: ✅")
        print("   - Optimized cache loading: ✅")
        print("   - Configurable batch processing: ✅")
        print("   - Performance metrics reporting: ✅")
    else:
        print("\n❌ Performance monitoring implementation needs fixes!")

    sys.exit(0 if success else 1)
