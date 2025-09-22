#!/usr/bin/env python3
"""
Validation script for performance monitoring implementation.
This validates that all required features are implemented correctly.
"""

import os
import sys

#t root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def validate_implementation():
    """Validate that all task requirements are implemented."""
    print("🔍 Validating Performance Monitoring Implementation")
    print("=" * 60)

    validation_results = {}

    # 1. Cache hit/miss tracking and performance metrics reporting
    print("1. Testing cache hit/miss tracking and metrics reporting...")
    try:
        from utils.einstellung_performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor("validation_test")

        # Test hit/miss tracking
        monitor.record_cache_hit(0.001)
        monitor.record_cache_miss(0.005)

        # Test metrics reporting
        report = monitor.get_performance_report()

        required_metrics = [
            'cache_metrics', 'timing_metrics', 'memory_metrics',
            'batch_metrics', 'performance_indicators'
        ]

        for metric in required_metrics:
            if metric not in report:
                raise ValueError(f"Missing metric: {metric}")

        # Validate specific cache metrics
        cache_metrics = report['cache_metrics']
        if cache_metrics['hits'] != 1 or cache_metrics['misses'] != 1:
            raise ValueError("Cache hit/miss tracking not working correctly")

        if abs(cache_metrics['hit_rate'] - 0.5) > 0.01:
            raise ValueError("Hit rate calculation incorrect")

        validation_results["cache_hit_miss_tracking"] = True
        print("   ✅ Cache hit/miss tracking and metrics reporting: PASSED")

    except Exception as e:
        validation_results["cache_hit_miss_tracking"] = False
        print(f"   ❌ Cache hit/miss tracking: FAILED - {e}")

    # 2. Memory usage monitoring during cache operations
    print("2. Testing memory usage monitoring...")
    try:
        from utils.einstellung_performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor("memory_validation")

        # Test memory snapshot functionality
        snapshot = monitor.take_memory_snapshot()

        required_snapshot_fields = ['timestamp', 'rss_mb', 'vms_mb', 'percent', 'available_mb']
        for field in required_snapshot_fields:
            if not hasattr(snapshot, field):
                raise ValueError(f"Missing snapshot field: {field}")

        # Test memory trend analysis
        for i in range(5):
            monitor.take_memory_snapshot()

        trend = monitor.get_memory_usage_trend()
        required_trend_fields = ['trend', 'change_mb', 'rate_mb_per_min']
        for field in required_trend_fields:
            if field not in trend:
                raise ValueError(f"Missing trend field: {field}")

        validation_results["memory_monitoring"] = True
        print("   ✅ Memory usage monitoring: PASSED")

    except Exception as e:
        validation_results["memory_monitoring"] = False
        print(f"   ❌ Memory usage monitoring: FAILED - {e}")

    # 3. Optimized cache loading for minimal memory footprint
    print("3. Testing optimized cache loading...")
    try:
        from utils.einstellung_performance_monitor import OptimizedCacheLoader, PerformanceMonitor
        import pickle
        import numpy as np

        monitor = PerformanceMonitor("loader_validation")
        loader = OptimizedCacheLoader(monitor)

        # Create test cache
        test_cache_path = "validation_cache.pkl"
        test_data = {
            'raw_images': np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
            'raw_targets': list(range(10)),
            'cache_key': 'validation_key',
            'version': '2.0'
        }

        try:
            with open(test_cache_path, 'wb') as f:
                pickle.dump(test_data, f)

            # Test optimized loading
            loaded_data = loader.load_cache_optimized(test_cache_path)

            if loaded_data is None:
                raise ValueError("Failed to load cache")

            # Test memory mapping decision
            should_use_mmap = monitor.should_use_memory_mapping(2000)  # 2GB cache
            if not isinstance(should_use_mmap, bool):
                raise ValueError("Memory mapping decision not boolean")

            validation_results["optimized_loading"] = True
            print("   ✅ Optimized cache loading: PASSED")

        finally:
            if os.path.exists(test_cache_path):
                os.remove(test_cache_path)

    except Exception as e:
        validation_results["optimized_loading"] = False
        print(f"   ❌ Optimized cache loading: FAILED - {e}")

    # 4. Configurable batch processing for large datasets
    print("4. Testing configurable batch processing...")
    try:
        from utils.einstellung_performance_monitor import PerformanceMonitor, OptimizedCacheLoader
        import numpy as np

        monitor = PerformanceMonitor("batch_validation")
        loader = OptimizedCacheLoader(monitor)

        # Test batch size optimization
        optimal_batch = monitor.optimize_batch_size(10000, 1024)  # 10k items, 1GB memory
        if not isinstance(optimal_batch, int) or optimal_batch <= 0:
            raise ValueError("Invalid batch size optimization")

        # Test batch processing
        test_data = {
            'processed_images': np.random.randint(0, 255, (50, 32, 32, 3), dtype=np.uint8),
            'targets': list(range(50))
        }

        def test_processor(images, targets):
            return [f"processed_{i}" for i in range(len(images))]

        results = loader.process_cache_in_batches(test_data, test_processor, batch_size=10)

        if len(results) != 50:
            raise ValueError(f"Batch processing returned wrong number of results: {len(results)}")

        # Test batch operation recording
        monitor.record_batch_operation(25, 0.1)
        report = monitor.get_performance_report()

        if report['batch_metrics']['operations'] != 6:  # 5 from processing + 1 from manual record
            raise ValueError("Batch operation recording not working correctly")

        validation_results["batch_processing"] = True
        print("   ✅ Configurable batch processing: PASSED")

    except Exception as e:
        validation_results["batch_processing"] = False
        print(f"   ❌ Configurable batch processing: FAILED - {e}")

    # 5. Integration with robust cache system
    print("5. Testing integration with robust cache system...")
    try:
        from utils.robust_einstellung_cache import RobustEinstellungCache
        import numpy as np

        # Create cache instance
        cache = RobustEinstellungCache(
            dataset_name="validation_cache",
            train=True,
            apply_shortcut=True,
            mask_shortcut=False,
            patch_size=4,
            patch_color=(255, 0, 255),
            resolution="32x32"
        )

        # Verify performance monitor is integrated
        if not hasattr(cache, 'performance_monitor'):
            raise ValueError("Performance monitor not integrated into cache")

        if not hasattr(cache, 'optimized_loader'):
            raise ValueError("Optimized loader not integrated into cache")

        # Test cache info includes performance metrics
        test_images = np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        test_targets = list(range(5))

        success = cache.load_or_build_cache(test_images, test_targets)
        if not success:
            raise ValueError("Failed to build cache")

        cache_info = cache.get_cache_info()
        if 'performance_metrics' not in cache_info:
            raise ValueError("Performance metrics not included in cache info")

        # Test performance summary methods
        perf_summary = cache.get_performance_summary()
        if not isinstance(perf_summary, dict):
            raise ValueError("Performance summary not returning dict")

        # Clean up
        if os.path.exists(cache._cache_path):
            os.remove(cache._cache_path)

        validation_results["cache_integration"] = True
        print("   ✅ Robust cache integration: PASSED")

    except Exception as e:
        validation_results["cache_integration"] = False
        print(f"   ❌ Robust cache integration: FAILED - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(validation_results.values())
    total = len(validation_results)

    for feature, passed_test in validation_results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} {feature}")

    print(f"\nOverall: {passed}/{total} features validated")

    if passed == total:
        print("\n🎉 ALL TASK REQUIREMENTS SUCCESSFULLY IMPLEMENTED!")
        print("\nImplemented Features:")
        print("✅ Cache hit/miss tracking and performance metrics reporting")
        print("✅ Memory usage monitoring during cache operations")
        print("✅ Optimized cache loading for minimal memory footprint")
        print("✅ Configurable batch processing for large datasets")
        print("✅ Full integration with existing robust cache system")

        print("\nPerformance Optimizations:")
        print("✅ Automatic batch size optimization based on available memory")
        print("✅ Memory mapping for large cache files")
        print("✅ Memory usage trend analysis")
        print("✅ Comprehensive performance reporting")

        return True
    else:
        print(f"\n⚠️  {total - passed} features need attention!")
        return False


if __name__ == '__main__':
    success = validate_implementation()
    sys.exit(0 if success else 1)
