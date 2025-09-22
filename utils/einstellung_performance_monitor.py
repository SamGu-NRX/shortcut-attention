#!/usr/bin/env python3
"""
Einstellung Dataset Performance Monitoring and Optimization

This module provides comprehensive performance monitoring and optimization
features for the Einstellung dataset caching system.

Key Features:
- Cache hit/miss tracking and metrics reporting
- Memory usage monduring cache operations
- Optimized cache loading with minimal memory footprint
- Configurable batch processing for large datasets
- Performance benchmarking and reporting
"""

import os
import sys
import time
import psutil
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class CacheMetrics:
    """Container for cache performance metrics."""
    cache_hits: int = 0
    cache_misses: int = 0
    cache_builds: int = 0
    cache_loads: int = 0
    cache_errors: int = 0

    # Timing metrics (in seconds)
    total_cache_build_time: float = 0.0
    total_cache_load_time: float = 0.0
    total_item_access_time: float = 0.0

    # Memory metrics (in MB)
    peak_memory_usage: float = 0.0
    cache_size_mb: float = 0.0

    # Performance metrics
    items_processed: int = 0
    avg_item_access_time_ms: float = 0.0
    cache_hit_rate: float = 0.0

    # Batch processing metrics
    batch_operations: int = 0
    avg_batch_size: float = 0.0
    total_batch_time: float = 0.0

    def update_hit_rate(self):
        """Update cache hit rate based on hits and misses."""
        total_accesses = self.cache_hits + self.cache_misses
        if total_accesses > 0:
            self.cache_hit_rate = self.cache_hits / total_accesses
        else:
            self.cache_hit_rate = 0.0

    def update_avg_access_time(self):
        """Update average item access time."""
        if self.items_processed > 0:
            self.avg_item_access_time_ms = (self.total_item_access_time * 1000) / self.items_processed
        else:
            self.avg_item_access_time_ms = 0.0


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float  # Available system memory


class PerformanceMonitor:
    """
    Performance monitoring system for Einstellung dataset caching.

    Provides comprehensive tracking of cache performance, memory usage,
    and optimization metrics.
    """

    def __init__(self, dataset_name: str = "einstellung", enable_detailed_logging: bool = True):
        """
        Initialize performance monitor.

        Args:
            dataset_name: Name of the dataset being monitored
            enable_detailed_logging: Whether to enable detailed performance logging
        """
        self.dataset_name = dataset_name
        self.enable_detailed_logging = enable_detailed_logging

        # Metrics tracking
        self.metrics = CacheMetrics()
        self.memory_snapshots: List[MemorySnapshot] = []

        # Thread safety
        self._lock = threading.Lock()

        # Process monitoring
        self.process = psutil.Process()

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{dataset_name}")

        # Performance optimization settings
        self.batch_size = 1000  # Default batch size for processing
        self.memory_limit_mb = 4096  # Memory limit for cache operations (4GB)
        self.enable_memory_mapping = True  # Use memory mapping for large caches

        self.logger.info(f"Performance monitor initialized for {dataset_name}")

    def record_cache_hit(self, access_time: float = 0.0):
        """Record a cache hit."""
        with self._lock:
            self.metrics.cache_hits += 1
            self.metrics.total_item_access_time += access_time
            self.metrics.items_processed += 1
            self.metrics.update_hit_rate()
            self.metrics.update_avg_access_time()

    def record_cache_miss(self, access_time: float = 0.0):
        """Record a cache miss."""
        with self._lock:
            self.metrics.cache_misses += 1
            self.metrics.total_item_access_time += access_time
            self.metrics.items_processed += 1
            self.metrics.update_hit_rate()
            self.metrics.update_avg_access_time()

    def record_cache_build(self, build_time: float, cache_size_mb: float):
        """Record cache build operation."""
        with self._lock:
            self.metrics.cache_builds += 1
            self.metrics.total_cache_build_time += build_time
            self.metrics.cache_size_mb = cache_size_mb

        if self.enable_detailed_logging:
            self.logger.info(f"Cache build completed: {build_time:.2f}s, {cache_size_mb:.1f}MB")

    def record_cache_load(self, load_time: float, cache_size_mb: float):
        """Record cache load operation."""
        with self._lock:
            self.metrics.cache_loads += 1
            self.metrics.total_cache_load_time += load_time
            self.metrics.cache_size_mb = cache_size_mb

        if self.enable_detailed_logging:
            self.logger.info(f"Cache load completed: {load_time:.2f}s, {cache_size_mb:.1f}MB")

    def record_cache_error(self, error_type: str, error_msg: str):
        """Record cache error."""
        with self._lock:
            self.metrics.cache_errors += 1

        self.logger.error(f"Cache error ({error_type}): {error_msg}")

    def record_batch_operation(self, batch_size: int, batch_time: float):
        """Record batch processing operation."""
        with self._lock:
            self.metrics.batch_operations += 1
            total_items = self.metrics.batch_operations * self.metrics.avg_batch_size + batch_size
            self.metrics.avg_batch_size = total_items / (self.metrics.batch_operations + 1) if self.metrics.batch_operations > 0 else batch_size
            self.metrics.total_batch_time += batch_time

    def take_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        try:
            memory_info = self.process.memory_info()
            system_memory = psutil.virtual_memory()

            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / (1024 * 1024),
                vms_mb=memory_info.vms / (1024 * 1024),
                percent=self.process.memory_percent(),
                available_mb=system_memory.available / (1024 * 1024)
            )

            with self._lock:
                self.memory_snapshots.append(snapshot)
                # Update peak memory usage
                if snapshot.rss_mb > self.metrics.peak_memory_usage:
                    self.metrics.peak_memory_usage = snapshot.rss_mb

            return snapshot

        except Exception as e:
            self.logger.warning(f"Failed to take memory snapshot: {e}")
            return MemorySnapshot(time.time(), 0, 0, 0, 0)

    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations."""
        start_time = time.time()
        start_snapshot = self.take_memory_snapshot()

        try:
            if self.enable_detailed_logging:
                self.logger.debug(f"Starting {operation_name}")
            yield
        finally:
            end_time = time.time()
            end_snapshot = self.take_memory_snapshot()
            duration = end_time - start_time

            memory_delta = end_snapshot.rss_mb - start_snapshot.rss_mb

            if self.enable_detailed_logging:
                self.logger.debug(f"Completed {operation_name}: {duration:.3f}s, "
                                f"memory delta: {memory_delta:+.1f}MB")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            # Calculate derived metrics
            total_accesses = self.metrics.cache_hits + self.metrics.cache_misses

            report = {
                "dataset_name": self.dataset_name,
                "timestamp": time.time(),

                # Cache performance
                "cache_metrics": {
                    "hits": self.metrics.cache_hits,
                    "misses": self.metrics.cache_misses,
                    "hit_rate": self.metrics.cache_hit_rate,
                    "total_accesses": total_accesses,
                    "builds": self.metrics.cache_builds,
                    "loads": self.metrics.cache_loads,
                    "errors": self.metrics.cache_errors
                },

                # Timing performance
                "timing_metrics": {
                    "avg_item_access_ms": self.metrics.avg_item_access_time_ms,
                    "total_cache_build_time": self.metrics.total_cache_build_time,
                    "total_cache_load_time": self.metrics.total_cache_load_time,
                    "avg_cache_build_time": (self.metrics.total_cache_build_time / self.metrics.cache_builds
                                           if self.metrics.cache_builds > 0 else 0),
                    "avg_cache_load_time": (self.metrics.total_cache_load_time / self.metrics.cache_loads
                                          if self.metrics.cache_loads > 0 else 0)
                },

                # Memory metrics
                "memory_metrics": {
                    "peak_usage_mb": self.metrics.peak_memory_usage,
                    "cache_size_mb": self.metrics.cache_size_mb,
                    "current_rss_mb": self.memory_snapshots[-1].rss_mb if self.memory_snapshots else 0,
                    "snapshots_count": len(self.memory_snapshots)
                },

                # Batch processing metrics
                "batch_metrics": {
                    "operations": self.metrics.batch_operations,
                    "avg_batch_size": self.metrics.avg_batch_size,
                    "total_batch_time": self.metrics.total_batch_time,
                    "avg_batch_time": (self.metrics.total_batch_time / self.metrics.batch_operations
                                     if self.metrics.batch_operations > 0 else 0)
                },

                # Performance indicators
                "performance_indicators": {
                    "cache_efficiency": "excellent" if self.metrics.cache_hit_rate > 0.95 else
                                      "good" if self.metrics.cache_hit_rate > 0.8 else
                                      "poor" if self.metrics.cache_hit_rate > 0.5 else "very_poor",
                    "memory_efficiency": "good" if self.metrics.peak_memory_usage < self.memory_limit_mb else "high",
                    "access_speed": "fast" if self.metrics.avg_item_access_time_ms < 1.0 else
                                  "moderate" if self.metrics.avg_item_access_time_ms < 5.0 else "slow"
                }
            }

        return report

    def print_performance_summary(self):
        """Print a human-readable performance summary."""
        report = self.get_performance_report()

        print(f"\n{'='*60}")
        print(f"Performance Summary: {self.dataset_name}")
        print(f"{'='*60}")

        # Cache performance
        cache_metrics = report["cache_metrics"]
        print(f"Cache Performance:")
        print(f"  Hits: {cache_metrics['hits']:,}")
        print(f"  Misses: {cache_metrics['misses']:,}")
        print(f"  Hit Rate: {cache_metrics['hit_rate']:.1%}")
        print(f"  Builds: {cache_metrics['builds']}")
        print(f"  Loads: {cache_metrics['loads']}")
        print(f"  Errors: {cache_metrics['errors']}")

        # Timing performance
        timing_metrics = report["timing_metrics"]
        print(f"\nTiming Performance:")
        print(f"  Avg Item Access: {timing_metrics['avg_item_access_ms']:.2f}ms")
        print(f"  Avg Cache Build: {timing_metrics['avg_cache_build_time']:.2f}s")
        print(f"  Avg Cache Load: {timing_metrics['avg_cache_load_time']:.2f}s")

        # Memory metrics
        memory_metrics = report["memory_metrics"]
        print(f"\nMemory Usage:")
        print(f"  Peak Usage: {memory_metrics['peak_usage_mb']:.1f}MB")
        print(f"  Cache Size: {memory_metrics['cache_size_mb']:.1f}MB")
        print(f"  Current RSS: {memory_metrics['current_rss_mb']:.1f}MB")

        # Performance indicators
        indicators = report["performance_indicators"]
        print(f"\nPerformance Indicators:")
        print(f"  Cache Efficiency: {indicators['cache_efficiency']}")
        print(f"  Memory Efficiency: {indicators['memory_efficiency']}")
        print(f"  Access Speed: {indicators['access_speed']}")

        print(f"{'='*60}\n")

    def optimize_batch_size(self, dataset_size: int, available_memory_mb: float) -> int:
        """
        Calculate optimal batch size based on dataset size and available memory.

        Args:
            dataset_size: Total number of items in dataset
            available_memory_mb: Available memory in MB

        Returns:
            Optimal batch size for processing
        """
        # Estimate memory per item (rough estimate for CIFAR-100 images)
        # 32x32x3 = 3072 bytes per image + overhead ≈ 4KB per item
        # 224x224x3 = 150KB per image + overhead ≈ 200KB per item
        estimated_item_size_mb = 0.2 if "224" in self.dataset_name else 0.004

        # Use 50% of available memory for batch processing
        usable_memory_mb = available_memory_mb * 0.5

        # Calculate batch size
        max_batch_size = int(usable_memory_mb / estimated_item_size_mb)

        # Apply reasonable limits
        min_batch_size = 100
        max_reasonable_batch_size = 10000

        optimal_batch_size = max(min_batch_size,
                               min(max_batch_size, max_reasonable_batch_size))

        # Ensure batch size doesn't exceed dataset size
        optimal_batch_size = min(optimal_batch_size, dataset_size)

        self.batch_size = optimal_batch_size

        if self.enable_detailed_logging:
            self.logger.info(f"Optimized batch size: {optimal_batch_size} "
                           f"(dataset: {dataset_size}, memory: {available_memory_mb:.1f}MB)")

        return optimal_batch_size

    def should_use_memory_mapping(self, cache_size_mb: float) -> bool:
        """
        Determine if memory mapping should be used based on cache size.

        Args:
            cache_size_mb: Cache size in MB

        Returns:
            True if memory mapping should be used
        """
        # Use memory mapping for caches larger than 1GB
        use_mmap = self.enable_memory_mapping and cache_size_mb > 1024

        if self.enable_detailed_logging and use_mmap:
            self.logger.info(f"Using memory mapping for large cache ({cache_size_mb:.1f}MB)")

        return use_mmap

    def get_memory_usage_trend(self, window_minutes: int = 5) -> Dict[str, float]:
        """
        Get memory usage trend over specified time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Dictionary with memory trend statistics
        """
        if not self.memory_snapshots:
            return {"trend": "no_data", "change_mb": 0.0, "rate_mb_per_min": 0.0}

        current_time = time.time()
        window_seconds = window_minutes * 60

        # Filter snapshots within time window
        recent_snapshots = [
            snapshot for snapshot in self.memory_snapshots
            if current_time - snapshot.timestamp <= window_seconds
        ]

        if len(recent_snapshots) < 2:
            return {"trend": "insufficient_data", "change_mb": 0.0, "rate_mb_per_min": 0.0}

        # Calculate trend
        first_snapshot = recent_snapshots[0]
        last_snapshot = recent_snapshots[-1]

        memory_change = last_snapshot.rss_mb - first_snapshot.rss_mb
        time_delta_minutes = (last_snapshot.timestamp - first_snapshot.timestamp) / 60

        if time_delta_minutes > 0:
            rate_mb_per_min = memory_change / time_delta_minutes
        else:
            rate_mb_per_min = 0.0

        # Determine trend direction
        if abs(memory_change) < 10:  # Less than 10MB change
            trend = "stable"
        elif memory_change > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        return {
            "trend": trend,
            "change_mb": memory_change,
            "rate_mb_per_min": rate_mb_per_min,
            "window_minutes": window_minutes,
            "snapshots_analyzed": len(recent_snapshots)
        }

    def reset_metrics(self):
        """Reset all performance metrics."""
        with self._lock:
            self.metrics = CacheMetrics()
            self.memory_snapshots.clear()

        self.logger.info("Performance metrics reset")


class OptimizedCacheLoader:
    """
    Optimized cache loader with minimal memory footprint and batch processing.
    """

    def __init__(self, performance_monitor: PerformanceMonitor):
        """
        Initialize optimized cache loader.

        Args:
            performance_monitor: Performance monitor instance
        """
        self.monitor = performance_monitor
        self.logger = logging.getLogger(f"{__name__}.loader")

    def load_cache_optimized(self, cache_path: str, use_memory_mapping: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load cache with memory optimization.

        Args:
            cache_path: Path to cache file
            use_memory_mapping: Whether to use memory mapping

        Returns:
            Cache data or None if loading failed
        """
        import pickle

        start_time = time.time()

        try:
            with self.monitor.monitor_operation(f"optimized_cache_load"):
                file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)

                if use_memory_mapping and file_size_mb > 100:
                    # Use memory mapping for large files
                    cache_data = self._load_with_memory_mapping(cache_path)
                else:
                    # Standard loading for smaller files
                    cache_data = self._load_standard(cache_path)

                load_time = time.time() - start_time
                self.monitor.record_cache_load(load_time, file_size_mb)

                return cache_data

        except Exception as e:
            self.monitor.record_cache_error("load_error", str(e))
            self.logger.error(f"Failed to load cache optimized: {e}")
            return None

    def _load_standard(self, cache_path: str) -> Dict[str, Any]:
        """Standard cache loading."""
        import pickle

        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    def _load_with_memory_mapping(self, cache_path: str) -> Dict[str, Any]:
        """Load cache using memory mapping for large files."""
        import pickle
        import mmap

        with open(cache_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                return pickle.load(mmapped_file)

    def process_cache_in_batches(self, cache_data: Dict[str, Any],
                               batch_processor_func, batch_size: Optional[int] = None) -> List[Any]:
        """
        Process cache data in batches to minimize memory usage.

        Args:
            cache_data: Cache data dictionary
            batch_processor_func: Function to process each batch
            batch_size: Batch size (uses monitor's optimized size if None)

        Returns:
            List of processed results
        """
        if batch_size is None:
            # Get available memory and optimize batch size
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            batch_size = self.monitor.optimize_batch_size(
                len(cache_data.get('processed_images', [])),
                available_memory
            )

        processed_images = cache_data.get('processed_images', [])
        targets = cache_data.get('targets', [])

        results = []
        total_items = len(processed_images)

        for i in range(0, total_items, batch_size):
            batch_start_time = time.time()

            end_idx = min(i + batch_size, total_items)
            batch_images = processed_images[i:end_idx]
            batch_targets = targets[i:end_idx] if targets else None

            with self.monitor.monitor_operation(f"batch_process_{i//batch_size}"):
                batch_result = batch_processor_func(batch_images, batch_targets)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])

            batch_time = time.time() - batch_start_time
            actual_batch_size = end_idx - i
            self.monitor.record_batch_operation(actual_batch_size, batch_time)

            # Log progress for large datasets
            if total_items > 10000 and (i // batch_size) % 10 == 0:
                progress = (end_idx / total_items) * 100
                self.logger.info(f"Batch processing progress: {progress:.1f}% "
                               f"({end_idx}/{total_items})")

        return results


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(dataset_name: str = "einstellung") -> PerformanceMonitor:
    """Get or create global performance monitor instance."""
    global _global_monitor

    if _global_monitor is None or _global_monitor.dataset_name != dataset_name:
        _global_monitor = PerformanceMonitor(dataset_name)

    return _global_monitor


def benchmark_cache_performance(dataset_class, root: str, num_samples: int = 1000) -> Dict[str, Any]:
    """
    Benchmark cache performance for a dataset class.

    Args:
        dataset_class: Dataset class to benchmark
        root: Dataset root directory
        num_samples: Number of samples to test

    Returns:
        Benchmark results dictionary
    """
    monitor = get_performance_monitor(dataset_class.__name__)

    print(f"🚀 Benchmarking cache performance for {dataset_class.__name__}")
    print(f"   Samples: {num_samples}")

    try:
        # Create dataset with caching enabled
        dataset = dataset_class(
            root=root,
            train=False,
            download=True,
            apply_shortcut=True,
            patch_size=4,
            enable_cache=True
        )

        # Warm up - access first few items
        warmup_samples = min(10, len(dataset))
        for i in range(warmup_samples):
            _ = dataset[i]

        # Benchmark access times
        start_time = time.time()

        for i in range(min(num_samples, len(dataset))):
            item_start = time.time()
            _ = dataset[i]
            item_time = time.time() - item_start

            # Record as cache hit (assuming cache is working)
            monitor.record_cache_hit(item_time)

            if i % 100 == 0:
                monitor.take_memory_snapshot()

        total_time = time.time() - start_time

        # Generate report
        report = monitor.get_performance_report()
        report["benchmark_info"] = {
            "dataset_class": dataset_class.__name__,
            "samples_tested": min(num_samples, len(dataset)),
            "total_benchmark_time": total_time,
            "avg_samples_per_second": min(num_samples, len(dataset)) / total_time if total_time > 0 else 0
        }

        print(f"   ✅ Benchmark completed in {total_time:.2f}s")
        print(f"   Average: {report['benchmark_info']['avg_samples_per_second']:.1f} samples/sec")

        return report

    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")
        monitor.record_cache_error("benchmark_error", str(e))
        return {"error": str(e), "success": False}


if __name__ == '__main__':
    # Test the performance monitoring system
    print("🔧 Testing Einstellung Performance Monitor")
    print("=" * 60)

    # Create test monitor
    monitor = PerformanceMonitor("test_dataset")

    # Simulate some operations
    monitor.record_cache_build(2.5, 150.0)
    monitor.record_cache_load(0.8, 150.0)

    for i in range(100):
        if i % 10 == 0:
            monitor.record_cache_miss(0.005)
        else:
            monitor.record_cache_hit(0.001)

        if i % 20 == 0:
            monitor.take_memory_snapshot()

    # Print performance summary
    monitor.print_performance_summary()

    print("Performance monitoring system ready for integration!")
