#!/usr/bin/env python3
"""
Cross-Method Cache Consistency Validation for Einstellung Experiments

This script validates that cached datasets provide identical data across different
continual learning methods in comparative experiments. This is critical for ensuring
fair comparisons in run_einstellung_experiment.py.

The test ensures that:
1. Different methods (SGD, EWC, DER++, etc.) get identical cached data
2. Cache keys are consistent across method invocations
3. Dataset iteration order is deterministic
4. Einstellung effects are applied consistently

Usage:
    python test_cross_method_cachistency.py [--verbose] [--sample-size N]
"""

import os
import sys
import tempfile
import shutil
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung
from datasets.seq_cifar100_einstellung_224 import MyEinstellungCIFAR100_224
from utils.conf import base_path
from utils.einstellung_cache_validation import EinstellungCacheValidator, ValidationResult


class CrossMethodCacheConsistencyValidator:
    """
    Validates cache consistency across different continual learning methods.

    This ensures that comparative experiments in run_einstellung_experiment.py
    use identical cached data for fair method comparison.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.base_validator = EinstellungCacheValidator(verbose=False)

    def validate_cross_method_consistency(self,
                                        dataset_class,
                                        root: str,
                                        train: bool = False,
                                        apply_shortcut: bool = True,
                                        patch_size: int = 4,
                                        sample_size: int = 100,
                                        methods: List[str] = None) -> ValidationResult:
        """
        Validate that different methods get identical cached data.

        This simulates how different continual learning methods would access
        the same cached dataset in a comparative experiment.

        Args:
            dataset_class: Dataset class to test
            root: Dataset root directory
            train: Whether to use training set
            apply_shortcut: Whether to apply shortcuts
            patch_size: Size of patches
            sample_size: Number of samples to validate
            methods: List of method names to simulate (for logging)

        Returns:
            ValidationResult indicating consistency across methods
        """
        if self.verbose:
            print("🔄 Testing cross-method cache consistency...")
            print(f"   Dataset: {dataset_class.__name__}")
            print(f"   Sample size: {sample_size}")
            print(f"   Methods simulated: {methods or ['Method1', 'Method2', 'Method3']}")

        methods = methods or ['SGD', 'EWC', 'DER++']

        try:
            # Create multiple dataset instances as different methods would
            datasets = {}
            cache_keys = {}

            for method in methods:
                if self.verbose:
                    print(f"   Creating dataset instance for {method}...")

                # Each method creates its own dataset instance
                dataset = dataset_class(
                    root=root,
                    train=train,
                    download=True,
                    apply_shortcut=apply_shortcut,
                    mask_shortcut=False,
                    patch_size=patch_size,
                    enable_cache=True
                )

                # Ensure cache is loaded
                if not dataset._cache_loaded:
                    return ValidationResult(
                        passed=False,
                        message=f"Cache not loaded for {method}",
                        details={"failed_method": method}
                    )

                datasets[method] = dataset
                cache_keys[method] = dataset._get_cache_key()

            # Validate cache keys are identical
            unique_cache_keys = set(cache_keys.values())
            if len(unique_cache_keys) != 1:
                return ValidationResult(
                    passed=False,
                    message=f"Cache keys differ across methods: {cache_keys}",
                    details={"cache_keys": cache_keys}
                )

            if self.verbose:
                print(f"   ✓ All methods use same cache key: {list(unique_cache_keys)[0][:16]}...")

            # Compare data across methods
            inconsistencies = []
            method_names = list(methods)
            reference_method = method_names[0]
            reference_dataset = datasets[reference_method]

            total_samples = min(sample_size, len(reference_dataset))

            for i in range(total_samples):
                # Get reference data
                ref_item = reference_dataset[i]
                ref_img_hash = self._hash_tensor_or_array(ref_item[0])
                ref_target = ref_item[1]

                # Compare with other methods
                for method in method_names[1:]:
                    method_dataset = datasets[method]
                    method_item = method_dataset[i]
                    method_img_hash = self._hash_tensor_or_array(method_item[0])
                    method_target = method_item[1]

                    # Check image consistency
                    if ref_img_hash != method_img_hash:
                        inconsistencies.append({
                            "index": i,
                            "type": "image_hash_mismatch",
                            "reference_method": reference_method,
                            "comparison_method": method,
                            "ref_hash": ref_img_hash[:16],
                            "method_hash": method_img_hash[:16]
                        })

                    # Check target consistency
                    if ref_target != method_target:
                        inconsistencies.append({
                            "index": i,
                            "type": "target_mismatch",
                            "reference_method": reference_method,
                            "comparison_method": method,
                            "ref_target": ref_target,
                            "method_target": method_target
                        })

                # Progress reporting
                if self.verbose and (i + 1) % 50 == 0:
                    print(f"   Validated {i + 1}/{total_samples} samples...")

            # Analyze results
            if len(inconsistencies) == 0:
                if self.verbose:
                    print(f"   ✓ Cross-method consistency validated for {total_samples} samples")

                return ValidationResult(
                    passed=True,
                    message=f"All {len(methods)} methods access identical cached data",
                    details={
                        "methods_tested": methods,
                        "samples_validated": total_samples,
                        "inconsistencies": 0,
                        "cache_key": list(unique_cache_keys)[0]
                    }
                )
            else:
                if self.verbose:
                    print(f"   ✗ Found {len(inconsistencies)} inconsistencies")
                    for inc in inconsistencies[:3]:  # Show first 3
                        print(f"     {inc}")

                return ValidationResult(
                    passed=False,
                    message=f"Found {len(inconsistencies)} inconsistencies across methods",
                    details={
                        "methods_tested": methods,
                        "samples_validated": total_samples,
                        "inconsistencies": len(inconsistencies),
                        "inconsistency_details": inconsistencies[:10]
                    }
                )

        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Cross-method validation failed: {str(e)}",
                details={"error_type": type(e).__name__}
            )

    def validate_deterministic_iteration_order(self,
                                             dataset_class,
                                             root: str,
                                             train: bool = False,
                                             apply_shortcut: bool = True,
                                             patch_size: int = 4,
                                             iterations: int = 3,
                                             sample_size: int = 50) -> ValidationResult:
        """
        Validate that dataset iteration order is deterministic across multiple runs.

        This ensures that different method runs will see data in the same order,
        which is crucial for fair comparison in experiments.

        Args:
            dataset_class: Dataset class to test
            root: Dataset root directory
            train: Whether to use training set
            apply_shortcut: Whether to apply shortcuts
            patch_size: Size of patches
            iterations: Number of iteration runs to compare
            sample_size: Number of samples to check per iteration

        Returns:
            ValidationResult indicating iteration determinism
        """
        if self.verbose:
            print("🔁 Testing deterministic iteration order...")
            print(f"   Iterations: {iterations}")
            print(f"   Sample size per iteration: {sample_size}")

        try:
            iteration_hashes = []

            for iteration in range(iterations):
                if self.verbose:
                    print(f"   Running iteration {iteration + 1}/{iterations}...")

                # Create fresh dataset instance
                dataset = dataset_class(
                    root=root,
                    train=train,
                    download=True,
                    apply_shortcut=apply_shortcut,
                    patch_size=patch_size,
                    enable_cache=True
                )

                # Collect hashes for this iteration
                iteration_data_hashes = []
                total_samples = min(sample_size, len(dataset))

                for i in range(total_samples):
                    item = dataset[i]
                    img_hash = self._hash_tensor_or_array(item[0])
                    target = item[1]

                    # Create combined hash for this sample
                    sample_hash = hashlib.sha256(f"{img_hash}_{target}".encode()).hexdigest()
                    iteration_data_hashes.append(sample_hash)

                # Create hash for entire iteration
                iteration_hash = hashlib.sha256("".join(iteration_data_hashes).encode()).hexdigest()
                iteration_hashes.append(iteration_hash)

            # Check if all iterations produced the same hash
            unique_hashes = set(iteration_hashes)

            if len(unique_hashes) == 1:
                if self.verbose:
                    print(f"   ✓ Iteration order is deterministic across {iterations} runs")

                return ValidationResult(
                    passed=True,
                    message=f"Iteration order is deterministic across {iterations} runs",
                    details={
                        "iterations_tested": iterations,
                        "samples_per_iteration": total_samples,
                        "unique_hashes": 1,
                        "iteration_hash": list(unique_hashes)[0][:16]
                    }
                )
            else:
                if self.verbose:
                    print(f"   ✗ Iteration order varies: {len(unique_hashes)} different patterns")

                return ValidationResult(
                    passed=False,
                    message=f"Iteration order not deterministic: {len(unique_hashes)} different patterns",
                    details={
                        "iterations_tested": iterations,
                        "samples_per_iteration": total_samples,
                        "unique_hashes": len(unique_hashes),
                        "iteration_hashes": [h[:16] for h in iteration_hashes]
                    }
                )

        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Iteration determinism validation failed: {str(e)}",
                details={"error_type": type(e).__name__}
            )

    def validate_einstellung_effect_consistency(self,
                                              dataset_class,
                                              root: str,
                                              train: bool = False,
                                              patch_size: int = 4,
                                              sample_size: int = 100) -> ValidationResult:
        """
        Validate that Einstellung effects are applied consistently across different
        dataset configurations that would be used by different methods.

        Args:
            dataset_class: Dataset class to test
            root: Dataset root directory
            train: Whether to use training set
            patch_size: Size of patches
            sample_size: Number of samples to analyze

        Returns:
            ValidationResult indicating Einstellung effect consistency
        """
        if self.verbose:
            print("🎯 Testing Einstellung effect consistency...")

        try:
            # Test different configurations that methods might use
            configs = [
                {"apply_shortcut": True, "mask_shortcut": False, "name": "shortcuts_enabled"},
                {"apply_shortcut": False, "mask_shortcut": True, "name": "shortcuts_masked"},
                {"apply_shortcut": False, "mask_shortcut": False, "name": "no_shortcuts"}
            ]

            config_results = {}

            for config in configs:
                if self.verbose:
                    print(f"   Testing configuration: {config['name']}")

                # Run statistical validation for this config
                stats_result = self.base_validator.validate_statistical_properties(
                    dataset_class=dataset_class,
                    root=root,
                    train=train,
                    apply_shortcut=config["apply_shortcut"],
                    patch_size=patch_size,
                    sample_size=sample_size
                )

                config_results[config['name']] = stats_result

            # Analyze consistency
            failed_configs = [name for name, result in config_results.items() if not result.passed]

            if len(failed_configs) == 0:
                if self.verbose:
                    print("   ✓ Einstellung effects consistent across all configurations")

                return ValidationResult(
                    passed=True,
                    message="Einstellung effects consistent across all configurations",
                    details={
                        "configurations_tested": len(configs),
                        "failed_configurations": 0,
                        "config_results": {name: result.details for name, result in config_results.items()}
                    }
                )
            else:
                if self.verbose:
                    print(f"   ✗ Einstellung effects inconsistent in: {failed_configs}")

                return ValidationResult(
                    passed=False,
                    message=f"Einstellung effects inconsistent in {len(failed_configs)} configurations",
                    details={
                        "configurations_tested": len(configs),
                        "failed_configurations": len(failed_configs),
                        "failed_config_names": failed_configs,
                        "config_results": {name: result.details for name, result in config_results.items()}
                    }
                )

        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Einstellung effect consistency validation failed: {str(e)}",
                details={"error_type": type(e).__name__}
            )

    def run_comprehensive_cross_method_validation(self,
                                                dataset_configs: List[Dict[str, Any]],
                                                methods: List[str] = None,
                                                sample_size: int = 100) -> Dict[str, ValidationResult]:
        """
        Run comprehensive cross-method validation on multiple dataset configurations.

        Args:
            dataset_configs: List of dataset configuration dictionaries
            methods: List of method names to simulate
            sample_size: Number of samples to validate per test

        Returns:
            Dictionary mapping test names to validation results
        """
        if self.verbose:
            print("🔬 Running comprehensive cross-method validation...")
            print(f"   Dataset configs: {len(dataset_configs)}")
            print(f"   Methods: {methods or ['SGD', 'EWC', 'DER++']}")
            print(f"   Sample size: {sample_size}")

        methods = methods or ['SGD', 'EWC', 'DER++']
        results = {}

        for i, config in enumerate(dataset_configs):
            config_name = config.get('name', f'config_{i}')
            if self.verbose:
                print(f"\n   Testing configuration: {config_name}")

            try:
                # Cross-method consistency test
                cross_method_result = self.validate_cross_method_consistency(
                    dataset_class=config['dataset_class'],
                    root=config['root'],
                    train=config.get('train', False),
                    apply_shortcut=config.get('apply_shortcut', True),
                    patch_size=config.get('patch_size', 4),
                    sample_size=sample_size,
                    methods=methods
                )
                results[f"{config_name}_cross_method"] = cross_method_result

                # Deterministic iteration test
                iteration_result = self.validate_deterministic_iteration_order(
                    dataset_class=config['dataset_class'],
                    root=config['root'],
                    train=config.get('train', False),
                    apply_shortcut=config.get('apply_shortcut', True),
                    patch_size=config.get('patch_size', 4),
                    iterations=3,
                    sample_size=min(50, sample_size)
                )
                results[f"{config_name}_iteration_determinism"] = iteration_result

                # Einstellung effect consistency test
                einstellung_result = self.validate_einstellung_effect_consistency(
                    dataset_class=config['dataset_class'],
                    root=config['root'],
                    train=config.get('train', False),
                    patch_size=config.get('patch_size', 4),
                    sample_size=sample_size
                )
                results[f"{config_name}_einstellung_consistency"] = einstellung_result

            except Exception as e:
                error_result = ValidationResult(
                    passed=False,
                    message=f"Validation failed for {config_name}: {str(e)}",
                    details={"error_type": type(e).__name__}
                )
                results[f"{config_name}_error"] = error_result

        return results

    def _hash_tensor_or_array(self, data) -> str:
        """Create hash of tensor or array data for comparison."""
        if hasattr(data, 'numpy'):  # PyTorch tensor
            data_np = data.numpy()
        elif hasattr(data, '__array__'):  # NumPy array or similar
            data_np = np.array(data)
        else:
            # Convert to string and hash
            return hashlib.sha256(str(data).encode()).hexdigest()

        # Create hash from numpy array bytes
        return hashlib.sha256(data_np.tobytes()).hexdigest()


def main():
    """Main function for running cross-method validation."""
    import argparse

    parser = argparse.ArgumentParser(description='Cross-Method Cache Consistency Validation')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of samples to validate')
    parser.add_argument('--methods', nargs='+', default=['SGD', 'EWC', 'DER++'],
                       help='Methods to simulate')
    parser.add_argument('--quick', action='store_true', help='Run quick validation with smaller samples')

    args = parser.parse_args()

    if args.quick:
        args.sample_size = 20

    print("🔄 Cross-Method Cache Consistency Validation")
    print("=" * 60)
    print("This validates that cached datasets provide identical data")
    print("across different continual learning methods in comparative experiments.")
    print()

    validator = CrossMethodCacheConsistencyValidator(verbose=args.verbose)

    # Define test configurations
    configs = [
        {
            'name': 'einstellung_32x32_shortcuts',
            'dataset_class': MyCIFAR100Einstellung,
            'root': os.path.join(base_path(), 'CIFAR100'),
            'train': False,
            'apply_shortcut': True,
            'patch_size': 4
        },
        {
            'name': 'einstellung_224x224_shortcuts',
            'dataset_class': MyEinstellungCIFAR100_224,
            'root': os.path.join(base_path(), 'CIFAR100'),
            'train': False,
            'apply_shortcut': True,
            'patch_size': 16
        }
    ]

    print(f"Running validation with {len(configs)} configurations...")
    print(f"Methods to simulate: {args.methods}")
    print(f"Sample size: {args.sample_size}")
    print()

    # Run comprehensive validation
    results = validator.run_comprehensive_cross_method_validation(
        dataset_configs=configs,
        methods=args.methods,
        sample_size=args.sample_size
    )

    # Print results summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result.passed:
            print(f"  Error: {result.message}")
            all_passed = False

    print("\n" + "=" * 60)
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    print(f"OVERALL RESULT: {overall_status}")

    if all_passed:
        print("\n🎉 Cross-method cache consistency validated!")
        print("Different continual learning methods will receive identical cached data")
        print("in comparative experiments, ensuring fair method comparison.")
    else:
        print("\n⚠️  Cross-method consistency issues detected!")
        print("This may lead to unfair comparisons in run_einstellung_experiment.py")
        print("Please review the failed tests above.")

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
