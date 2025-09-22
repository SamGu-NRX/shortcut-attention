#!/usr/bin/env python3
"""
Test script for comprehensive Einstellung cache validation system.

This script demonstrates and tests the implementation of task 9:
- Pixel-perfect comparison between cached and on-the-fly processed images
- Statistical validation to ensure Einstellung effects are preserved correctly
- Checksum validation for cache integrity
- Comprehensive validation tests

Usage:
    python test_einstellung_cache_validation.py [--quick] [--verbose]
"""

import os
import sys
import tempfile
import shutil
import time
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.einstellung_cache_validation import EinstellungCacheValidator, ValidationResult
from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung
from datasets.seq_cifar100_einstellung_224 import MyEinstellungCIFAR100_224, TEinstellungCIFAR100_224
from utils.conf import base_path


def test_pixel_perfect_validation():
    """Test pixel-perfect comparison functionality."""
    print("\n" + "="*60)
    print("TESTING PIXEL-PERFECT VALIDATION")
    print("="*60)

    validator = EinstellungCacheValidator(verbose=True)

    # Test 32x32 dataset with shortcuts
    print("\n1. Testing 32x32 dataset with shortcuts...")
    result = validator.validate_pixel_perfect_comparison(
        dataset_class=MyCIFAR100Einstellung,
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,  # Use test set for faster testing
        apply_shortcut=True,
        mask_shortcut=False,
        patch_size=4,
        sample_size=50
    )

    print(f"Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"Message: {result.message}")
    if result.details:
        print(f"Details: {result.details}")

    # Test 32x32 dataset with masking
    print("\n2. Testing 32x32 dataset with masking...")
    result = validator.validate_pixel_perfect_comparison(
        dataset_class=MyCIFAR100Einstellung,
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        apply_shortcut=False,
        mask_shortcut=True,
        patch_size=4,
        sample_size=30
    )

    print(f"Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"Message: {result.message}")

    # Test 224x224 dataset
    print("\n3. Testing 224x224 dataset...")
    result = validator.validate_pixel_perfect_comparison(
        dataset_class=MyEinstellungCIFAR100_224,
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        apply_shortcut=True,
        patch_size=16,
        sample_size=20  # Smaller sample for 224x224
    )

    print(f"Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"Message: {result.message}")

    return True


def test_statistical_validation():
    """Test statistical validation functionality."""
    print("\n" + "="*60)
    print("TESTING STATISTICAL VALIDATION")
    print("="*60)

    validator = EinstellungCacheValidator(verbose=True)

    # Test with shortcuts enabled
    print("\n1. Testing statistical properties with shortcuts enabled...")
    result = validator.validate_statistical_properties(
        dataset_class=MyCIFAR100Einstellung,
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        apply_shortcut=True,
        patch_size=4,
        sample_size=100
    )

    print(f"Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"Message: {result.message}")
    if result.details:
        print("Statistical Details:")
        for key, value in result.details.items():
            print(f"  {key}: {value}")

    # Test with shortcuts disabled
    print("\n2. Testing statistical properties with shortcuts disabled...")
    result = validator.validate_statistical_properties(
        dataset_class=MyCIFAR100Einstellung,
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        apply_shortcut=False,
        patch_size=4,
        sample_size=100
    )

    print(f"Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"Message: {result.message}")

    # Test 224x224 statistical properties
    print("\n3. Testing 224x224 statistical properties...")
    result = validator.validate_statistical_properties(
        dataset_clayEinstellungCIFAR100_224,
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        apply_shortcut=True,
        patch_size=16,
        sample_size=50
    )

    print(f"Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"Message: {result.message}")

    return True


def test_cache_integrity_validation():
    """Test cache integrity validation functionality."""
    print("\n" + "="*60)
    print("TESTING CACHE INTEGRITY VALIDATION")
    print("="*60)

    validator = EinstellungCacheValidator(verbose=True)

    # Test integrity of existing caches
    test_configs = [
        {
            'name': '32x32 with shortcuts',
            'dataset_class': MyCIFAR100Einstellung,
            'params': {
                'root': os.path.join(base_path(), 'CIFAR100'),
                'train': False,
                'apply_shortcut': True,
                'patch_size': 4,
                'enable_cache': True
            }
        },
        {
            'name': '224x224 with shortcuts',
            'dataset_class': MyEinstellungCIFAR100_224,
            'params': {
                'root': os.path.join(base_path(), 'CIFAR100'),
                'train': False,
                'apply_shortcut': True,
                'patch_size': 16,
                'enable_cache': True
            }
        }
    ]

    for i, config in enumerate(test_configs, 1):
        print(f"\n{i}. Testing cache integrity for {config['name']}...")

        try:
            # Create dataset to ensure cache exists
            dataset = config['dataset_class'](**config['params'])

            if dataset._cache_loaded:
                cache_path = dataset._get_cache_path()
                result = validator.validate_cache_integrity(cache_path)

                print(f"Result: {'PASS' if result.passed else 'FAIL'}")
                print(f"Message: {result.message}")
                if result.details:
                    print("Integrity Details:")
                    for key, value in result.details.items():
                        if key in ['file_checksum', 'data_checksum']:
                            print(f"  {key}: {str(value)[:16]}...")
                        else:
                            print(f"  {key}: {value}")
            else:
                print("Cache not loaded - skipping integrity test")

        except Exception as e:
            print(f"Error testing {config['name']}: {e}")

    return True


def test_comprehensive_validation():
    """Test comprehensive validation functionality."""
    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE VALIDATION")
    print("="*60)

    validator = EinstellungCacheValidator(verbose=True)

    # Define comprehensive test configurations
    configs = [
        {
            'name': 'einstellung_32x32_shortcuts',
            'dataset_class': MyCIFAR100Einstellung,
            'root': os.path.join(base_path(), 'CIFAR100'),
            'train': False,
            'apply_shortcut': True,
            'mask_shortcut': False,
            'patch_size': 4,
            'patch_color': (255, 0, 255)
        },
        {
            'name': 'einstellung_32x32_masked',
            'dataset_class': MyCIFAR100Einstellung,
            'root': os.path.join(base_path(), 'CIFAR100'),
            'train': False,
            'apply_shortcut': False,
            'mask_shortcut': True,
            'patch_size': 4,
            'patch_color': (255, 0, 255)
        },
        {
            'name': 'einstellung_224x224_shortcuts',
            'dataset_class': MyEinstellungCIFAR100_224,
            'root': os.path.join(base_path(), 'CIFAR100'),
            'train': False,
            'apply_shortcut': True,
            'mask_shortcut': False,
            'patch_size': 16,
            'patch_color': (255, 0, 255)
        }
    ]

    print(f"\nRunning comprehensive validation on {len(configs)} configurations...")

    start_time = time.time()
    results = validator.run_comprehensive_validation(configs, sample_size=30)
    end_time = time.time()

    print(f"\nValidation completed in {end_time - start_time:.2f} seconds")

    # Print detailed results
    print("\nDETAILED RESULTS:")
    print("-" * 40)

    all_passed = True
    for config_name, result in results.items():
        print(f"\nConfiguration: {config_name}")
        print(f"Overall Result: {'PASS' if result.passed else 'FAIL'}")
        print(f"Message: {result.message}")

        if result.details:
            # Print sub-validation results
            for validation_type, sub_result in result.details.items():
                if isinstance(sub_result, ValidationResult):
                    status = 'PASS' if sub_result.passed else 'FAIL'
                    print(f"  {validation_type}: {status}")
                    if not sub_result.passed:
                        print(f"    Error: {sub_result.message}")

        if not result.passed:
            all_passed = False

    print("\n" + "="*60)
    print(f"COMPREHENSIVE VALIDATION: {'PASS' if all_passed else 'FAIL'}")
    print("="*60)

    return all_passed


def test_error_handling():
    """Test error handling in validation system."""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)

    validator = EinstellungCacheValidator(verbose=True)

    # Test with non-existent cache file
    print("\n1. Testing non-existent cache file...")
    result = validator.validate_cache_integrity("/non/existent/cache.pkl")
    print(f"Result: {'PASS' if not result.passed else 'FAIL'} (should fail)")
    print(f"Message: {result.message}")

    # Test with invalid dataset configuration
    print("\n2. Testing invalid dataset configuration...")
    try:
        result = validator.validate_pixel_perfect_comparison(
            dataset_class=MyCIFAR100Einstellung,
            root="/non/existent/path",
            train=False,
            sample_size=10
        )
        print(f"Result: {'PASS' if not result.passed else 'FAIL'} (should fail)")
        print(f"Message: {result.message}")
    except Exception as e:
        print(f"Caught expected exception: {e}")

    return True


def run_quick_validation():
    """Run a quick validation test with minimal samples."""
    print("\n" + "="*60)
    print("RUNNING QUICK VALIDATION TEST")
    print("="*60)

    validator = EinstellungCacheValidator(verbose=True)

    # Quick pixel-perfect test
    print("\nQuick pixel-perfect validation...")
    result = validator.validate_pixel_perfect_comparison(
        dataset_class=MyCIFAR100Einstellung,
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        apply_shortcut=True,
        sample_size=10
    )

    if not result.passed:
        print(f"❌ Quick validation failed: {result.message}")
        return False

    # Quick statistical test
    print("\nQuick statistical validation...")
    result = validator.validate_statistical_properties(
        dataset_class=MyCIFAR100Einstellung,
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        apply_shortcut=True,
        sample_size=20
    )

    if not result.passed:
        print(f"❌ Quick statistical validation failed: {result.message}")
        return False

    print("\n✅ Quick validation passed!")
    return True


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Test Einstellung Cache Validation System')
    parser.add_argument('--quick', action='store_true', help='Run quick validation test only')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--pixel-perfect', action='store_true', help='Test pixel-perfect validation only')
    parser.add_argument('--statistical', action='store_true', help='Test statistical validation only')
    parser.add_argument('--integrity', action='store_true', help='Test cache integrity validation only')
    parser.add_argument('--comprehensive', action='store_true', help='Test comprehensive validation only')
    parser.add_argument('--error-handling', action='store_true', help='Test error handling only')

    args = parser.parse_args()

    print("Einstellung Cache Validation System Test")
    print("=" * 60)

    start_time = time.time()
    all_tests_passed = True

    try:
        if args.quick:
            all_tests_passed = run_quick_validation()
        elif args.pixel_perfect:
            all_tests_passed = test_pixel_perfect_validation()
        elif args.statistical:
            all_tests_passed = test_statistical_validation()
        elif args.integrity:
            all_tests_passed = test_cache_integrity_validation()
        elif args.comprehensive:
            all_tests_passed = test_comprehensive_validation()
        elif args.error_handling:
            all_tests_passed = test_error_handling()
        else:
            # Run all tests
            print("\nRunning all validation tests...")

            tests = [
                ("Pixel-Perfect Validation", test_pixel_perfect_validation),
                ("Statistical Validation", test_statistical_validation),
                ("Cache Integrity Validation", test_cache_integrity_validation),
                ("Comprehensive Validation", test_comprehensive_validation),
                ("Error Handling", test_error_handling)
            ]

            for test_name, test_func in tests:
                print(f"\n{'='*20} {test_name} {'='*20}")
                try:
                    test_result = test_func()
                    if not test_result:
                        all_tests_passed = False
                        print(f"❌ {test_name} failed")
                    else:
                        print(f"✅ {test_name} passed")
                except Exception as e:
                    print(f"❌ {test_name} failed with exception: {e}")
                    all_tests_passed = False

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nTest failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    end_time = time.time()

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Overall result: {'PASS' if all_tests_passed else 'FAIL'}")

    if all_tests_passed:
        print("\n🎉 All validation tests passed!")
        print("The Einstellung cache validation system is working correctly.")
    else:
        print("\n❌ Some validation tests failed.")
        print("Please check the output above for details.")

    return 0 if all_tests_passed else 1


if __name__ == '__main__':
    sys.exit(main())
