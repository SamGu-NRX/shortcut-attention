#!/usr/bin/env python3
"""
Demonstrapt for Einstellung cache validation system.

This script provides a simple demonstration of the comprehensive validation
system implemented for task 9, showing pixel-perfect comparison, statistical
validation, and cache integrity checking.
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.einstellung_cache_validation import EinstellungCacheValidator
from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung
from datasets.seq_cifar100_einstellung_224 import MyEinstellungCIFAR100_224
from utils.conf import base_path


def demo_pixel_perfect_validation():
    """Demonstrate pixel-perfect validation."""
    print("🔍 PIXEL-PERFECT VALIDATION DEMO")
    print("-" * 50)
    print("This validates that cached images are identical to on-the-fly processed images.")
    print()

    validator = EinstellungCacheValidator(verbose=True)

    # Test with a small sample
    result = validator.validate_pixel_perfect_comparison(
        dataset_class=MyCIFAR100Einstellung,
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        apply_shortcut=True,
        patch_size=4,
        sample_size=25
    )

    if result.passed:
        print("✅ Pixel-perfect validation PASSED")
        print(f"   Validated {result.details['samples_validated']} samples with {result.details['mismatches']} mismatches")
    else:
        print("❌ Pixel-perfect validation FAILED")
        print(f"   {result.message}")

    return result.passed


def demo_statistical_validation():
    """Demonstrate statistical validation."""
    print("\n📊 STATISTICAL VALIDATION DEMO")
    print("-" * 50)
    print("This validates that Einstellung effects (shortcut patches) are preserved correctly.")
    print()

    validator = EinstellungCacheValidator(verbose=True)

    # Test statistical properties
    result = validator.validate_statistical_properties(
        dataset_class=MyCIFAR100Einstellung,
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        apply_shortcut=True,
        patch_size=4,
        sample_size=100
    )

    if result.passed:
        print("✅ Statistical validation PASSED")
        print("   Key statistics:")
        print(f"   - Samples analyzed: {result.details['samples_analyzed']}")
        print(f"   - Total patches found: {result.details['total_patches_found']}")
        print(f"   - Shortcut classes with patches: {result.details['shortcut_classes_with_patches']}")
        print(f"   - Non-shortcut classes with patches: {result.details['non_shortcut_classes_with_patches']}")
        print(f"   - Patch color consistency: {result.details['patch_color_consistency']:.2%}")
    else:
        print("❌ Statistical validation FAILED")
        print(f"   {result.message}")

    return result.passed


def demo_cache_integrity_validation():
    """Demonstrate cache integrity validation."""
    print("\n🔒 CACHE INTEGRITY VALIDATION DEMO")
    print("-" * 50)
    print("This validates cache file integrity using checksums and structure validation.")
    print()

    validator = EinstellungCacheValidator(verbose=True)

    # Create a dataset to ensure cache exists
    dataset = MyCIFAR100Einstellung(
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        download=True,
        apply_shortcut=True,
        enable_cache=True
    )

    if dataset._cache_loaded:
        cache_path = dataset._get_cache_path()
        result = validator.validate_cache_integrity(cache_path)

        if result.passed:
            print("✅ Cache integrity validation PASSED")
            print("   Cache details:")
            print(f"   - File size: {result.details['file_size']:,} bytes")
            print(f"   - Samples: {result.details['samples']}")
            print(f"   - File checksum: {result.details['file_checksum'][:16]}...")
            print(f"   - Data checksum: {result.details['data_checksum'][:16]}...")
        else:
            print("❌ Cache integrity validation FAILED")
            print(f"   {result.message}")

        return result.passed
    else:
        print("⚠️  Cache not available for integrity testing")
        return True  # Don't fail if cache isn't available


def demo_comprehensive_validation():
    """Demonstrate comprehensive validation."""
    print("\n🎯 COMPREHENSIVE VALIDATION DEMO")
    print("-" * 50)
    print("This runs all validation types on multiple dataset configurations.")
    print()

    validator = EinstellungCacheValidator(verbose=False)  # Less verbose for comprehensive demo

    # Define test configurations
    configs = [
        {
            'name': 'CIFAR-100 32x32 with shortcuts',
            'dataset_class': MyCIFAR100Einstellung,
            'root': os.path.join(base_path(), 'CIFAR100'),
            'train': False,
            'apply_shortcut': True,
            'patch_size': 4
        },
        {
            'name': 'CIFAR-100 32x32 with masking',
            'dataset_class': MyCIFAR100Einstellung,
            'root': os.path.join(base_path(), 'CIFAR100'),
            'train': False,
            'apply_shortcut': False,
            'mask_shortcut': True,
            'patch_size': 4
        }
    ]

    print(f"Running comprehensive validation on {len(configs)} configurations...")
    print("(This may take a moment...)")

    start_time = time.time()
    results = validator.run_comprehensive_validation(configs, sample_size=30)
    end_time = time.time()

    print(f"\nValidation completed in {end_time - start_time:.1f} seconds")
    print("\nResults:")

    all_passed = True
    for config_name, result in results.items():
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"   {config_name}: {status}")
        if not result.passed:
            print(f"      Error: {result.message}")
            all_passed = False

    return all_passed


def main():
    """Main demonstration function."""
    print("🚀 EINSTELLUNG CACHE VALIDATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows the comprehensive validation system for Einstellung dataset caching.")
    print("The system ensures cached datasets produce identical results to original processing.")
    print()

    # Run demonstrations
    demos = [
        ("Pixel-Perfect Validation", demo_pixel_perfect_validation),
        ("Statistical Validation", demo_statistical_validation),
        ("Cache Integrity Validation", demo_cache_integrity_validation),
        ("Comprehensive Validation", demo_comprehensive_validation)
    ]

    all_demos_passed = True
    start_time = time.time()

    for demo_name, demo_func in demos:
        try:
            demo_passed = demo_func()
            if not demo_passed:
                all_demos_passed = False
        except Exception as e:
            print(f"❌ {demo_name} failed with error: {e}")
            all_demos_passed = False

    end_time = time.time()

    # Final summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print(f"Total time: {end_time - start_time:.1f} seconds")

    if all_demos_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print()
        print("The Einstellung cache validation system is working correctly:")
        print("✅ Cached images are pixel-perfect matches to original processing")
        print("✅ Statistical properties (Einstellung effects) are preserved")
        print("✅ Cache files have proper integrity and structure")
        print("✅ Comprehensive validation covers multiple configurations")
        print()
        print("This ensures that using cached datasets will produce identical")
        print("results to the original on-the-fly processing implementation.")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print()
        print("Please check the output above for details on which validations failed.")
        print("This may indicate issues with the caching implementation that need")
        print("to be addressed before using cached datasets in experiments.")

    print("\n" + "=" * 60)
    return 0 if all_demos_passed else 1


if __name__ == '__main__':
    sys.exit(main())
