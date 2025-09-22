#!/usr/bin/env python3
"""
Test script to verify that the deterministic transform fix works correctly.

This script tests that different dataset instances (simulating different methods)
produce identical results after the deterministic transform fixes.
"""

import os
import sys
import hashlib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung
from utils.conf import base_path


def test_deterministic_consistency():
    """Test that multiple dataset instances produce identical results."""
    print("🔧 Testing Deterministic Transform Fix")
    print("=" * 50)

    # Create multiple dataset instances (simulating different methods)
    datasets = []
    method_names = ['SGD', 'EWC', 'DER++']

    for method in method_names:
        print(f"Creating dataset instance for {method}...")
        dataset = MyCIFAR100Einstellung(
            root=os.path.join(base_path(), 'CIFAR100'),
            train=False,
            download=True,
            apply_shortcut=True,
            patch_size=4,
            enable_cache=False  # Disable cache to test raw processing
        )
        datasets.append(dataset)

    # Test consistency across instances
    print(f"\nTesting consistency across {len(datasets)} instances...")

    sample_size = 20
    inconsistencies = []

    reference_dataset = datasets[0]

    for i in range(min(sample_size, len(reference_dataset))):
        # Get reference data
        ref_item = reference_dataset[i]
        ref_img_hash = hashlib.sha256(np.array(ref_item[0]).tobytes()).hexdigest()
        ref_target = ref_item[1]

        # Compare with other instances
        for j, dataset in enumerate(datasets[1:], 1):
            item = dataset[i]
            img_hash = hashlib.sha256(np.array(item[0]).tobytes()).hexdigest()
            target = item[1]

            if ref_img_hash != img_hash:
                inconsistencies.append({
                    "index": i,
                    "type": "image_mismatch",
                    "reference_method": method_names[0],
                    "comparison_method": method_names[j],
                    "ref_hash": ref_img_hash[:16],
                    "comp_hash": img_hash[:16]
                })

            if ref_target != target:
                inconsistencies.append({
                    "index": i,
                    "type": "target_mismatch",
                    "reference_method": method_names[0],
                    "comparison_method": method_names[j],
                    "ref_target": ref_target,
                    "comp_target": target
                })

    # Report results
    if len(inconsistencies) == 0:
        print(f"✅ SUCCESS: All {len(datasets)} instances produce identical results!")
        print(f"   Tested {sample_size} samples with 0 inconsistencies")
        print("   The deterministic transform fix is working correctly.")
        return True
    else:
        print(f"❌ FAILURE: Found {len(inconsistencies)} inconsistencies")
        print("   First few inconsistencies:")
        for inc in inconsistencies[:3]:
            print(f"     {inc}")
        return False


def test_patch_determinism():
    """Test that patch placement is deterministic."""
    print("\n🎯 Testing Patch Placement Determinism")
    print("=" * 50)

    # Create dataset with shortcuts
    dataset = MyCIFAR100Einstellung(
        root=os.path.join(base_path(), 'CIFAR100'),
        train=False,
        download=True,
        apply_shortcut=True,
        patch_size=4,
        enable_cache=False
    )

    # Test same item multiple times
    test_indices = [0, 1, 2, 5, 10]

    for idx in test_indices:
        # Get item multiple times
        item1 = dataset[idx]
        item2 = dataset[idx]

        # Compare hashes
        hash1 = hashlib.sha256(np.array(item1[0]).tobytes()).hexdigest()
        hash2 = hashlib.sha256(np.array(item2[0]).tobytes()).hexdigest()

        if hash1 != hash2:
            print(f"❌ Index {idx}: Patch placement not deterministic")
            print(f"   Hash 1: {hash1[:16]}...")
            print(f"   Hash 2: {hash2[:16]}...")
            return False

    print(f"✅ Patch placement is deterministic for all {len(test_indices)} test indices")
    return True


def main():
    """Main test function."""
    print("🧪 Deterministic Transform Fix Validation")
    print("=" * 60)

    success = True

    try:
        # Test cross-instance consistency
        if not test_deterministic_consistency():
            success = False

        # Test patch determinism
        if not test_patch_determinism():
            success = False

        print("\n" + "=" * 60)
        if success:
            print("🎉 ALL TESTS PASSED!")
            print("The deterministic transform fix is working correctly.")
            print("Different continual learning methods will now get identical data.")
        else:
            print("❌ SOME TESTS FAILED!")
            print("The deterministic transform fix needs more work.")

        return 0 if success else 1

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
