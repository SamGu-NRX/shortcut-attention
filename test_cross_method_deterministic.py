#!/usr/bin/env python3
"""
Updated cross-method consistency test for the deterministic approach.

This test validates that different continual learning methods get identical data
after the deterministic transform fixes, without relying on complex caching.
"""

import os
import sys
import hashlib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung
from datasets.seq_cifar100_einstellung_224 import MyEinstellungCIFAR100_224
from utils.conf import base_path


def test_cross_method_consistency(dataset_class, dataset_name, sample_size=50):
    """Test cross-method consistency for a dataset class."""
    print(f"\n🔄 Testing {dataset_name}")
    print("-" * 50)

    methods = ['SGD', 'EWC', 'DER++']

    # Create dataset instances for each method
    datasets = []
    for method in methods:
        if dataset_class == MyEinstellungCIFAR100_224:
            dataset = dataset_class(
                root=os.path.join(base_path(), 'CIFAR100'),
                train=False,
                download=True,
                apply_shortcut=True,
                patch_size=16,
                enable_cache=False  # Use deterministic processing without cache
            )
        else:
            dataset = dataset_class(
                root=os.path.join(base_path(), 'CIFAR100'),
                train=False,
                download=True,
                apply_shortcut=True,
                patch_size=4,
                enable_cache=False  # Use deterministic processing without cache
            )
        datasets.append(dataset)

    print(f"Created {len(datasets)} dataset instances")

    # Test consistency
    inconsistencies = []
    reference_dataset = datasets[0]

    test_size = min(sample_size, len(reference_dataset))

    for i in range(test_size):
        ref_item = reference_dataset[i]
        ref_img_hash = hashlib.sha256(np.array(ref_item[0]).tobytes()).hexdigest()
        ref_target = ref_item[1]

        for j, dataset in enumerate(datasets[1:], 1):
            item = dataset[i]
            img_hash = hashlib.sha256(np.array(item[0]).tobytes()).hexdigest()
            target = item[1]

            if ref_img_hash != img_hash:
                inconsistencies.append({
                    "index": i,
                    "type": "image_mismatch",
                    "reference_method": methods[0],
                    "comparison_method": methods[j],
                    "ref_hash": ref_img_hash[:16],
                    "comp_hash": img_hash[:16]
                })

            if ref_target != target:
                inconsistencies.append({
                    "index": i,
                    "type": "target_mismatch",
                    "reference_method": methods[0],
                    "comparison_method": methods[j],
                    "ref_target": ref_target,
                    "comp_target": target
                })

    # Report results
    if len(inconsistencies) == 0:
        print(f"✅ SUCCESS: All {len(datasets)} instances produce identical results!")
        print(f"   Tested {test_size} samples with 0 inconsistencies")
        return True
    else:
        print(f"❌ FAILURE: Found {len(inconsistencies)} inconsistencies")
        print("   First few inconsistencies:")
        for inc in inconsistencies[:3]:
            print(f"     {inc}")
        return False


def test_iteration_determinism(dataset_class, dataset_name, iterations=3, sample_size=20):
    """Test that dataset iteration is deterministic across multiple runs."""
    print(f"\n🔁 Testing Iteration Determinism for {dataset_name}")
    print("-" * 50)

    iteration_hashes = []

    for iteration in range(iterations):
        print(f"Running iteration {iteration + 1}/{iterations}...")

        if dataset_class == MyEinstellungCIFAR100_224:
            dataset = dataset_class(
                root=os.path.join(base_path(), 'CIFAR100'),
                train=False,
                download=True,
                apply_shortcut=True,
                patch_size=16,
                enable_cache=False
            )
        else:
            dataset = dataset_class(
                root=os.path.join(base_path(), 'CIFAR100'),
                train=False,
                download=True,
                apply_shortcut=True,
                patch_size=4,
                enable_cache=False
            )

        # Collect hashes for this iteration
        iteration_data_hashes = []
        test_size = min(sample_size, len(dataset))

        for i in range(test_size):
            item = dataset[i]
            img_hash = hashlib.sha256(np.array(item[0]).tobytes()).hexdigest()
            target = item[1]

            sample_hash = hashlib.sha256(f"{img_hash}_{target}".encode()).hexdigest()
            iteration_data_hashes.append(sample_hash)

        # Create hash for entire iteration
        iteration_hash = hashlib.sha256("".join(iteration_data_hashes).encode()).hexdigest()
        iteration_hashes.append(iteration_hash)

    # Check if all iterations produced the same hash
    unique_hashes = set(iteration_hashes)

    if len(unique_hashes) == 1:
        print(f"✅ Iteration order is deterministic across {iterations} runs")
        return True
    else:
        print(f"❌ Iteration order varies: {len(unique_hashes)} different patterns")
        return False


def main():
    """Main test function."""
    print("🧪 Cross-Method Deterministic Consistency Test")
    print("=" * 60)
    print("Testing the deterministic transform approach without complex caching")

    test_configs = [
        {
            'class': MyCIFAR100Einstellung,
            'name': 'Einstellung 32x32',
            'sample_size': 50
        },
        {
            'class': MyEinstellungCIFAR100_224,
            'name': 'Einstellung 224x224',
            'sample_size': 30  # Smaller for 224x224
        }
    ]

    all_passed = True

    for config in test_configs:
        try:
            # Test cross-method consistency
            if not test_cross_method_consistency(
                config['class'],
                config['name'],
                config['sample_size']
            ):
                all_passed = False

            # Test iteration determinism
            if not test_iteration_determinism(
                config['class'],
                config['name'],
                iterations=3,
                sample_size=20
            ):
                all_passed = False

        except Exception as e:
            print(f"❌ Test failed for {config['name']}: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Cross-method consistency achieved with deterministic transforms")
        print("✅ Iteration order is deterministic")
        print("✅ Ready for fair comparative experiments")
    else:
        print("❌ SOME TESTS FAILED!")
        print("The deterministic approach needs more refinement")

    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
