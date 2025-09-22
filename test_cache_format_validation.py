#!/usr/bin/env python3
"""
Test script to verify cached data maintains exact same format as original dataset.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung
from utils.conf import base_path


def test_cache_format_consistency():
    """Test that cached data maintains exact same format as original dataset."""
    print("Testing cache format consistency...")

    try:
        # Create two identical datasets - one with cache, one without
        print("\n1. Creating datasets...")

        dataset_original = MyCIFAR100Einstellung(
            root=base_path() + 'CIFAR100',
            train=False,  # Use test set for faster testing
            download=True,
            apply_shortcut=True,
            patch_size=4,
            patch_color=(255, 0, 255),
            enable_cache=False  # No cache
        )

        dataset_cached = MyCIFAR100Einstellung(
            root=base_path() + 'CIFAR100',
            train=False,
            download=True,
            apply_shortcut=True,
            patch_size=4,
            patch_color=(255, 0, 255),
            enable_cache=True  # With cache
        )

        # Test with a small subset for speed
        test_indices = [0, 1, 2, 5, 10]

        print("\n2. Comparing data formats...")

        for i, idx in enumerate(test_indices):
            print(f"Testing item {idx}...")

            # Get items from both datasets
            original_item = dataset_original[idx]
            cached_item = dataset_cached[idx]

            # Check tuple length
            if len(original_item) != len(cached_item):
                print(f"✗ Tuple length mismatch: original={len(original_item)}, cached={len(cached_item)}")
                return False

            # Check each component
            orig_img, orig_target, orig_not_aug = original_item[:3]
            cached_img, cached_target, cached_not_aug = cached_item[:3]

            # Check target type and value
            if type(orig_target) != type(cached_target):
                print(f"✗ Target type mismatch: original={type(orig_target)}, cached={type(cached_target)}")
                return False

            if orig_target != cached_target:
                print(f"✗ Target value mismatch: original={orig_target}, cached={cached_target}")
                return False

            # Check image types
            if type(orig_img) != type(cached_img):
                print(f"✗ Image type mismatch: original={type(orig_img)}, cached={type(cached_img)}")
                return False

            if type(orig_not_aug) != type(cached_not_aug):
                print(f"✗ Not-aug image type mismatch: original={type(orig_not_aug)}, cached={type(cached_not_aug)}")
                return False

            # Check tensor shapes if they are tensors
            if isinstance(orig_img, torch.Tensor):
                if orig_img.shape != cached_img.shape:
                    print(f"✗ Image shape mismatch: original={orig_img.shape}, cached={cached_img.shape}")
                    return False

            if isinstance(orig_not_aug, torch.Tensor):
                if orig_not_aug.shape != cached_not_aug.shape:
                    print(f"✗ Not-aug shape mismatch: original={orig_not_aug.shape}, cached={cached_not_aug.shape}")
                    return False

            # Check if logits are present in both or neither
            has_orig_logits = len(original_item) > 3
            has_cached_logits = len(cached_item) > 3

            if has_orig_logits != has_cached_logits:
                print(f"✗ Logits presence mismatch: original={has_orig_logits}, cached={has_cached_logits}")
                return False

            print(f"✓ Item {idx} format matches")

        print("\n3. Testing data consistency...")

        # Test that the same index returns consistent results
        for idx in test_indices:
            cached_item1 = dataset_cached[idx]
            cached_item2 = dataset_cached[idx]

            # Should be identical
            if cached_item1[1] != cached_item2[1]:  # target
                print(f"✗ Cached data inconsistent for index {idx}")
                return False

        print("✓ Cached data is consistent")

        print("\n4. Testing error handling...")

        # Test invalid index handling
        try:
            invalid_idx = len(dataset_cached) + 100
            dataset_cached[invalid_idx]
            print("✗ Should have raised IndexError for invalid index")
            return False
        except IndexError:
            print("✓ Invalid index correctly handled")

        print("\n✓ All cache format consistency tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_cache_format_consistency()
    sys.exit(0 if success else 1)
