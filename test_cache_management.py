#!/usr/bin/env python3
"""
Test script for Einstellung dataset cache management utilities.
Tests the core cache management methods implemented in task 2.
"""

import os
import sys
import tempfile
import shutil
import hashlib
from unittest.mock import patch

# Add the project root to Python path
sys.path.insert, os.path.dirname(os.path.abspath(__file__))

from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung
from utils.conf import base_path


def test_get_cache_key():
    """Test that _get_cache_key generates consistent hashes for same parameters."""
    print("Testing _get_cache_key method...")

    # Create two datasets with identical parameters
    dataset1 = MyCIFAR100Einstellung(
        root=tempfile.mkdtemp(),
        train=True,
        download=False,
        apply_shortcut=True,
        mask_shortcut=False,
        patch_size=4,
        patch_color=(255, 0, 255),
        enable_cache=False  # Disable cache for testing
    )

    dataset2 = MyCIFAR100Einstellung(
        root=tempfile.mkdtemp(),
        train=True,
        download=False,
        apply_shortcut=True,
        mask_shortcut=False,
        patch_size=4,
        patch_color=(255, 0, 255),
        enable_cache=False
    )

    # Test that identical parameters produce identical cache keys
    key1 = dataset1._get_cache_key()
    key2 = dataset2._get_cache_key()

    assert key1 == key2, f"Identical parameters should produce identical cache keys: {key1} != {key2}"
    print(f"✓ Identical parameters produce identical cache key: {key1}")

    # Test that different parameters produce different cache keys
    dataset3 = MyCIFAR100Einstellung(
        root=tempfile.mkdtemp(),
        train=True,
        download=False,
        apply_shortcut=False,  # Different parameter
        mask_shortcut=False,
        patch_size=4,
        patch_color=(255, 0, 255),
        enable_cache=False
    )

    key3 = dataset3._get_cache_key()
    assert key1 != key3, f"Different parameters should produce different cache keys: {key1} == {key3}"
    print(f"✓ Different parameters produce different cache keys: {key1} vs {key3}")

    # Test that cache key is a valid hash (16 characters, hexadecimal)
    assert len(key1) == 16, f"Cache key should be 16 characters long: {len(key1)}"
    assert all(c in '0123456789abcdef' for c in key1), f"Cache key should be hexadecimal: {key1}"
    print(f"✓ Cache key format is valid: {key1}")

    # Test all Einstellung parameters are included in hash
    test_cases = [
        {'apply_shortcut': False, 'mask_shortcut': True},
        {'patch_size': 8},
        {'patch_color': (0, 255, 0)},
        {'train': False}
    ]

    for i, params in enumerate(test_cases):
        dataset_test = MyCIFAR100Einstellung(
            root=tempfile.mkdtemp(),
            train=params.get('train', True),
            download=False,
            apply_shortcut=params.get('apply_shortcut', True),
            mask_shortcut=params.get('mask_shortcut', False),
            patch_size=params.get('patch_size', 4),
            patch_color=params.get('patch_color', (255, 0, 255)),
            enable_cache=False
        )
        key_test = dataset_test._get_cache_key()
        assert key_test != key1, f"Parameter change {i+1} should change cache key: {key_test} == {key1}"
        print(f"✓ Parameter change {i+1} produces different cache key: {key_test}")


def test_get_cache_path():
    """Test that _get_cache_path uses correct directory structure and naming."""
    print("\nTesting _get_cache_path method...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock base_path to use our temporary directory
        with patch('datasets.seq_cifar100_einstellung.base_path', return_value=temp_dir):
            # Test train dataset
            dataset_train = MyCIFAR100Einstellung(
                root=temp_dir,
                train=True,
                download=False,
                apply_shortcut=True,
                mask_shortcut=False,
                patch_size=4,
                patch_color=(255, 0, 255),
                enable_cache=False
            )

            cache_path_train = dataset_train._get_cache_path()
            expected_dir = os.path.join(temp_dir, 'CIFAR100', 'einstellung_cache')

            # Test directory structure
            assert cache_path_train.startswith(expected_dir), f"Cache path should be in correct directory: {cache_path_train}"
            print(f"✓ Cache path uses correct directory: {expected_dir}")

            # Test that directory is created
            assert os.path.exists(expected_dir), f"Cache directory should be created: {expected_dir}"
            print(f"✓ Cache directory is created automatically")

            # Test filename format
            cache_filename = os.path.basename(cache_path_train)
            assert cache_filename.startswith('train_'), f"Train cache filename should start with 'train_': {cache_filename}"
            assert cache_filename.endswith('.pkl'), f"Cache filename should end with '.pkl': {cache_filename}"
            print(f"✓ Train cache filename format is correct: {cache_filename}")

            # Test test dataset
            dataset_test = MyCIFAR100Einstellung(
                root=temp_dir,
                train=False,
                download=False,
                apply_shortcut=True,
                mask_shortcut=False,
                patch_size=4,
                patch_color=(255, 0, 255),
                enable_cache=False
            )

            cache_path_test = dataset_test._get_cache_path()
            cache_filename_test = os.path.basename(cache_path_test)
            assert cache_filename_test.startswith('test_'), f"Test cache filename should start with 'test_': {cache_filename_test}"
            print(f"✓ Test cache filename format is correct: {cache_filename_test}")

            # Test that different parameters produce different paths
            dataset_different = MyCIFAR100Einstellung(
                root=temp_dir,
                train=True,
                download=False,
                apply_shortcut=False,  # Different parameter
                mask_shortcut=False,
                patch_size=4,
                patch_color=(255, 0, 255),
                enable_cache=False
            )

            cache_path_different = dataset_different._get_cache_path()
            assert cache_path_train != cache_path_different, f"Different parameters should produce different cache paths"
            print(f"✓ Different parameters produce different cache paths")


def test_parameter_validation():
    """Test parameter validation using hash comparison."""
    print("\nTesting parameter validation...")

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('datasets.seq_cifar100_einstellung.base_path', return_value=temp_dir):
            # Create dataset with specific parameters
            dataset = MyCIFAR100Einstellung(
                root=temp_dir,
                train=True,
                download=False,
                apply_shortcut=True,
                mask_shortcut=False,
                patch_size=4,
                patch_color=(255, 0, 255),
                enable_cache=False
            )

            # Test that parameter hash is consistent
            hash1 = dataset._get_cache_key()
            hash2 = dataset._get_cache_key()
            assert hash1 == hash2, f"Cache key should be consistent: {hash1} != {hash2}"
            print(f"✓ Parameter hash is consistent: {hash1}")

            # Test that changing parameters changes hash
            original_apply_shortcut = dataset.apply_shortcut
            dataset.apply_shortcut = not original_apply_shortcut
            hash3 = dataset._get_cache_key()
            assert hash1 != hash3, f"Changing parameters should change hash: {hash1} == {hash3}"
            print(f"✓ Parameter changes detected by hash: {hash1} -> {hash3}")

            # Restore original parameter
            dataset.apply_shortcut = original_apply_shortcut
            hash4 = dataset._get_cache_key()
            assert hash1 == hash4, f"Restoring parameters should restore hash: {hash1} != {hash4}"
            print(f"✓ Parameter restoration detected by hash: {hash3} -> {hash4}")


def test_cache_key_security():
    """Test that cache key uses secure hashing (SHA-256)."""
    print("\nTesting cache key security...")

    dataset = MyCIFAR100Einstellung(
        root=tempfile.mkdtemp(),
        train=True,
        download=False,
        apply_shortcut=True,
        mask_shortcut=False,
        patch_size=4,
        patch_color=(255, 0, 255),
        enable_cache=False
    )

    # Get the cache key
    cache_key = dataset._get_cache_key()

    # Manually compute expected hash to verify it's using SHA-256
    params = {
        'apply_shortcut': dataset.apply_shortcut,
        'mask_shortcut': dataset.mask_shortcut,
        'patch_size': dataset.patch_size,
        'patch_color': tuple(dataset.patch_color),
        'train': dataset.train
    }
    param_str = str(sorted(params.items()))
    expected_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]

    assert cache_key == expected_hash, f"Cache key should match SHA-256 hash: {cache_key} != {expected_hash}"
    print(f"✓ Cache key uses secure SHA-256 hashing: {cache_key}")


def main():
    """Run all cache management tests."""
    print("Testing Einstellung Dataset Cache Management Utilities")
    print("=" * 60)

    try:
        test_get_cache_key()
        test_get_cache_path()
        test_parameter_validation()
        test_cache_key_security()

        print("\n" + "=" * 60)
        print("✅ All cache management utility tests passed!")
        print("\nTask 2 implementation verified:")
        print("- ✓ _get_cache_key() uses secure hash of Einstellung parameters")
        print("- ✓ _get_cache_path() uses Mammoth's base_path() + 'CIFAR100/einstellung_cache/'")
        print("- ✓ Parameter validation using hash comparison detects configuration changes")
        print("- ✓ All requirements (1.3, 1.4, 2.3, 6.1) are satisfied")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
