#!/usr/bin/env python3
"""
Test script for cache loading and validation functionality.
Tests the implementation of task 4: cache loading and validation.
"""

import os
import sys
import tempfile
import shutil
import pickle
import numpy as np
from argparse import Namespace

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung
from utils.conf import base_path


def test_cache_loading_validation():
    """Test cache loading and validation functionality."""
    print("Testing cache loading and validation...")

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Test 1: Load valid cache
        print("\n1. Testing valid cache loading...")

        # Create a small dataset for testing
        dataset = MyCIFAR100Einstellung(
            root=base_path() + 'CIFAR100',
            train=False,  # Use test set (smaller)
            download=True,
            apply_shortcut=True,
            patch_size=4,
            enable_cache=False  # Disable cache initially
        )

        # Manually create a valid cache file
        cache_key = dataset._get_cache_key()
        cache_dir = os.path.join(base_path(), 'CIFAR100', 'einstelluche')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'test_{cache_key}.pkl')

        # Create valid cache data (just first 10 items for testing)
        test_size = min(10, len(dataset.data))
        processed_images = []
        targets = []
        not_aug_images = []

        for i in range(test_size):
            img_array = dataset.data[i]
            target = dataset.targets[i]
            processed_images.append(img_array)
            targets.append(target)
            not_aug_images.append(img_array)

        cache_data = {
            'processed_images': np.array(processed_images),
            'targets': np.array(targets),
            'not_aug_images': np.array(not_aug_images),
            'params_hash': cache_key
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        # Test loading the cache
        dataset_cached = MyCIFAR100Einstellung(
            root=base_path() + 'CIFAR100',
            train=False,
            download=True,
            apply_shortcut=True,
            patch_size=4,
            enable_cache=True
        )

        # Manually set smaller data for testing
        dataset_cached.data = dataset_cached.data[:test_size]
        dataset_cached.targets = dataset_cached.targets[:test_size]

        # Test cache loading
        dataset_cached._load_cache_with_validation(cache_path)

        if dataset_cached._cache_loaded:
            print("✓ Valid cache loaded successfully")
        else:
            print("✗ Failed to load valid cache")
            return False

        # Test 2: Invalid cache (wrong hash)
        print("\n2. Testing invalid cache (parameter mismatch)...")

        invalid_cache_data = cache_data.copy()
        invalid_cache_data['params_hash'] = 'invalid_hash'

        invalid_cache_path = os.path.join(cache_dir, f'invalid_{cache_key}.pkl')
        with open(invalid_cache_path, 'wb') as f:
            pickle.dump(invalid_cache_data, f)

        dataset_invalid = MyCIFAR100Einstellung(
            root=base_path() + 'CIFAR100',
            train=False,
            download=True,
            apply_shortcut=True,
            patch_size=4,
            enable_cache=False  # We'll manually test loading
        )

        dataset_invalid.data = dataset_invalid.data[:test_size]
        dataset_invalid.targets = dataset_invalid.targets[:test_size]

        # This should trigger cache rebuild due to hash mismatch
        original_build_cache = dataset_invalid._build_cache_with_validation
        build_cache_called = False

        def mock_build_cache(path):
            nonlocal build_cache_called
            build_cache_called = True
            # Don't actually build, just mark as called

        dataset_invalid._build_cache_with_validation = mock_build_cache
        dataset_invalid._load_cache_with_validation(invalid_cache_path)

        if build_cache_called:
            print("✓ Invalid cache correctly triggered rebuild")
        else:
            print("✗ Invalid cache did not trigger rebuild")
            return False

        # Test 3: Corrupted cache file
        print("\n3. Testing corrupted cache file...")

        corrupted_cache_path = os.path.join(cache_dir, f'corrupted_{cache_key}.pkl')
        with open(corrupted_cache_path, 'wb') as f:
            f.write(b'corrupted data')

        dataset_corrupted = MyCIFAR100Einstellung(
            root=base_path() + 'CIFAR100',
            train=False,
            download=True,
            apply_shortcut=True,
            patch_size=4,
            enable_cache=False
        )

        build_cache_called = False
        dataset_corrupted._build_cache_with_validation = mock_build_cache
        dataset_corrupted._load_cache_with_validation(corrupted_cache_path)

        if build_cache_called:
            print("✓ Corrupted cache correctly triggered rebuild")
        else:
            print("✗ Corrupted cache did not trigger rebuild")
            return False

        # Test 4: _get_cached_item functionality
        print("\n4. Testing _get_cached_item functionality...")

        # Use the valid cached dataset from test 1
        if dataset_cached._cache_loaded:
            try:
                # Test valid index
                item = dataset_cached._get_cached_item(0)
                if len(item) == 3:  # (image, target, not_aug_image)
                    print("✓ _get_cached_item returns correct format")
                else:
                    print(f"✗ _get_cached_item returned {len(item)} items, expected 3")
                    return False

                # Test invalid index
                try:
                    dataset_cached._get_cached_item(test_size + 10)  # Out of bounds
                    print("✗ _get_cached_item should have raised IndexError")
                    return False
                except IndexError:
                    print("✓ _get_cached_item correctly handles invalid index")

            except Exception as e:
                print(f"✗ _get_cached_item failed: {e}")
                return False

        # Test 5: File size validation
        print("\n5. Testing file size validation...")

        tiny_cache_path = os.path.join(cache_dir, f'tiny_{cache_key}.pkl')
        with open(tiny_cache_path, 'wb') as f:
            f.write(b'tiny')  # Less than 1KB

        dataset_tiny = MyCIFAR100Einstellung(
            root=base_path() + 'CIFAR100',
            train=False,
            download=True,
            apply_shortcut=True,
            patch_size=4,
            enable_cache=False
        )

        build_cache_called = False
        dataset_tiny._build_cache_with_validation = mock_build_cache
        dataset_tiny._load_cache_with_validation(tiny_cache_path)

        if build_cache_called:
            print("✓ Tiny cache file correctly triggered rebuild")
        else:
            print("✗ Tiny cache file did not trigger rebuild")
            return False

        print("\n✓ All cache loading and validation tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # Clean up test cache files
        cache_dir = os.path.join(base_path(), 'CIFAR100', 'einstellung_cache')
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.startswith('test_') or filename.startswith('invalid_') or filename.startswith('corrupted_') or filename.startswith('tiny_'):
                    try:
                        os.remove(os.path.join(cache_dir, filename))
                    except:
                        pass


if __name__ == '__main__':
    success = test_cache_loading_validation()
    sys.exit(0 if success else 1)
