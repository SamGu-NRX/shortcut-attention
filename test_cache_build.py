#!/usr/bin/env python3
"""
Test script to verify the cache building functionality works correctly.
"""

import sys
import os
import logging
import tempfile
import shutil
from argparse import Namespace

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_cache_build():
    """Test that cache building works correctly."""

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing cache build in temporary directory: {temp_dir}")

        # Create a small test dataset with caching enabled
        dataset = MyCIFAR100Einstellung(
            root=temp_dir,
            train=True,
            download=True,
            apply_shortcut=True,
            mask_shortcut=False,
            patch_size=4,
            patch_color=(255, 0, 255),
            enable_cache=True
        )

        print(f"Dataset created with {len(dataset.data)} images")

        # Check if cache was built
        if dataset._cache_loaded and dataset._cached_data is not None:
            print("✅ Cache was built successfully!")
            print(f"   - Cached images: {len(dataset._cached_data['processed_images'])}")
            print(f"   - Cache params hash: {dataset._cached_data['params_hash']}")

            # Test getting a few items
            for i in range(min(3, len(dataset))):
                img, target, not_aug_img = dataset[i]
                print(f"   - Item {i}: target={target}, img_shape={img.shape if hasattr(img, 'shape') else 'PIL'}")

            return True
        else:
            print("❌ Cache was not built")
            return False

if __name__ == "__main__":
    success = test_cache_build()
    sys.exit(0 if success else 1)
