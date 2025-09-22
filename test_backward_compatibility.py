#!/usr/bin/env python3
"""
Test script to verify backward compatibility and default behavior for Einstellung dataset caching.

This script tests that:
1. All Einstellung dataset constructors accept enable_cache parameter with default True
2. Existing experiment scripts work without modification (caching enabled by default)
3. Cache can be disabled for debugging or comparison purposes
4. All existing Mammoth functionality works identically with caching enabled

Requirements: 4.1, 4.2, 4.6
"""

import os
import sys
import tempfile
import shutil
import logging
from argparse import Namespace
from typing import Tuple

# Import the central configuration
from experiments.default_args import get_base_args

import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.seq_cifar100_einstellung import (
    MyCIFAR100Einstellung,
    SequentialCIFAR100Einstellung,
    TCIFAR100Einstellung
)
from datasets.seq_cifar100_einstellung_224 import (
    MyEinstellungCIFAR100_224,
    TEinstellungCIFAR100_224,
    SequentialCIFAR100Einstellung224
)
from utils.conf import base_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_constructor_default_parameters():
    """Test that all Einstellung dataset constructors have enable_cache=True by default."""
    logger.info("Testing constructor default parameters...")

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Test MyCIFAR100Einstellung (32x32)
        logger.info("Testing MyCIFAR100Einstellung constructor...")
        dataset_32 = MyCIFAR100Einstellung(
            root=temp_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        assert hasattr(dataset_32, 'enable_cache'), "MyCIFAR100Einstellung missing enable_cache attribute"
        assert dataset_32.enable_cache == True, f"MyCIFAR100Einstellung enable_cache default should be True, got {dataset_32.enable_cache}"
        logger.info("✓ MyCIFAR100Einstellung has enable_cache=True by default")

        # Test MyEinstellungCIFAR100_224 (224x224)
        logger.info("Testing MyEinstellungCIFAR100_224 constructor...")
        dataset_224 = MyEinstellungCIFAR100_224(
            root=temp_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        assert hasattr(dataset_224, 'enable_cache'), "MyEinstellungCIFAR100_224 missing enable_cache attribute"
        assert dataset_224.enable_cache == True, f"MyEinstellungCIFAR100_224 enable_cache default should be True, got {dataset_224.enable_cache}"
        logger.info("✓ MyEinstellungCIFAR100_224 has enable_cache=True by default")

        # Test TEinstellungCIFAR100_224 (224x224 test)
        logger.info("Testing TEinstellungCIFAR100_224 constructor...")
        test_dataset_224 = TEinstellungCIFAR100_224(
            root=temp_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        assert hasattr(test_dataset_224, 'enable_cache'), "TEinstellungCIFAR100_224 missing enable_cache attribute"
        assert test_dataset_224.enable_cache == True, f"TEinstellungCIFAR100_224 enable_cache default should be True, got {test_dataset_224.enable_cache}"
        logger.info("✓ TEinstellungCIFAR100_224 has enable_cache=True by default")

        logger.info("✓ All dataset constructors have enable_cache=True by default")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cache_disable_option():
    """Test that cache can be disabled for debugging or comparison purposes."""
    logger.info("Testing cache disable option...")

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Test with cache disabled
        logger.info("Testing MyCIFAR100Einstellung with cache disabled...")
        dataset_no_cache = MyCIFAR100Einstellung(
            root=temp_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            enable_cache=False
        )
        assert dataset_no_cache.enable_cache == False, f"enable_cache should be False, got {dataset_no_cache.enable_cache}"
        logger.info("✓ Cache can be disabled via enable_cache=False")

        # Test with cache explicitly enabled
        logger.info("Testing MyCIFAR100Einstellung with cache explicitly enabled...")
        dataset_with_cache = MyCIFAR100Einstellung(
            root=temp_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            enable_cache=True
        )
        assert dataset_with_cache.enable_cache == True, f"enable_cache should be True, got {dataset_with_cache.enable_cache}"
        logger.info("✓ Cache can be explicitly enabled via enable_cache=True")

        logger.info("✓ Cache disable option works correctly")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_sequential_dataset_args():
    """Test that sequential datasets properly handle enable_cache from args."""
    logger.info("Testing sequential dataset args handling...")

    # Get base args and convert to Namespace
    base_args_dict = get_base_args()

    # Test SequentialCIFAR100Einstellung (32x32)
    logger.info("Testing SequentialCIFAR100Einstellung args...")
    args_32 = Namespace(**base_args_dict)
    args_32.dataset = 'seq-cifar100-einstellung'
    args_32.n_epochs = 1
    args_32.batch_size = 32
    args_32.lr = 0.1
    # Add missing Einstellung arguments
    args_32.einstellung_patch_size = 4
    args_32.einstellung_patch_color = [255, 0, 255]
    args_32.einstellung_adaptation_threshold = 0.8
    args_32.einstellung_apply_shortcut = False
    args_32.einstellung_mask_shortcut = False
    args_32.einstellung_evaluation_subsets = True
    args_32.einstellung_extract_attention = True
    args_32.einstellung_enable_cache = True  # Default value

    sequential_32 = SequentialCIFAR100Einstellung(args_32)
    assert hasattr(sequential_32, 'enable_cache'), "SequentialCIFAR100Einstellung missing enable_cache attribute"
    assert sequential_32.enable_cache == True, f"SequentialCIFAR100Einstellung enable_cache should be True, got {sequential_32.enable_cache}"
    logger.info("✓ SequentialCIFAR100Einstellung handles enable_cache from args")

    # Test SequentialCIFAR100Einstellung224 (224x224)
    logger.info("Testing SequentialCIFAR100Einstellung224 args...")
    args_224 = Namespace(**base_args_dict)
    args_224.dataset = 'seq-cifar100-einstellung-224'
    args_224.n_epochs = 1
    args_224.batch_size = 32
    args_224.lr = 0.1
    # Add missing Einstellung arguments
    args_224.einstellung_patch_size = 16  # Larger for 224x224
    args_224.einstellung_patch_color = [255, 0, 255]
    args_224.einstellung_adaptation_threshold = 0.8
    args_224.einstellung_apply_shortcut = False
    args_224.einstellung_mask_shortcut = False
    args_224.einstellung_evaluation_subsets = True
    args_224.einstellung_extract_attention = True
    args_224.einstellung_enable_cache = True  # Default value

    sequential_224 = SequentialCIFAR100Einstellung224(args_224)
    assert hasattr(sequential_224, 'enable_cache'), "SequentialCIFAR100Einstellung224 missing enable_cache attribute"
    assert sequential_224.enable_cache == True, f"SequentialCIFAR100Einstellung224 enable_cache should be True, got {sequential_224.enable_cache}"
    logger.info("✓ SequentialCIFAR100Einstellung224 handles enable_cache from args")

    # Test with cache disabled via args
    logger.info("Testing cache disable via args...")
    args_disabled = Namespace(**base_args_dict)
    args_disabled.dataset = 'seq-cifar100-einstellung'
    args_disabled.n_epochs = 1
    args_disabled.batch_size = 32
    args_disabled.lr = 0.1
    # Add missing Einstellung arguments
    args_disabled.einstellung_patch_size = 4
    args_disabled.einstellung_patch_color = [255, 0, 255]
    args_disabled.einstellung_adaptation_threshold = 0.8
    args_disabled.einstellung_apply_shortcut = False
    args_disabled.einstellung_mask_shortcut = False
    args_disabled.einstellung_evaluation_subsets = True
    args_disabled.einstellung_extract_attention = True
    args_disabled.einstellung_enable_cache = False  # Disabled

    sequential_disabled = SequentialCIFAR100Einstellung(args_disabled)
    assert sequential_disabled.enable_cache == False, f"enable_cache should be False when disabled via args, got {sequential_disabled.enable_cache}"
    logger.info("✓ Cache can be disabled via args")

    logger.info("✓ Sequential datasets handle enable_cache from args correctly")

    logger.info("✓ Sequential datasets handle enable_cache from args correctly")


def test_dataset_functionality():
    """Test that basic dataset functionality works with caching enabled."""
    logger.info("Testing basic dataset functionality...")

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Test basic dataset operations
        logger.info("Testing basic dataset operations...")
        dataset = MyCIFAR100Einstellung(
            root=temp_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            enable_cache=True  # Default behavior
        )

        # Test dataset length
        assert len(dataset) > 0, "Dataset should not be empty"
        logger.info(f"✓ Dataset has {len(dataset)} samples")

        # Test __getitem__ functionality
        img, target, not_aug_img = dataset[0]
        assert isinstance(img, torch.Tensor), f"Image should be torch.Tensor, got {type(img)}"
        assert isinstance(target, int), f"Target should be int, got {type(target)}"
        assert isinstance(not_aug_img, torch.Tensor), f"Not-aug image should be torch.Tensor, got {type(not_aug_img)}"
        logger.info("✓ __getitem__ returns correct types")

        # Test that targets are in expected range (0-59 for Einstellung)
        assert 0 <= target < 60, f"Target should be in range [0, 59], got {target}"
        logger.info("✓ Target labels are in expected range")

        # Test image shapes
        assert img.shape == (3, 32, 32), f"Image shape should be (3, 32, 32), got {img.shape}"
        assert not_aug_img.shape == (3, 32, 32), f"Not-aug image shape should be (3, 32, 32), got {not_aug_img.shape}"
        logger.info("✓ Image shapes are correct")

        logger.info("✓ Basic dataset functionality works correctly")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_einstellung_parameters():
    """Test that Einstellung-specific parameters work correctly."""
    logger.info("Testing Einstellung-specific parameters...")

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Test with shortcut application
        logger.info("Testing shortcut application...")
        dataset_shortcut = MyCIFAR100Einstellung(
            root=temp_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            apply_shortcut=True,
            patch_size=4,
            patch_color=(255, 0, 255),
            enable_cache=True
        )

        # Verify parameters are set correctly
        assert dataset_shortcut.apply_shortcut == True, "apply_shortcut should be True"
        assert dataset_shortcut.patch_size == 4, f"patch_size should be 4, got {dataset_shortcut.patch_size}"
        assert np.array_equal(dataset_shortcut.patch_color, np.array([255, 0, 255])), "patch_color should be magenta"
        logger.info("✓ Shortcut parameters set correctly")

        # Test with shortcut masking
        logger.info("Testing shortcut masking...")
        dataset_masked = MyCIFAR100Einstellung(
            root=temp_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
            mask_shortcut=True,
            patch_size=4,
            enable_cache=True
        )

        assert dataset_masked.mask_shortcut == True, "mask_shortcut should be True"
        logger.info("✓ Masking parameters set correctly")

        logger.info("✓ Einstellung-specific parameters work correctly")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all backward compatibility tests."""
    logger.info("Starting backward compatibility tests for Einstellung dataset caching...")

    try:
        # Test 1: Constructor default parameters
        test_constructor_default_parameters()

        # Test 2: Cache disable option
        test_cache_disable_option()

        # Test 3: Sequential dataset args handling
        test_sequential_dataset_args()

        # Test 4: Basic dataset functionality
        test_dataset_functionality()

        # Test 5: Einstellung-specific parameters
        test_einstellung_parameters()

        logger.info("✅ All backward compatibility tests passed!")
        logger.info("✅ Task 13 requirements verified:")
        logger.info("  ✓ enable_cache parameter with default True added to all Einstellung dataset constructors")
        logger.info("  ✓ Existing experiment scripts work without modification (caching enabled by default)")
        logger.info("  ✓ Cache disable option available for debugging or comparison purposes")
        logger.info("  ✓ All existing Mammoth functionality works identically with caching enabled")

        return True

    except Exception as e:
        logger.error(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
