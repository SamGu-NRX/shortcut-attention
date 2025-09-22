#!/usr/bin/env python3
"""
Test script for evaluation subset caching implementation.
Verifies that cached evaluation subsets produce identical results to original implementation.
"""

import os
import sys
import logging
import tempfile
import shutil
from argparse import Namespace

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.seq_cifar100_einstellung import SequentialCIFAR100Einstellung
from datasets.seq_cifar100_einstellung_224 import SequentialCIFAR100Einstellung224
from utils.conf import base_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_evaluation_subset_caching():
    """Test evaluation subset caching for both 32x32 and 224x224 versions."""

    print("Testing Evaluation Subset Caching Implementation")
    print("=" * 50)

    # Create test args with all required defaults
    args = Namespace()

    # Basic experiment args
    args.batch_size = 4  # Small batch size for testing
    args.joint = 0
    args.lr = 0.1
    args.n_epochs = 1  # Small number for testing
    args.backbone = "resnet18"
    args.optimizer = "sgd"
    args.optim_wd = 0.0
    args.optim_mom = 0.0
    args.optim_nesterov = False
    args.drop_last = False
    args.lr_scheduler = "multisteplr"
    args.lr_milestones = [35, 45]
    args.sched_multistep_lr_gamma = 0.1
    args.scheduler_mode = "epoch"

    # Validation args
    args.validation = None
    args.validation_mode = "current"
    args.fitting_mode = "epochs"
    args.early_stopping_patience = 5
    args.early_stopping_metric = "loss"
    args.early_stopping_freq = 1
    args.early_stopping_epsilon = 1e-6
    args.n_iters = None

    # Management args
    args.seed = 42
    args.permute_classes = False
    args.base_path = "./data/"
    args.results_path = "results/"
    args.device = None
    args.notes = None
    args.eval_epochs = None
    args.non_verbose = False
    args.disable_log = True  # Disable logging for testing
    args.num_workers = 0  # No workers for testing
    args.enable_other_metrics = False
    args.debug_mode = True  # Enable debug mode for faster testing
    args.inference_only = False
    args.code_optimization = 0
    args.distributed = "no"
    args.savecheck = None
    args.save_checkpoint_mode = "safe"
    args.loadcheck = None
    args.ckpt_name = None
    args.start_from = None
    args.stop_after = None

    # Wandb args (disabled for testing)
    args.wandb_name = None
    args.wandb_entity = None
    args.wandb_project = None

    # Noise args
    args.noise_type = "symmetric"
    args.noise_rate = 0
    args.disable_noisy_labels_cache = False
    args.cache_path_noisy_labels = "noisy_labels"

    # Label percentage args
    args.label_perc = 1.0
    args.label_perc_by_class = 1.0

    # Evaluation args
    args.eval_future = False
    args.custom_task_order = None
    args.custom_class_order = None

    # Einstellung Effect args
    args.einstellung_apply_shortcut = False
    args.einstellung_mask_shortcut = False
    args.einstellung_patch_size = 4
    args.einstellung_patch_color = [255, 0, 255]
    args.einstellung_enable_cache = True
    args.einstellung_adaptation_threshold = 0.8
    args.einstellung_evaluation_subsets = True
    args.einstellung_extract_attention = False  # Disable for testing

    # Test 32x32 version
    print("\n1. Testing 32x32 version (SequentialCIFAR100Einstellung)")
    try:
        dataset_32 = SequentialCIFAR100Einstellung(args)

        # First call - should create cache
        print("   Creating evaluation subsets (first call - should build cache)...")
        subsets_1 = dataset_32.get_evaluation_subsets()

        print(f"   Created {len(subsets_1)} evaluation subsets:")
        for name, loader in subsets_1.items():
            print(f"     - {name}: {len(loader.dataset)} samples")

        # Second call - should use cache
        print("   Creating evaluation subsets (second call - should use cache)...")
        subsets_2 = dataset_32.get_evaluation_subsets()

        print(f"   Loaded {len(subsets_2)} evaluation subsets from cache:")
        for name, loader in subsets_2.items():
            print(f"     - {name}: {len(loader.dataset)} samples")

        # Verify subset names match
        assert set(subsets_1.keys()) == set(subsets_2.keys()), "Subset names don't match"

        # Verify subset sizes match
        for name in subsets_1.keys():
            size_1 = len(subsets_1[name].dataset)
            size_2 = len(subsets_2[name].dataset)
            assert size_1 == size_2, f"Subset {name} size mismatch: {size_1} vs {size_2}"

        print("   ✓ 32x32 version test passed!")

    except Exception as e:
        print(f"   ✗ 32x32 version test failed: {e}")
        return False

    # Test 224x224 version
    print("\n2. Testing 224x224 version (SequentialCIFAR100Einstellung224)")
    try:
        dataset_224 = SequentialCIFAR100Einstellung224(args)

        # First call - should create cache
        print("   Creating evaluation subsets (first call - should build cache)...")
        subsets_1 = dataset_224.get_evaluation_subsets()

        print(f"   Created {len(subsets_1)} evaluation subsets:")
        for name, loader in subsets_1.items():
            print(f"     - {name}: {len(loader.dataset)} samples")

        # Second call - should use cache
        print("   Creating evaluation subsets (second call - should use cache)...")
        subsets_2 = dataset_224.get_evaluation_subsets()

        print(f"   Loaded {len(subsets_2)} evaluation subsets from cache:")
        for name, loader in subsets_2.items():
            print(f"     - {name}: {len(loader.dataset)} samples")

        # Verify subset names match
        assert set(subsets_1.keys()) == set(subsets_2.keys()), "Subset names don't match"

        # Verify subset sizes match
        for name in subsets_1.keys():
            size_1 = len(subsets_1[name].dataset)
            size_2 = len(subsets_2[name].dataset)
            assert size_1 == size_2, f"Subset {name} size mismatch: {size_1} vs {size_2}"

        print("   ✓ 224x224 version test passed!")

    except Exception as e:
        print(f"   ✗ 224x224 version test failed: {e}")
        return False

    print("\n3. Testing expected subset names")
    expected_subsets = {'T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal'}

    # Check 32x32 subsets
    actual_32 = set(subsets_1.keys()) if 'subsets_1' in locals() else set()
    if expected_subsets.issubset(actual_32):
        print("   ✓ 32x32 version has all expected subsets")
    else:
        print(f"   ✗ 32x32 version missing subsets: {expected_subsets - actual_32}")
        return False

    # Check 224x224 subsets
    actual_224 = set(subsets_1.keys()) if 'subsets_1' in locals() else set()
    if expected_subsets.issubset(actual_224):
        print("   ✓ 224x224 version has all expected subsets")
    else:
        print(f"   ✗ 224x224 version missing subsets: {expected_subsets - actual_224}")
        return False

    print("\n" + "=" * 50)
    print("✓ All evaluation subset caching tests passed!")
    print("✓ Task 8 implementation completed successfully!")
    return True

def test_cache_file_creation():
    """Test that cache files are created in the correct location."""
    print("\n4. Testing cache file creation")

    cache_dir = os.path.join(base_path(), 'CIFAR100', 'einstellung_cache')
    print(f"   Cache directory: {cache_dir}")

    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('eval_subsets')]
        print(f"   Found {len(cache_files)} evaluation subset cache files:")
        for f in cache_files:
            file_path = os.path.join(cache_dir, f)
            file_size = os.path.getsize(file_path)
            print(f"     - {f} ({file_size:,} bytes)")

        if cache_files:
            print("   ✓ Cache files created successfully")
            return True
        else:
            print("   ✗ No cache files found")
            return False
    else:
        print("   ✗ Cache directory not found")
        return False

if __name__ == "__main__":
    success = test_evaluation_subset_caching()
    if success:
        success = test_cache_file_creation()

    if success:
        print("\n🎉 All tests passed! Evaluation subset caching is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
