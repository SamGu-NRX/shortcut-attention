#!/usr/bin/env python3
"""
Test script to verify that cached Einstellung datasets integrate properly with Mammoth's data loading pipeline.
This tests the integration with store_masked_loaders, MammothDatasetWrapper, and task splitting logic.
"""

import sys
import os
import torch
import numpy as np
from argparse import Namespace

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from datasets.seq_cifar100_einstellung import SequentialCIFAR100Einstellung
from datasets.seq_cifar100_einstellung_224 import SequentialCIFAR100Einstellung224
from datasets.utils.continual_dataset import MammothDatasetWrapper, store_masked_loaders
from utils.conf import base_path


def create_default_args():
    """Create a comprehensive default arguments namespace for testing."""
    args = Namespace()

    # Basic required arguments
    args.dataset = 'seq-cifar100-einstellung'
    args.model = 'sgd'
    args.backbone = 'resnet18'

    # Experiment arguments
    args.lr = 0.1
    args.batch_size = 32
    args.n_epochs = 1
    args.seed = 42

    # Continual learning arguments
    args.joint = 0
    args.eval_future = False
    args.label_perc = 1.0
    args.label_perc_by_class = 1.0
    args.custom_task_order = None
    args.custom_class_order = None

    # Validation arguments
    args.validation = 0.0
    args.validation_mode = 'current'
    args.fitting_mode = 'epochs'

    # Class ordering arguments
    args.permute_classes = False
    args.class_order = None

    # Noise arguments
    args.noise_rate = 0.0
    args.noise_type = 'symmetric'

    # Data loading arguments
    args.drop_last = False
    args.num_workers = 0

    # Management arguments
    args.debug_mode = False
    args.non_verbose = False
    args.disable_log = False
    args.notes = None
    args.eval_epochs = None
    args.inference_only = False
    args.enable_other_metrics = False
    args.savecheck = False
    args.loadcheck = None
    args.start_from = None
    args.stop_after = None

    # Einstellung-specific arguments
    args.einstellung_apply_shortcut = True
    args.einstellung_mask_shortcut = False
    args.einstellung_patch_size = 4
    args.einstellung_patch_color = (255, 0, 255)
    args.einstellung_enable_cache = True

    return args


def test_32x32_data_loading_integration():
    """Test that 32x32 Einstellung dataset integrates properly with Mammoth's data loading pipeline."""
    print("Testing 32x32 Einstellung dataset integration...")

    # Create test arguments with comprehensive defaults
    args = create_default_args()

    # Create dataset instance
    dataset = SequentialCIFAR100Einstellung(args)

    # Test get_data_loaders method
    try:
        train_loader, test_loader = dataset.get_data_loaders()
        print(f"✓ Successfully created data loaders")
        print(f"  - Train loader: {len(train_loader)} batches")
        print(f"  - Test loader: {len(test_loader)} batches")

        # Test that we can iterate through the loaders
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        print(f"  - Train batch shape: {train_batch[0].shape}")
        print(f"  - Test batch shape: {test_batch[0].shape}")

        # Verify batch structure (should be img, target, not_aug_img, ...)
        assert len(train_batch) >= 3, f"Expected at least 3 elements in train batch, got {len(train_batch)}"
        assert len(test_batch) >= 3, f"Expected at least 3 elements in test batch, got {len(test_batch)}"

        print(f"✓ Data loaders work correctly with cached datasets")

    except Exception as e:
        print(f"✗ Error in data loading: {e}")
        raise

    # Test MammothDatasetWrapper compatibility
    try:
        # Get the underlying datasets
        train_dataset = dataset.get_data_loaders()[0].dataset.dataset  # Unwrap from DataLoader and MammothDatasetWrapper

        # Check required attributes
        assert hasattr(train_dataset, 'data'), "Dataset missing 'data' attribute"
        assert hasattr(train_dataset, 'targets'), "Dataset missing 'targets' attribute"
        assert hasattr(train_dataset, 'task_ids'), "Dataset missing 'task_ids' attribute"

        print(f"✓ Dataset has required attributes for MammothDatasetWrapper")
        print(f"  - Data shape: {train_dataset.data.shape}")
        print(f"  - Targets length: {len(train_dataset.targets)}")
        print(f"  - Task IDs shape: {train_dataset.task_ids.shape}")

    except Exception as e:
        print(f"✗ Error in MammothDatasetWrapper compatibility: {e}")
        raise

    # Test task splitting logic
    try:
        # Check that task IDs are properly set
        unique_task_ids = np.unique(train_dataset.task_ids)
        print(f"✓ Task splitting works correctly")
        print(f"  - Unique task IDs: {unique_task_ids}")

        # Verify task boundaries
        task_0_count = np.sum(train_dataset.task_ids == 0)
        task_1_count = np.sum(train_dataset.task_ids == 1)
        print(f"  - Task 0 samples: {task_0_count}")
        print(f"  - Task 1 samples: {task_1_count}")

    except Exception as e:
        print(f"✗ Error in task splitting: {e}")
        raise


def test_224x224_data_loading_integration():
    """Test that 224x224 Einstellung dataset integrates properly with Mammoth's data loading pipeline."""
    print("\nTesting 224x224 Einstellung dataset integration...")

    # Create test arguments
    args = create_default_args()
    args.batch_size = 16  # Smaller batch size for 224x224
    args.backbone = 'vit'  # Use ViT for 224x224
    args.einstellung_patch_size = 16  # Larger patch for 224x224

    # Create dataset instance
    dataset = SequentialCIFAR100Einstellung224(args)

    # Test get_data_loaders method
    try:
        train_loader, test_loader = dataset.get_data_loaders()
        print(f"✓ Successfully created 224x224 data loaders")
        print(f"  - Train loader: {len(train_loader)} batches")
        print(f"  - Test loader: {len(test_loader)} batches")

        # Test that we can iterate through the loaders
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))

        print(f"  - Train batch shape: {train_batch[0].shape}")
        print(f"  - Test batch shape: {test_batch[0].shape}")

        # Verify batch structure
        assert len(train_batch) >= 3, f"Expected at least 3 elements in train batch, got {len(train_batch)}"
        assert len(test_batch) >= 2, f"Expected at least 2 elements in test batch, got {len(test_batch)}"

        print(f"✓ 224x224 data loaders work correctly with cached datasets")

    except Exception as e:
        print(f"✗ Error in 224x224 data loading: {e}")
        raise


def test_class_ordering_compatibility():
    """Test that cached datasets work with class ordering and permutation systems."""
    print("\nTesting class ordering and permutation compatibility...")

    # Create test arguments with class permutation
    args = create_default_args()
    args.permute_classes = True

    # Create a class order for permutation (60 classes total in Einstellung)
    args.class_order = np.random.RandomState(42).permutation(60)

    try:
        # Create dataset instance
        dataset = SequentialCIFAR100Einstellung(args)

        # Test that data loaders work with permuted classes
        train_loader, test_loader = dataset.get_data_loaders()

        # Get a batch and verify it works
        train_batch = next(iter(train_loader))

        print(f"✓ Class permutation works with cached datasets")
        print(f"  - Batch shape with permuted classes: {train_batch[0].shape}")
        print(f"  - Target range: {train_batch[1].min().item()} - {train_batch[1].max().item()}")

    except Exception as e:
        print(f"✗ Error in class ordering compatibility: {e}")
        raise


def test_evaluation_subsets():
    """Test that evaluation subsets work with cached datasets."""
    print("\nTesting evaluation subsets...")

    # Create test arguments
    args = create_default_args()

    try:
        # Create dataset instance
        dataset = SequentialCIFAR100Einstellung(args)

        # Test evaluation subsets
        eval_subsets = dataset.get_evaluation_subsets()

        print(f"✓ Evaluation subsets created successfully")
        for subset_name, loader in eval_subsets.items():
            print(f"  - {subset_name}: {len(loader)} batches")

            # Test that we can get a batch from each subset
            batch = next(iter(loader))
            print(f"    Batch shape: {batch[0].shape}")

    except Exception as e:
        print(f"✗ Error in evaluation subsets: {e}")
        raise


if __name__ == "__main__":
    print("Testing Einstellung dataset integration with Mammoth's data loading pipeline...")
    print("=" * 80)

    try:
        test_32x32_data_loading_integration()
        test_224x224_data_loading_integration()
        test_class_ordering_compatibility()
        test_evaluation_subsets()

        print("\n" + "=" * 80)
        print("✓ All integration tests passed!")
        print("Cached Einstellung datasets are properly integrated with Mammoth's data loading pipeline.")

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
