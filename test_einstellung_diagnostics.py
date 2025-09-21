#!/usr/bin/env python3
"""
Diagnostic script to test Einstellung integration step by step.
This will help identify the exact issues before running the full experiment.
"""

import sys
import os
import logging
from pathlib import Path
from argparse import Namespace

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_csv_log_argument():
    """Test if --csv_log argument exists in Mammoth."""
    print("1. Testing --csv_log argument...")
    try:
        from utils.args import add_management_args, add_experiment_args
        import argparse

        parser = argparse.ArgumentParser()
        add_management_args(parser)
        add_experiment_args(parser)

        # Check if csv_log exists in the parser
        action_names = [action.dest for action in parser._actions]
        if 'csv_log' in action_names:
            print("   ✓ csv_log argument exists")
            return True
        else:
            print("   ✗ csv_log argument NOT found")
            print(f"   Available args: {sorted(action_names)}")
            return False
    except Exception as e:
        print(f"   ✗ Error testing csv_log: {e}")
        return False

def test_dataset_discovery():
    """Test if Einstellung datasets are properly discovered."""
    print("\n2. Testing dataset discovery...")
    try:
        from datasets import get_dataset_names

        names = get_dataset_names(names_only=True)
        einstellung_datasets = [name for name in names if 'einstellung' in name]

        print(f"   Found {len(einstellung_datasets)} Einstellung datasets:")
        for dataset in einstellung_datasets:
            print(f"     - {dataset}")

        if 'seq-cifar100-einstellung' in einstellung_datasets:
            print("   ✓ seq-cifar100-einstellung found")
        else:
            print("   ✗ seq-cifar100-einstellung NOT found")

        if 'seq-cifar100-einstellung-224' in einstellung_datasets:
            print("   ✓ seq-cifar100-einstellung-224 found")
        else:
            print("   ✗ seq-cifar100-einstellung-224 NOT found")

        return len(einstellung_datasets) > 0

    except Exception as e:
        print(f"   ✗ Error testing dataset discovery: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test if Einstellung datasets can be loaded."""
    print("\n3. Testing dataset loading...")
    try:
        from datasets import get_dataset

        # Test args for the basic dataset - include all required ContinualDataset args
        args = Namespace(
            dataset='seq-cifar100-einstellung',
            seed=42,
            base_path='./data/',
            batch_size=32,
            joint=False,  # Required by ContinualDataset
            custom_task_order=None,  # Required by ContinualDataset
            custom_class_order=None,  # Required by ContinualDataset
            permute_classes=False,   # Required by ContinualDataset
            label_perc=1.0,         # Required by ContinualDataset
            label_perc_by_class=1.0, # Required by ContinualDataset
            validation=None,         # Required by ContinualDataset
            validation_mode='current', # Required by ContinualDataset
            start_from=None,         # Required by ContinualDataset
            stop_after=None,         # Required by ContinualDataset
            enable_other_metrics=False, # Required by ContinualDataset
            eval_future=False,       # Required by ContinualDataset
            noise_rate=0.0,          # Required by ContinualDataset
            einstellung_patch_size=4,
            einstellung_patch_color=[255, 0, 255],
            einstellung_apply_shortcut=True,
            einstellung_mask_shortcut=False,
            einstellung_evaluation_subsets=True,
            einstellung_extract_attention=True,
            einstellung_adaptation_threshold=0.8
        )

        print("   Loading seq-cifar100-einstellung...")
        dataset = get_dataset(args)
        print(f"   ✓ Dataset loaded: {dataset.NAME}")
        print(f"     - N_TASKS: {dataset.N_TASKS}")
        print(f"     - N_CLASSES: {dataset.N_CLASSES}")

        # Test basic methods
        class_names = dataset.get_class_names()
        print(f"     - Class names count: {len(class_names)}")

        return True

    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_einstellung_arguments():
    """Test if Einstellung arguments are properly registered."""
    print("\n4. Testing Einstellung arguments...")
    try:
        from utils.args import add_management_args, add_experiment_args
        import argparse

        parser = argparse.ArgumentParser()
        add_management_args(parser)
        add_experiment_args(parser)

        # Check Einstellung arguments
        action_names = [action.dest for action in parser._actions]
        einstellung_args = [arg for arg in action_names if 'einstellung' in arg]

        print(f"   Found {len(einstellung_args)} Einstellung arguments:")
        for arg in einstellung_args:
            print(f"     - {arg}")

        expected_args = [
            'einstellung_patch_size',
            'einstellung_patch_color',
            'einstellung_adaptation_threshold',
            'einstellung_apply_shortcut',
            'einstellung_mask_shortcut',
            'einstellung_evaluation_subsets',
            'einstellung_extract_attention'
        ]

        missing_args = [arg for arg in expected_args if arg not in einstellung_args]
        if missing_args:
            print(f"   ✗ Missing arguments: {missing_args}")
            return False
        else:
            print("   ✓ All Einstellung arguments found")
            return True

    except Exception as e:
        print(f"   ✗ Error testing arguments: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_command():
    """Test minimal command construction."""
    print("\n5. Testing minimal command construction...")
    try:
        # Build a minimal command without csv_log
        cmd_args = [
            '--dataset', 'seq-cifar100-einstellung',
            '--model', 'derpp',
            '--backbone', 'resnet18',
            '--n_epochs', '2',  # Very short for testing
            '--batch_size', '32',
            '--lr', '0.01',
            '--seed', '42',
            '--buffer_size', '50',  # Small buffer for testing
            '--alpha', '0.1',
            '--beta', '0.5',
            '--einstellung_patch_size', '4',
            '--einstellung_patch_color', '255', '0', '255',
            '--einstellung_adaptation_threshold', '0.8',
            '--einstellung_apply_shortcut', '1',
            '--einstellung_evaluation_subsets', '1',
            '--einstellung_extract_attention', '1'
        ]

        print(f"   Command length: {len(cmd_args)} arguments")
        print(f"   Command: {' '.join(cmd_args)}")

                # Test parsing without execution
        from utils.args import add_management_args, add_experiment_args, add_rehearsal_args
        from models import get_model_class
        import argparse

        parser = argparse.ArgumentParser()
        add_management_args(parser)
        add_experiment_args(parser)
        add_rehearsal_args(parser)  # DerPP needs rehearsal arguments

        # Add dataset and model arguments that are required
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--backbone', type=str, default='resnet18')

        # Add DerPP specific arguments
        parser.add_argument('--alpha', type=float, required=True)
        parser.add_argument('--beta', type=float, required=True)

        args = parser.parse_args(cmd_args)
        print("   ✓ Command parsed successfully")
        print(f"     - Dataset: {args.dataset}")
        print(f"     - Model: {args.model}")
        print(f"     - Einstellung patch size: {args.einstellung_patch_size}")

        return True

    except Exception as e:
        print(f"   ✗ Error testing command: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_continual_dataset_structure():
    """Test the ContinualDataset parent class structure."""
    print("\n6. Testing ContinualDataset parent class structure...")
    try:
        from datasets.utils.continual_dataset import ContinualDataset
        import inspect

        # Inspect the ContinualDataset class
        print("   ContinualDataset class inspection:")

        # Check if current_task exists and what type it is
        if hasattr(ContinualDataset, 'current_task'):
            current_task_attr = getattr(ContinualDataset, 'current_task')
            print(f"     - current_task exists: {type(current_task_attr)}")

            # Check if it's a property
            if isinstance(current_task_attr, property):
                print("     - current_task is a property")
                print(f"       - Has getter: {current_task_attr.fget is not None}")
                print(f"       - Has setter: {current_task_attr.fset is not None}")
                print(f"       - Has deleter: {current_task_attr.fdel is not None}")
            else:
                print(f"     - current_task is: {type(current_task_attr)}")
        else:
            print("     - current_task does not exist in parent class")

        # Check all properties of the class
        properties = [name for name, obj in inspect.getmembers(ContinualDataset)
                     if isinstance(obj, property)]
        print(f"     - Properties in ContinualDataset: {properties}")

        # Check required attributes
        required_attrs = ['N_CLASSES_PER_TASK', 'N_TASKS', 'N_CLASSES', 'NAME', 'SETTING']
        print("     - Required class attributes:")
        for attr in required_attrs:
            if hasattr(ContinualDataset, attr):
                print(f"       - {attr}: defined in parent")
            else:
                print(f"       - {attr}: NOT defined in parent")

        # Test minimal initialization
        print("   Testing minimal ContinualDataset initialization...")

        # Create minimal args
        from argparse import Namespace
        minimal_args = Namespace(
            joint=False,
            custom_task_order=None,
            custom_class_order=None,
            permute_classes=False,
            label_perc=1.0,
            label_perc_by_class=1.0,
            validation=None,
            validation_mode='current',
            start_from=None,
            stop_after=None,
            enable_other_metrics=False,
            eval_future=False,
            noise_rate=0.0,
            batch_size=32,
            base_path='./data/'
        )

        # Try to create a test class that inherits from ContinualDataset
        class TestDataset(ContinualDataset):
            NAME = 'test-dataset'
            SETTING = 'class-il'
            N_CLASSES_PER_TASK = 10
            N_TASKS = 2
            N_CLASSES = 20

            def __init__(self, args):
                print("       - Before super().__init__()")
                super().__init__(args)
                print("       - After super().__init__()")
                print(f"       - current_task after init: {getattr(self, 'current_task', 'NOT SET')}")

        test_dataset = TestDataset(minimal_args)
        print("     ✓ Minimal ContinualDataset initialization successful")

        return True

    except Exception as e:
        print(f"   ✗ Error testing ContinualDataset structure: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_einstellung_dataset_fix():
    """Test different approaches to fix the current_task issue."""
    print("\n7. Testing Einstellung dataset fixes...")
    try:
        from datasets.utils.continual_dataset import ContinualDataset
        from argparse import Namespace

        # Test args
        args = Namespace(
            dataset='seq-cifar100-einstellung',
            seed=42,
            base_path='./data/',
            batch_size=32,
            joint=False,
            custom_task_order=None,
            custom_class_order=None,
            permute_classes=False,
            label_perc=1.0,
            label_perc_by_class=1.0,
            validation=None,
            validation_mode='current',
            start_from=None,
            stop_after=None,
            enable_other_metrics=False,
            eval_future=False,
            noise_rate=0.0,
            einstellung_patch_size=4,
            einstellung_patch_color=[255, 0, 255],
            einstellung_apply_shortcut=True,
            einstellung_mask_shortcut=False,
            einstellung_evaluation_subsets=True,
            einstellung_extract_attention=True,
            einstellung_adaptation_threshold=0.8
        )

        print("   Testing different initialization approaches...")

        # Approach 1: Don't set current_task at all
        print("     Approach 1: Remove current_task assignment")
        class TestEinstellungDataset1(ContinualDataset):
            NAME = 'test-einstellung-1'
            SETTING = 'class-il'
            N_CLASSES_PER_TASK = 30
            N_TASKS = 2
            N_CLASSES = 60

            def __init__(self, args):
                super().__init__(args)
                # Don't set current_task
                print(f"       - current_task is: {getattr(self, 'current_task', 'NOT SET')}")

        test1 = TestEinstellungDataset1(args)
        print("     ✓ Approach 1 successful")

        # Approach 2: Set current_task before super().__init__()
        print("     Approach 2: Set current_task before super()")
        class TestEinstellungDataset2(ContinualDataset):
            NAME = 'test-einstellung-2'
            SETTING = 'class-il'
            N_CLASSES_PER_TASK = 30
            N_TASKS = 2
            N_CLASSES = 60

            def __init__(self, args):
                # Try to set before super()
                try:
                    self.current_task = 0
                    print("       - Set current_task before super()")
                except Exception as e:
                    print(f"       - Failed to set current_task before super(): {e}")

                super().__init__(args)
                print(f"       - current_task after super(): {getattr(self, 'current_task', 'NOT SET')}")

        test2 = TestEinstellungDataset2(args)
        print("     ✓ Approach 2 completed")

        return True

    except Exception as e:
        print(f"   ✗ Error testing dataset fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests."""
    print("Running Einstellung Integration Diagnostics")
    print("=" * 50)

    results = {}
    results['csv_log'] = test_csv_log_argument()
    results['discovery'] = test_dataset_discovery()
    results['loading'] = test_dataset_loading()
    results['arguments'] = test_einstellung_arguments()
    results['command'] = test_minimal_command()
    results['continual_structure'] = test_continual_dataset_structure()
    results['dataset_fixes'] = test_einstellung_dataset_fix()

    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<20}: {status}")

    total_passed = sum(results.values())
    print(f"\nPassed: {total_passed}/{len(results)} tests")

    if total_passed == len(results):
        print("\n🎉 All tests passed! The integration should work.")
    else:
        print(f"\n⚠️  {len(results) - total_passed} tests failed. Fix these issues first.")

if __name__ == '__main__':
    main()
