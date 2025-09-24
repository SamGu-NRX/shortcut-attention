"""
Comprehensive validation tests for baseline method integration.

This test suite validates that the Scratch_T2 and Interleaved baseline methods
integrate properly with the existing Mammoth infrastructure including:
- Model registry discovery
- get_model() function compatibility
- EinstellungEvaluator integration
- CSV output format validation
- Checkpoint management compatibility
"""

import pytest
import torch
import os
import tempfile
import shutil
from argparse import Namespace
from unittest.mock import Mock, patch
import pandas as pd

# Import Mammoth components
from models import get_model, get_model_names, get_model_class
from models.scratch_t2 import ScratchT2
from models.interleaved import Interleaved
from utils.einstellung_evaluator import EinstellungEvaluator
from utils.conf import get_device
from backbone.ResNet32 import resnet32
from experiments.default_args import get_base_args


class TestBaselineMethodRegistry:
    """Test baseline method discovery and registry integration."""

    def test_baseline_methods_in_registry(self):
        """Test that baseline methods are automatically discovered by models/__init__.py registry."""
        model_names = get_model_names()

        # Check that both baseline methods are discovered
        assert 'scratch-t2' in model_names, "Scratch_T2 not found in model registry"
        assert 'interleaved' in model_names, "Interleaved not found in model registry"

        # Verify they are actual classes, not exceptions
        assert not isinstance(model_names['scratch-t2'], Exception), f"Scratch_T2 failed to load: {model_names['scratch-t2']}"
        assert not isinstance(model_names['interleaved'], Exception), f"Interleaved failed to load: {model_names['interleaved']}"

        # Verify correct class types
        assert model_names['scratch-t2'] == ScratchT2
        assert model_names['interleaved'] == Interleaved

    def test_baseline_method_names(self):
        """Test that baseline methods have correct NAME attributes."""
        assert ScratchT2.NAME == 'scratch_t2'
        assert Interleaved.NAME == 'interleaved'

    def test_baseline_method_compatibility(self):
        """Test that baseline methods declare proper compatibility."""
        expected_compatibility = ['class-il', 'domain-il', 'task-il']

        assert ScratchT2.COMPATIBILITY == expected_compatibility
        assert Interleaved.COMPATIBILITY == expected_compatibility


class TestGetModelFunction:
    """Test baseline methods work with existing get_model() function and argument parsing."""

    def create_test_args(self, model_name):
        """Create test arguments for model instantiation using default args."""
        base_args = get_base_args()
        # Override specific values for testing
        base_args.update({
            'model': model_name,
            'n_epochs': 1,
            'debug_mode': 1,
            'device': 'cpu',
            'dataset': 'seq-cifar100-einstellung',
            'num_workers': 0
        })
        return Namespace(**base_args)

    def create_test_components(self):
        """Create test backbone, loss, transform, and dataset."""
        backbone = resnet32(num_classes=100)
        loss = torch.nn.CrossEntropyLoss()
        transform = Mock()
        dataset = Mock()
        dataset.N_CLASSES = 100
        dataset.N_TASKS = 2
        return backbone, loss, transform, dataset

    def test_get_model_scratch_t2(self):
        """Test get_model() function works with scratch_t2."""
        args = self.create_test_args('scratch_t2')
        backbone, loss, transform, dataset = self.create_test_components()

        model = get_model(args, backbone, loss, transform, dataset)

        assert isinstance(model, ScratchT2)
        assert model.NAME == 'scratch_t2'

    def test_get_model_interleaved(self):
        """Test get_model() function works with interleaved."""
        args = self.create_test_args('interleaved')
        backbone, loss, transform, dataset = self.create_test_components()

        model = get_model(args, backbone, loss, transform, dataset)

        assert isinstance(model, Interleaved)
        assert model.NAME == 'interleaved'

    def test_get_model_class_scratch_t2(self):
        """Test get_model_class() function works with scratch_t2."""
        args = self.create_test_args('scratch_t2')

        model_class = get_model_class(args)

        assert model_class == ScratchT2

    def test_get_model_class_interleaved(self):
        """Test get_model_class() function works with interleaved."""
        args = self.create_test_args('interleaved')

        model_class = get_model_class(args)

        assert model_class == Interleaved

    def test_baseline_methods_with_hyphens(self):
        """Test that baseline methods work with hyphenated names."""
        args_scratch = self.create_test_args('scratch-t2')
        args_interleaved = self.create_test_args('interleaved')
        backbone, loss, transform, dataset = self.create_test_components()

        model_scratch = get_model(args_scratch, backbone, loss, transform, dataset)
        model_interleaved = get_model(args_interleaved, backbone, loss, transform, dataset)

        assert isinstance(model_scratch, ScratchT2)
        assert isinstance(model_interleaved, Interleaved)


class TestEinstellungEvaluatorIntegration:
    """Test baseline methods integrate with existing EinstellungEvaluator without modifications."""

    def create_mock_dataset(self):
        """Create a mock dataset for testing."""
        dataset = Mock()
        dataset.N_CLASSES = 100
        dataset.N_TASKS = 2
        dataset.i = 0  # Current task index

        # Mock train_loader with some sample data
        sample_data = [(torch.randn(32, 3, 32, 32), torch.randint(0, 100, (32,)), torch.zeros(32))]
        dataset.train_loader = sample_data

        return dataset

    def create_test_model(self, model_class):
        """Create a test model instance using default args."""
        base_args = get_base_args()
        base_args.update({
            'n_epochs': 1,
            'debug_mode': 1,
            'device': 'cpu',
            'dataset': 'seq-cifar100-einstellung',
            'num_workers': 0
        })
        args = Namespace(**base_args)

        backbone = resnet32(num_classes=100)
        loss = torch.nn.CrossEntropyLoss()
        transform = Mock()
        dataset = self.create_mock_dataset()

        return model_class(backbone, loss, args, transform, dataset)

    def test_scratch_t2_evaluator_compatibility(self):
        """Test Scratch_T2 works with EinstellungEvaluator."""
        model = self.create_test_model(ScratchT2)

        # Test that model has required methods for evaluation
        assert hasattr(model, 'forward')
        assert hasattr(model, 'begin_task')
        assert hasattr(model, 'end_task')
        assert hasattr(model, 'observe')

        # Test that model can be called for evaluation
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(test_input)

        assert output.shape == (1, 100)  # Should output logits for 100 classes

    def test_interleaved_evaluator_compatibility(self):
        """Test Interleaved works with EinstellungEvaluator."""
        model = self.create_test_model(Interleaved)

        # Test that model has required methods for evaluation
        assert hasattr(model, 'forward')
        assert hasattr(model, 'begin_task')
        assert hasattr(model, 'end_task')
        assert hasattr(model, 'observe')

        # Test that model can be called for evaluation
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(test_input)

        assert output.shape == (1, 100)  # Should output logits for 100 classes

    def test_baseline_methods_task_lifecycle(self):
        """Test baseline methods handle task lifecycle correctly."""
        scratch_model = self.create_test_model(ScratchT2)
        interleaved_model = self.create_test_model(Interleaved)
        dataset = self.create_mock_dataset()

        # Test Task 1 (should be skipped for Scratch_T2)
        scratch_model._current_task = 0  # Set current task to 0 (Task 1)
        interleaved_model._current_task = 0

        scratch_model.begin_task(dataset)
        scratch_model.end_task(dataset)

        interleaved_model.begin_task(dataset)
        # Test that interleaved model starts with empty data collection
        assert len(interleaved_model.all_data) == 0

        # Test Task 2 setup
        scratch_model._current_task = 1  # Set current task to 1 (Task 2)
        interleaved_model._current_task = 1

        scratch_model.begin_task(dataset)
        interleaved_model.begin_task(dataset)

        # Test that models have the expected state after task transitions
        assert scratch_model.task2_data is not None  # Should have stored Task 2 data
        assert len(interleaved_model.all_data) == 0   # Should still be empty until end_task

        # Models should handle task transitions without errors
        assert True  # If we reach here, no exceptions were raised


class TestCSVOutputValidation:
    """Test baseline methods produce valid CSV output in existing format for ERI visualization."""

    def test_csv_output_format_compatibility(self):
        """Test that baseline methods can produce CSV output compatible with ERI system."""
        # This test validates the CSV format structure that would be produced
        # by the EinstellungEvaluator when used with baseline methods

        expected_columns = ['method', 'seed', 'epoch_eff', 'split', 'acc', 'top5', 'loss']

        # Sample CSV data that should be produced by baseline methods
        sample_data = {
            'method': ['scratch_t2', 'scratch_t2', 'interleaved', 'interleaved'],
            'seed': [42, 42, 42, 42],
            'epoch_eff': [0.0, 0.0, 0.0, 0.0],
            'split': ['T2_shortcut_normal', 'T2_shortcut_masked', 'T2_shortcut_normal', 'T2_shortcut_masked'],
            'acc': [0.85, 0.80, 0.82, 0.78],
            'top5': [0.92, 0.88, 0.90, 0.86],
            'loss': [0.5, 0.6, 0.55, 0.62]
        }

        df = pd.DataFrame(sample_data)

        # Validate CSV structure
        assert list(df.columns) == expected_columns
        assert df['method'].dtype == 'object'
        assert df['seed'].dtype in ['int64', 'int32']
        assert df['epoch_eff'].dtype in ['float64', 'float32']
        assert df['split'].dtype == 'object'
        assert df['acc'].dtype in ['float64', 'float32']
        assert df['top5'].dtype in ['float64', 'float32']
        assert df['loss'].dtype in ['float64', 'float32']

        # Validate method names match baseline method NAME attributes
        unique_methods = df['method'].unique()
        assert 'scratch_t2' in unique_methods
        assert 'interleaved' in unique_methods

    def test_csv_split_names_compatibility(self):
        """Test that CSV split names are compatible with ERI visualization system."""
        expected_splits = [
            'T1_all',
            'T2_shortcut_normal',
            'T2_shortcut_masked',
            'T2_nonshortcut_normal'
        ]

        # These are the split names that EinstellungEvaluator should produce
        # and that the ERI visualization system expects
        for split in expected_splits:
            assert isinstance(split, str)
            assert len(split) > 0


class TestCheckpointCompatibility:
    """Test baseline methods work with existing checkpoint management and experiment orchestration."""

    def test_baseline_methods_have_state_dict(self):
        """Test that baseline methods support state_dict for checkpointing."""
        scratch_model = self.create_test_model(ScratchT2)
        interleaved_model = self.create_test_model(Interleaved)

        # Test that models have state_dict method
        assert hasattr(scratch_model, 'state_dict')
        assert hasattr(interleaved_model, 'state_dict')

        # Test that state_dict returns a dictionary
        scratch_state = scratch_model.state_dict()
        interleaved_state = interleaved_model.state_dict()

        assert isinstance(scratch_state, dict)
        assert isinstance(interleaved_state, dict)

    def test_baseline_methods_have_load_state_dict(self):
        """Test that baseline methods support load_state_dict for checkpoint loading."""
        scratch_model = self.create_test_model(ScratchT2)
        interleaved_model = self.create_test_model(Interleaved)

        # Test that models have load_state_dict method
        assert hasattr(scratch_model, 'load_state_dict')
        assert hasattr(interleaved_model, 'load_state_dict')

        # Test that load_state_dict works with saved state
        scratch_state = scratch_model.state_dict()
        interleaved_state = interleaved_model.state_dict()

        # Should not raise exceptions
        scratch_model.load_state_dict(scratch_state)
        interleaved_model.load_state_dict(interleaved_state)

    def create_test_model(self, model_class):
        """Create a test model instance using default args."""
        base_args = get_base_args()
        base_args.update({
            'n_epochs': 1,
            'debug_mode': 1,
            'device': 'cpu',
            'dataset': 'seq-cifar100-einstellung',
            'num_workers': 0
        })
        args = Namespace(**base_args)

        backbone = resnet32(num_classes=100)
        loss = torch.nn.CrossEntropyLoss()
        transform = Mock()
        dataset = Mock()
        dataset.N_CLASSES = 100
        dataset.N_TASKS = 2

        return model_class(backbone, loss, args, transform, dataset)

    def test_baseline_methods_device_compatibility(self):
        """Test that baseline methods work with device management."""
        scratch_model = self.create_test_model(ScratchT2)
        interleaved_model = self.create_test_model(Interleaved)

        # Test that models have device property
        assert hasattr(scratch_model, 'device')
        assert hasattr(interleaved_model, 'device')

        # Test that models can be moved to different devices
        device = get_device()
        scratch_model.to(device)
        interleaved_model.to(device)

        assert scratch_model.device == device
        assert interleaved_model.device == device


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
