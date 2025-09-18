# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for GPM (Gradient Projection Memory) integration with Mammoth framework.

These tests verify that the GPM adapter correctly integrates with Mammoth's
ContinualModel interface and maintains the original GPM functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from torchvision import transforms

from models.gpm import GPMAdapter, GPMMammoth
from utils.args import ArgumentParser


class SimpleTestModel(nn.Module):
    """Simple model for testing GPM functionality."""

    def __init__(self, input_size=32*32*3, hidden_size=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        self.backbone = self.net[:-1]  # All layers except the last

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.net(x)


class TestGPMAdapter:
    """Test cases for GPMAdapter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.model = SimpleTestModel()
        self.adapter = GPMAdapter(
            model=self.model,
            energy_threshold=0.95,
            max_collection_batches=5,
            device=self.device
        )

    def test_initialization(self):
        """Test GPMAdapter initialization."""
        assert self.adapter.model is self.model
        assert self.adapter.device == self.device
        assert self.adapter.energy_threshold == 0.95
        assert self.adapter.max_collection_batches == 5
        assert isinstance(self.adapter.feature_list, list)
        assert len(self.adapter.feature_list) == 0
        assert isinstance(self.adapter.projection_matrices, list)
        assert len(self.adapter.projection_matrices) == 0

    def test_model_structure_analysis(self):
        """Test model structure analysis."""
        layer_info = self.adapter.layer_info
        assert isinstance(layer_info, list)
        assert len(layer_info) > 0

        # Check that layer info contains expected fields
        for info in layer_info:
            assert 'name' in info
            assert 'module' in info
            assert 'type' in info
            assert info['type'] in ['conv', 'linear']

    def test_activation_collection(self):
        """Test activation collection functionality."""
        # Create synthetic data
        batch_size = 16
        input_size = 32 * 32 * 3
        inputs = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))

        dataset = torch.utils.data.TensorDataset(inputs, labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

        # Collect activations
        mat_list = self.adapter.collect_activations(data_loader)

        # Verify results
        assert isinstance(mat_list, list)
        assert len(mat_list) > 0

        for mat in mat_list:
            assert isinstance(mat, np.ndarray)
            assert mat.ndim == 2
            assert mat.shape[0] > 0
            assert mat.shape[1] > 0

    def test_memory_update_first_task(self):
        """Test memory update for first task."""
        # Create synthetic activation matrices
        mat_list = [
            np.random.randn(128, 100),
            np.random.randn(64, 100)
        ]

        # Update memory
        self.adapter.update_memory(mat_list, task_id=0)

        # Verify results
        assert len(self.adapter.feature_list) == len(mat_list)
        assert len(self.adapter.projection_matrices) == len(mat_list)

        for i, feature_matrix in enumerate(self.adapter.feature_list):
            assert isinstance(feature_matrix, np.ndarray)
            assert feature_matrix.ndim == 2
            assert feature_matrix.shape[0] == mat_list[i].shape[0]
            assert feature_matrix.shape[1] <= mat_list[i].shape[1]

    def test_memory_update_subsequent_task(self):
        """Test memory update for subsequent tasks."""
        # Initialize with first task
        mat_list_1 = [
            np.random.randn(128, 100),
            np.random.randn(64, 100)
        ]
        self.adapter.update_memory(mat_list_1, task_id=0)

        # Store original basis sizes
        original_sizes = [f.shape[1] for f in self.adapter.feature_list]

        # Update with second task
        mat_list_2 = [
            np.random.randn(128, 100),
            np.random.randn(64, 100)
        ]
        self.adapter.update_memory(mat_list_2, task_id=1)

        # Verify that bases were updated
        assert len(self.adapter.feature_list) == len(mat_list_2)

        for i, feature_matrix in enumerate(self.adapter.feature_list):
            assert isinstance(feature_matrix, np.ndarray)
            assert feature_matrix.ndim == 2
            assert feature_matrix.shape[0] == mat_list_2[i].shape[0]
            # Basis size should be >= original size (may grow)
            assert feature_matrix.shape[1] >= original_sizes[i]

    def test_gradient_projection(self):
        """Test gradient projection functionality."""
        # Initialize with some basis
        mat_list = [
            np.random.randn(128, 100),
            np.random.randn(64, 100)
        ]
        self.adapter.update_memory(mat_list, task_id=0)

        # Create synthetic gradients
        for param in self.model.parameters():
            param.grad = torch.randn_like(param)

        # Store original gradients
        original_grads = [param.grad.clone() for param in self.model.parameters()]

        # Apply projection
        self.adapter.project_gradients()

        # Verify that gradients were modified
        projected_grads = [param.grad for param in self.model.parameters()]

        # At least some gradients should be different after projection
        any_different = False
        for orig, proj in zip(original_grads, projected_grads):
            if not torch.allclose(orig, proj, atol=1e-6):
                any_different = True
                break

        assert any_different, "Gradients should be modified by projection"

    def test_svd_computation_stability(self):
        """Test SVD computation with edge cases."""
        # Test with rank-deficient matrix
        rank_deficient = np.random.randn(100, 50)
        rank_deficient = np.dot(rank_deficient, rank_deficient.T)  # Make it rank 50

        mat_list = [rank_deficient]

        # Should not raise an exception
        self.adapter.update_memory(mat_list, task_id=0)

        assert len(self.adapter.feature_list) == 1
        assert self.adapter.feature_list[0].shape[1] <= 50  # Rank should be <= 50


class TestGPMMammoth:
    """Test cases for GPMMammoth class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock arguments
        self.args = Mock()
        self.args.gpm_energy_threshold = 0.95
        self.args.gpm_max_collection_batches = 5
        self.args.lr = 0.01
        self.args.optimizer = 'sgd'
        self.args.optim_wd = 0.0
        self.args.optim_mom = 0.9
        self.args.optim_nesterov = False
        self.args.dataset = 'seq-cifar100'  # Add required dataset attribute

        # Create simple backbone and loss
        self.backbone = SimpleTestModel()
        self.loss = nn.CrossEntropyLoss()
        self.transform = None

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.N_CLASSES = 10
        mock_dataset.N_TASKS = 2
        mock_dataset.N_CLASSES_PER_TASK = 5
        mock_dataset.SETTING = 'class-il'
        mock_dataset.get_normalization_transform.return_value = transforms.Normalize((0.5,), (0.5,))

        # Create GPM model with mock dataset
        self.model = GPMMammoth(
            backbone=self.backbone,
            loss=self.loss,
            args=self.args,
            transform=self.transform,
            dataset=mock_dataset
        )

    def test_initialization(self):
        """Test GPMMammoth initialization."""
        assert self.model.NAME == 'gpm'
        assert 'class-il' in self.model.COMPATIBILITY
        assert hasattr(self.model, 'gpm_adapter')
        assert isinstance(self.model.gpm_adapter, GPMAdapter)
        assert self.model.gpm_adapter.energy_threshold == 0.95

    def test_parser_extension(self):
        """Test that parser is correctly extended with GPM arguments."""
        parser = ArgumentParser()
        extended_parser = GPMMammoth.get_parser(parser)

        # Check that GPM-specific arguments were added
        args = extended_parser.parse_args([
            '--gpm_energy_threshold', '0.9',
            '--gpm_max_collection_batches', '100'
        ])

        assert args.gpm_energy_threshold == 0.9
        assert args.gpm_max_collection_batches == 100

    def test_observe_functionality(self):
        """Test the observe method."""
        # Create synthetic data
        batch_size = 8
        inputs = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        not_aug_inputs = inputs.clone()

        # Call observe
        loss_value = self.model.observe(inputs, labels, not_aug_inputs)

        # Verify results
        assert isinstance(loss_value, float)
        assert loss_value >= 0.0
        assert len(self.model.current_task_data) > 0

    def test_task_lifecycle(self):
        """Test begin_task and end_task methods."""
        # Mock dataset
        mock_dataset = Mock()

        # Test begin_task
        self.model.begin_task(mock_dataset)
        assert len(self.model.current_task_data) == 0

        # Simulate some training
        batch_size = 8
        inputs = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        not_aug_inputs = inputs.clone()

        self.model.observe(inputs, labels, not_aug_inputs)
        assert len(self.model.current_task_data) > 0

        # Test end_task
        with patch.object(self.model.gpm_adapter, 'collect_activations') as mock_collect:
            with patch.object(self.model.gpm_adapter, 'update_memory') as mock_update:
                mock_collect.return_value = [np.random.randn(128, 100)]

                self.model.end_task(mock_dataset)

                # Verify that memory update was called
                mock_collect.assert_called_once()
                mock_update.assert_called_once()

        # Data should be cleared after end_task
        assert len(self.model.current_task_data) == 0

    def test_gradient_projection_integration(self):
        """Test that gradient projection is applied during training."""
        # Initialize with some basis (simulate previous task)
        mat_list = [np.random.randn(128, 100)]
        self.model.gpm_adapter.update_memory(mat_list, task_id=0)

        # Create synthetic data
        batch_size = 8
        inputs = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        not_aug_inputs = inputs.clone()

        # Store original parameters
        original_params = [param.clone() for param in self.model.parameters()]

        # Perform training step
        loss_value = self.model.observe(inputs, labels, not_aug_inputs)

        # Verify that parameters were updated
        updated_params = [param for param in self.model.parameters()]

        any_updated = False
        for orig, updated in zip(original_params, updated_params):
            if not torch.allclose(orig, updated, atol=1e-6):
                any_updated = True
                break

        assert any_updated, "Parameters should be updated during training"
        assert isinstance(loss_value, float)


class TestGPMIntegrationWithEinstellung:
    """Integration tests with Einstellung dataset and evaluator."""

    @pytest.mark.slow
    def test_einstellung_compatibility(self):
        """Test GPM compatibility with Einstellung dataset."""
        # This test would require the full Mammoth setup
        # For now, we'll create a minimal test

        # Mock the necessary components
        args = Mock()
        args.gpm_energy_threshold = 0.95
        args.gpm_max_collection_batches = 10
        args.dataset = 'seq-cifar10'
        args.lr = 0.01
        args.optimizer = 'sgd'
        args.optim_wd = 0.0
        args.optim_mom = 0.9
        args.optim_nesterov = False

        # Create a simple model
        backbone = SimpleTestModel()
        loss = nn.CrossEntropyLoss()

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.N_CLASSES = 10
        mock_dataset.N_TASKS = 2
        mock_dataset.N_CLASSES_PER_TASK = 5
        mock_dataset.SETTING = 'class-il'
        mock_dataset.get_normalization_transform.return_value = transforms.Normalize((0.5,), (0.5,))
        mock_dataset.get_offsets.return_value = (0, 5)  # (n_past_classes, n_seen_classes)

        # Create model (will use proper device detection from Mammoth)
        model = GPMMammoth(backbone, loss, args, None, dataset=mock_dataset)

        # Test basic functionality
        inputs = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))

        # Test that observe works
        loss_val = model.observe(inputs, labels, inputs)
        assert isinstance(loss_val, float)

        # Test that GPM adapter is properly initialized
        assert hasattr(model, 'gpm_adapter')
        assert model.gpm_adapter.energy_threshold == 0.95

        # Test that we can manually update memory (simulating end_task)
        mat_list = [np.random.randn(128, 100)]
        model.gpm_adapter.update_memory(mat_list, task_id=0)

        # Verify that GPM memory was built
        assert len(model.gpm_adapter.feature_list) > 0
        assert len(model.gpm_adapter.projection_matrices) > 0


class TestGPMPerformance:
    """Performance tests for GPM implementation."""

    def test_memory_usage(self):
        """Test that GPM doesn't consume excessive memory."""
        model = SimpleTestModel()
        adapter = GPMAdapter(model, max_collection_batches=10)

        # Simulate multiple tasks
        for task_id in range(3):
            mat_list = [np.random.randn(128, 100) for _ in range(2)]
            adapter.update_memory(mat_list, task_id)

        # Check that memory usage is reasonable
        total_basis_size = sum(f.size for f in adapter.feature_list)
        assert total_basis_size < 1e6  # Less than 1M parameters in bases

    def test_training_time_overhead(self):
        """Test that GPM doesn't add excessive training time."""
        import time

        # Create models
        args = Mock()
        args.gpm_energy_threshold = 0.95
        args.gpm_max_collection_batches = 5
        args.lr = 0.01
        args.optimizer = 'sgd'
        args.optim_wd = 0.0
        args.optim_mom = 0.9
        args.optim_nesterov = False
        args.dataset = 'seq-cifar100'

        backbone = SimpleTestModel()
        loss = nn.CrossEntropyLoss()

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.N_CLASSES = 10
        mock_dataset.N_TASKS = 2
        mock_dataset.N_CLASSES_PER_TASK = 5
        mock_dataset.SETTING = 'class-il'
        mock_dataset.get_normalization_transform.return_value = transforms.Normalize((0.5,), (0.5,))

        gpm_model = GPMMammoth(backbone, loss, args, None, dataset=mock_dataset)

        # Warm up
        inputs = torch.randn(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))

        # Time GPM training
        start_time = time.time()
        for _ in range(10):
            gpm_model.observe(inputs, labels, inputs)
        gpm_time = time.time() - start_time

        # The overhead should be reasonable (this is a basic check)
        assert gpm_time < 10.0  # Should complete in less than 10 seconds


if __name__ == '__main__':
    pytest.main([__file__])
