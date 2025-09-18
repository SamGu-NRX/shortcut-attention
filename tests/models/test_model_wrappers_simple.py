"""
Simplified unit tests for Mammoth Model Wrappers

This module tests the basic functionality of the ContinualModel wrappers.
"""

import pytest
import torch
import torch.nn as nn
from argparse import Namespace
from unittest.mock import Mock, patch

# Import the model wrappers
from models.gpm_model import GPMModel
from models.dgr_model import DGRModel
from models.gpm_dgr_hybrid_model import GPMDGRHybridModel
from models.utils.continual_model import ContinualModel


class MockBackbone(nn.Module):
    """Mock backbone for testing."""

    def __init__(self):
        super().__init__()
        self.layer3 = nn.Linear(3072, 50)  # 3*32*32 = 3072
        self.classifier = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer3(x))
        return self.classifier(x)


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self):
        self.N_CLASSES = 10
        self.N_TASKS = 2
        self.N_CLASSES_PER_TASK = 5
        self.SETTING = 'class-il'

    def get_data_loaders(self):
        # Create mock data loader
        mock_data = torch.randn(4, 3, 32, 32)
        mock_labels = torch.randint(0, 5, (4,))
        mock_loader = [(mock_data, mock_labels)]
        return mock_loader, mock_loader

    def get_offsets(self, task):
        return task * 5, (task + 1) * 5

    def get_normalization_transform(self):
        return nn.Identity()


@pytest.fixture
def mock_args():
    """Create mock arguments for testing."""
    args = Namespace()

    # Common args
    args.lr = 0.01
    args.optimizer = 'sgd'
    args.optim_wd = 0.0001
    args.optim_mom = 0.9
    args.optim_nesterov = False
    args.nowand = True
    args.label_perc = 1.0
    args.num_workers = 0

    return args


@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    backbone = MockBackbone()
    loss = nn.CrossEntropyLoss()
    transform = nn.Identity()
    dataset = MockDataset()

    return backbone, loss, transform, dataset


class TestBasicFunctionality:
    """Test basic functionality of model wrappers."""

    def test_gpm_model_creation(self, mock_args, mock_components):
        """Test that GPMModel can be created."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_model.GPMAdapter'):
            model = GPMModel(backbone, loss, mock_args, transform, dataset)

        assert isinstance(model, ContinualModel)
        assert model.NAME == 'gpm_adapted'
        assert hasattr(model, 'energy_threshold')
        assert hasattr(model, 'max_collection_batches')

    def test_dgr_model_creation(self, mock_args, mock_components):
        """Test that DGRModel can be created."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.dgr_model.DGRVAE'):
            model = DGRModel(backbone, loss, mock_args, transform, dataset)

        assert isinstance(model, ContinualModel)
        assert model.NAME == 'dgr_adapted'
        assert hasattr(model, 'z_dim')
        assert hasattr(model, 'vae_lr')

    def test_hybrid_model_creation(self, mock_args, mock_components):
        """Test that GPMDGRHybridModel can be created."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_dgr_hybrid_model.GPMAdapter'), \
             patch('models.gpm_dgr_hybrid_model.DGRVAE'):
            model = GPMDGRHybridModel(backbone, loss, mock_args, transform, dataset)

        assert isinstance(model, ContinualModel)
        assert model.NAME == 'gpm_dgr_hybrid_adapted'
        assert hasattr(model, 'gpm_energy_threshold')
        assert hasattr(model, 'dgr_z_dim')

    def test_gpm_model_observe(self, mock_args, mock_components):
        """Test GPM model observe method."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_model.GPMAdapter') as mock_adapter:
            model = GPMModel(backbone, loss, mock_args, transform, dataset)
            mock_adapter_instance = mock_adapter.return_value

        # Create mock input data
        inputs = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 5, (2,))
        not_aug_inputs = inputs.clone()

        # Mock optimizer
        model.opt = Mock()

        # Call observe
        loss_value = model.observe(inputs, labels, not_aug_inputs)

        # Check that gradient projection was called
        mock_adapter_instance.project_gradients.assert_called_once()

        # Check return value
        assert isinstance(loss_value, float)

    def test_dgr_model_observe(self, mock_args, mock_components):
        """Test DGR model observe method."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.dgr_model.DGRVAE'):
            model = DGRModel(backbone, loss, mock_args, transform, dataset)

        # Create mock input data
        inputs = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 5, (2,))
        not_aug_inputs = inputs.clone()

        # Mock optimizer
        model.opt = Mock()

        # Mock replay generation (no replay)
        with patch.object(model, '_generate_replay_data', return_value=(None, None)):
            loss_value = model.observe(inputs, labels, not_aug_inputs)

        # Check return value
        assert isinstance(loss_value, float)

    def test_hybrid_model_observe(self, mock_args, mock_components):
        """Test hybrid model observe method."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_dgr_hybrid_model.GPMAdapter') as mock_gpm_adapter, \
             patch('models.gpm_dgr_hybrid_model.DGRVAE'):
            model = GPMDGRHybridModel(backbone, loss, mock_args, transform, dataset)
            mock_gpm_adapter_instance = mock_gpm_adapter.return_value

        # Create mock input data
        inputs = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 5, (2,))
        not_aug_inputs = inputs.clone()

        # Mock optimizer
        model.opt = Mock()

        # Mock replay generation
        with patch.object(model, '_generate_replay_data', return_value=(None, None)):
            loss_value = model.observe(inputs, labels, not_aug_inputs)

        # Check that GPM projection was called
        mock_gpm_adapter_instance.project_gradients.assert_called_once()

        # Check return value
        assert isinstance(loss_value, float)


class TestTaskLifecycle:
    """Test task lifecycle methods."""

    def test_gpm_model_task_lifecycle(self, mock_args, mock_components):
        """Test GPM model task lifecycle."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_model.GPMAdapter') as mock_adapter:
            model = GPMModel(backbone, loss, mock_args, transform, dataset)
            mock_adapter_instance = mock_adapter.return_value

        # Test begin_task
        model.begin_task(dataset)
        assert len(model.current_task_data) == 0

        # Add some mock data
        mock_data = torch.randn(5, 3, 32, 32)
        mock_labels = torch.randint(0, 5, (5,))
        model.current_task_data = [(x, y) for x, y in zip(mock_data, mock_labels)]

        # Test end_task
        model.end_task(dataset)
        mock_adapter_instance.update_memory.assert_called_once()

    def test_dgr_model_task_lifecycle(self, mock_args, mock_components):
        """Test DGR model task lifecycle."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.dgr_model.DGRVAE') as mock_vae:
            model = DGRModel(backbone, loss, mock_args, transform, dataset)

        # Test begin_task
        model.current_vae = Mock()
        model.begin_task(dataset)

        # Check that previous VAE was stored and new VAE created
        assert model.previous_vae is not None
        mock_vae.assert_called()
        assert len(model.current_task_buffer) == 0

        # Add mock data to buffer
        model.current_task_buffer = [torch.randn(3, 32, 32) for _ in range(5)]

        # Test end_task
        with patch.object(model, '_train_vae_on_current_task') as mock_train:
            model.end_task(dataset)
            mock_train.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
