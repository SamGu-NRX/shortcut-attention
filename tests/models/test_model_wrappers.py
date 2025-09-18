"""
Unit tests for Mammoth Model Wrappers

This module tests the ContinualModel wrappers for GPM, DGR, and hybrid methods,
ensuring proper interface compliance and method-specific functionality.
"""

import pytest
import torch
import torch.nn as nn
from argparse import Namespace
from unittest.mock import Mock, patch, MagicMock

# Import the model wrappers
from models.gpm_model import GPMModel
from models.dgr_model import DGRModel
from models.gpm_dgr_hybrid_model import GPMDGRHybridModel
from models.utils.continual_model import ContinualModel


class MockBackbone(nn.Module):
    """Mock backbone for testing."""

    def __init__(self, input_dim=3*224*224, hidden_dim=512, output_dim=100):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer3 = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.layer3(x))
        return self.classifier(x)


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self):
        self.N_CLASSES = 100
        self.N_TASKS = 10
        self.N_CLASSES_PER_TASK = 10
        self.SETTING = 'class-il'

    def get_data_loaders(self):
        # Create mock data loader
        mock_data = torch.randn(32, 3, 224, 224)
        mock_labels = torch.randint(0, 10, (32,))
        mock_loader = [(mock_data, mock_labels)]
        return mock_loader, mock_loader

    def get_offsets(self, task):
        return task * 10, (task + 1) * 10

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
    args.seed = 42
    args.nowand = True
    args.label_perc = 1.0
    args.num_workers = 0

    # GPM args
    args.gmp_energy_threshold = 0.95
    args.gmp_max_collection_batches = 200
    args.gmp_layer_names = ['layer3', 'classifier']
    args.gmp_device = 'auto'

    # DGR args
    args.dgr_z_dim = 100
    args.dgr_vae_lr = 0.001
    args.dgr_vae_fc_layers = 3
    args.dgr_vae_fc_units = 400
    args.dgr_replay_weight = 0.5
    args.dgr_vae_train_epochs = 1
    args.dgr_buffer_size = 1000

    # Hybrid args
    args.hybrid_gmp_energy_threshold = 0.95
    args.hybrid_gmp_max_collection_batches = 200
    args.hybrid_gmp_layer_names = ['layer3', 'classifier']
    args.hybrid_dgr_z_dim = 100
    args.hybrid_dgr_vae_lr = 0.001
    args.hybrid_dgr_vae_fc_layers = 3
    args.hybrid_dgr_vae_fc_units = 400
    args.hybrid_dgr_replay_weight = 0.5
    args.hybrid_dgr_vae_train_epochs = 1
    args.hybrid_dgr_buffer_size = 1000
    args.hybrid_coordination_mode = 'sequential'

    return args


@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    backbone = MockBackbone()
    loss = nn.CrossEntropyLoss()
    transform = nn.Identity()
    dataset = MockDataset()

    return backbone, loss, transform, dataset


class TestContinualModelInterface:
    """Test ContinualModel interface compliance."""

    def test_gpm_model_inheritance(self, mock_args, mock_components):
        """Test that GPMModel properly inherits from ContinualModel."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_model.GPMAdapter'):
            model = GPMModel(backbone, loss, mock_args, transform, dataset)

        assert isinstance(model, ContinualModel)
        assert model.NAME == 'gmp_adapted'
        assert 'class-il' in model.COMPATIBILITY
        assert hasattr(model, 'begin_task')
        assert hasattr(model, 'end_task')
        assert hasattr(model, 'observe')
        assert hasattr(model, 'forward')

    def test_dgr_model_inheritance(self, mock_args, mock_components):
        """Test that DGRModel properly inherits from ContinualModel."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.dgr_model.DGRVAE'):
            model = DGRModel(backbone, loss, mock_args, transform, dataset)

        assert isinstance(model, ContinualModel)
        assert model.NAME == 'dgr_adapted'
        assert 'class-il' in model.COMPATIBILITY
        assert hasattr(model, 'begin_task')
        assert hasattr(model, 'end_task')
        assert hasattr(model, 'observe')
        assert hasattr(model, 'forward')

    def test_hybrid_model_inheritance(self, mock_args, mock_components):
        """Test that GPMDGRHybridModel properly inherits from ContinualModel."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_dgr_hybrid_model.GPMAdapter'), \
             patch('models.gpm_dgr_hybrid_model.DGRVAE'):
            model = GPMDGRHybridModel(backbone, loss, mock_args, transform, dataset)

        assert isinstance(model, ContinualModel)
        assert model.NAME == 'gpm_dgr_hybrid_adapted'
        assert 'class-il' in model.COMPATIBILITY
        assert hasattr(model, 'begin_task')
        assert hasattr(model, 'end_task')
        assert hasattr(model, 'observe')
        assert hasattr(model, 'forward')


class TestGPMModel:
    """Test GPM model wrapper functionality."""

    def test_initialization(self, mock_args, mock_components):
        """Test GPM model initialization."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.get_model.GPMAdapter') as mock_adapter:
            model = GPMModel(backbone, loss, mock_args, transform, dataset)

        # Check configuration extraction
        assert model.energy_threshold == 0.95
        assert model.max_collection_batches == 200
        assert model.layer_names == ['layer3', 'classifier']

        # Check GPM adapter initialization
        mock_adapter.assert_called_once()
        assert hasattr(model, 'gmp_adapter')
        assert hasattr(model, 'current_task_data')

    def test_begin_task(self, mock_args, mock_components):
        """Test GPM begin_task functionality."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.get_model.GPMAdapter'):
            model = GPMModel(backbone, loss, mock_args, transform, dataset)

        # Add some data to task buffer
        model.current_task_data = [('dummy', 'data')]

        # Call begin_task
        model.begin_task(dataset)

        # Check that task data is cleared
        assert len(model.current_task_data) == 0

    def test_end_task(self, mock_args, mock_components):
        """Test GPM end_task functionality."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.get_model.GPMAdapter') as mock_adapter:
            model = GPMModel(backbone, loss, mock_args, transform, dataset)
            mock_adapter_instance = mock_adapter.return_value

        # Add some mock data
        mock_data = torch.randn(10, 3, 224, 224)
        mock_labels = torch.randint(0, 10, (10,))
        model.current_task_data = [(x, y) for x, y in zip(mock_data, mock_labels)]

        # Call end_task
        model.end_task(dataset)

        # Check that update_memory was called
        mock_adapter_instance.update_memory.assert_called_once()

    def test_observe(self, mock_args, mock_components):
        """Test GPM observe functionality."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.get_model.GPMAdapter') as mock_adapter:
            model = GPMModel(backbone, loss, mock_args, transform, dataset)
            mock_adapter_instance = mock_adapter.return_value

        # Create mock input data
        inputs = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 10, (4,))
        not_aug_inputs = inputs.clone()

        # Mock optimizer
        model.opt = Mock()

        # Call observe
        loss_value = model.observe(inputs, labels, not_aug_inputs)

        # Check that gradient projection was called
        mock_adapter_instance.project_gradients.assert_called_once()

        # Check that optimizer was used
        model.opt.zero_grad.assert_called_once()
        model.opt.step.assert_called_once()

        # Check return value
        assert isinstance(loss_value, float)

    def test_parser(self):
        """Test GPM parser functionality."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser = GPMModel.get_parser(parser)

        # Check that GPM arguments were added
        args = parser.parse_args([
            '--gmp_energy_threshold', '0.9',
            '--gmp_max_collection_batches', '100'
        ])

        assert args.gmp_energy_threshold == 0.9
        assert args.gmp_max_collection_batches == 100


class TestDGRModel:
    """Test DGR model wrapper functionality."""

    def test_initialization(self, mock_args, mock_components):
        """Test DGR model initialization."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.dgr_model.DGRVAE'):
            model = DGRModel(backbone, loss, mock_args, transform, dataset)

        # Check configuration extraction
        assert model.z_dim == 100
        assert model.vae_lr == 0.001
        assert model.replay_weight == 0.5
        assert model.image_shape == (3, 224, 224)

        # Check initialization
        assert hasattr(model, 'current_vae')
        assert hasattr(model, 'previous_vae')
        assert hasattr(model, 'current_task_buffer')

    def test_begin_task(self, mock_args, mock_components):
        """Test DGR begin_task functionality."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.dgr_model.DGRVAE') as mock_vae:
            model = DGRModel(backbone, loss, mock_args, transform, dataset)

        # Set up previous VAE
        model.current_vae = Mock()

        # Call begin_task
        model.begin_task(dataset)

        # Check that previous VAE was stored and new VAE created
        assert model.previous_vae is not None
        mock_vae.assert_called()  # New VAE should be created
        assert len(model.current_task_buffer) == 0

    def test_end_task(self, mock_args, mock_components):
        """Test DGR end_task functionality."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.dgr_model.DGRVAE'):
            model = DGRModel(backbone, loss, mock_args, transform, dataset)

        # Add mock data to buffer
        model.current_task_buffer = [torch.randn(3, 224, 224) for _ in range(10)]

        # Mock the VAE training method
        with patch.object(model, '_train_vae_on_current_task') as mock_train:
            model.end_task(dataset)
            mock_train.assert_called_once()

    def test_observe_without_replay(self, mock_args, mock_components):
        """Test DGR observe without replay data."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.dgr_model.DGRVAE'):
            model = DGRModel(backbone, loss, mock_args, transform, dataset)

        # Create mock input data
        inputs = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 10, (4,))
        not_aug_inputs = inputs.clone()

        # Mock optimizer
        model.opt = Mock()

        # Mock replay generation (no replay)
        with patch.object(model, '_generate_replay_data', return_value=(None, None)):
            loss_value = model.observe(inputs, labels, not_aug_inputs)

        # Check that optimizer was used
        model.opt.zero_grad.assert_called_once()
        model.opt.step.assert_called_once()

        # Check return value
        assert isinstance(loss_value, float)

    def test_observe_with_replay(self, mock_args, mock_components):
        """Test DGR observe with replay data."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.dgr_model.DGRVAE'):
            model = DGRModel(backbone, loss, mock_args, transform, dataset)

        # Create mock input data
        inputs = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 10, (4,))
        not_aug_inputs = inputs.clone()

        # Mock optimizer
        model.opt = Mock()

        # Mock replay generation (with replay)
        replay_inputs = torch.randn(4, 3, 224, 224)
        replay_labels = torch.randint(0, 5, (4,))
        with patch.object(model, '_generate_replay_data', return_value=(replay_inputs, replay_labels)):
            loss_value = model.observe(inputs, labels, not_aug_inputs)

        # Check that optimizer was used
        model.opt.zero_grad.assert_called_once()
        model.opt.step.assert_called_once()

        # Check return value
        assert isinstance(loss_value, float)


class TestGPMDGRHybridModel:
    """Test hybrid model wrapper functionality."""

    def test_initialization(self, mock_args, mock_components):
        """Test hybrid model initialization."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_dgr_hybrid_model.GPMAdapter'), \
             patch('models.gpm_dgr_hybrid_model.DGRVAE'):
            model = GPMDGRHybridModel(backbone, loss, mock_args, transform, dataset)

        # Check GPM configuration
        assert model.gmp_energy_threshold == 0.95
        assert model.gmp_max_collection_batches == 200

        # Check DGR configuration
        assert model.dgr_z_dim == 100
        assert model.dgr_vae_lr == 0.001
        assert model.dgr_replay_weight == 0.5

        # Check coordination
        assert model.coordination_mode == 'sequential'

        # Check initialization
        assert hasattr(model, 'gmp_adapter')
        assert hasattr(model, 'current_vae')
        assert hasattr(model, 'previous_vae')
        assert hasattr(model, 'current_task_buffer')
        assert hasattr(model, 'current_task_data')

    def test_begin_task(self, mock_args, mock_components):
        """Test hybrid begin_task functionality."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_dgr_hybrid_model.GPMAdapter'), \
             patch('models.gpm_dgr_hybrid_model.DGRVAE') as mock_vae:
            model = GPMDGRHybridModel(backbone, loss, mock_args, transform, dataset)

        # Set up previous state
        model.current_vae = Mock()
        model.current_task_data = ['dummy_data']
        model.current_task_buffer = ['dummy_buffer']

        # Call begin_task
        model.begin_task(dataset)

        # Check GPM preparation
        assert len(model.current_task_data) == 0

        # Check DGR preparation
        assert model.previous_vae is not None
        mock_vae.assert_called()  # New VAE should be created
        assert len(model.current_task_buffer) == 0

    def test_end_task_sequential(self, mock_args, mock_components):
        """Test hybrid end_task with sequential coordination."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_dgr_hybrid_model.GPMAdapter'), \
             patch('models.gpm_dgr_hybrid_model.DGRVAE'):
            model = GPMDGRHybridModel(backbone, loss, mock_args, transform, dataset)

        # Set coordination mode
        model.coordination_mode = 'sequential'

        # Mock the update methods
        with patch.object(model, '_update_gmp_memory') as mock_gmp, \
             patch.object(model, '_train_dgr_vae') as mock_dgr:
            model.end_task(dataset)

            # Check that both methods were called
            mock_gmp.assert_called_once()
            mock_dgr.assert_called_once()

    def test_observe(self, mock_args, mock_components):
        """Test hybrid observe functionality."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_dgr_hybrid_model.GPMAdapter') as mock_gmp_adapter, \
             patch('models.gpm_dgr_hybrid_model.DGRVAE'):
            model = GPMDGRHybridModel(backbone, loss, mock_args, transform, dataset)
            mock_gmp_adapter_instance = mock_gmp_adapter.return_value

        # Create mock input data
        inputs = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 10, (4,))
        not_aug_inputs = inputs.clone()

        # Mock optimizer
        model.opt = Mock()

        # Mock replay generation
        with patch.object(model, '_generate_replay_data', return_value=(None, None)):
            loss_value = model.observe(inputs, labels, not_aug_inputs)

        # Check that GPM projection was called
        mock_gmp_adapter_instance.project_gradients.assert_called_once()

        # Check that optimizer was used
        model.opt.zero_grad.assert_called_once()
        model.opt.step.assert_called_once()

        # Check return value
        assert isinstance(loss_value, float)


class TestConfigurationParsing:
    """Test configuration parameter parsing."""

    def test_gmp_default_parameters(self, mock_components):
        """Test GPM default parameter handling."""
        backbone, loss, transform, dataset = mock_components

        # Create minimal args
        args = Namespace()
        args.lr = 0.01
        args.optimizer = 'sgd'
        args.optim_wd = 0.0001
        args.optim_mom = 0.9
        args.optim_nesterov = False
        args.nowand = True
        args.label_perc = 1.0
        args.num_workers = 0

        with patch('models.get_model.GPMAdapter'):
            model = GPMModel(backbone, loss, args, transform, dataset)

        # Check default values
        assert model.energy_threshold == 0.95
        assert model.max_collection_batches == 200
        assert model.layer_names == ['backbone.layer3', 'classifier']

    def test_dgr_default_parameters(self, mock_components):
        """Test DGR default parameter handling."""
        backbone, loss, transform, dataset = mock_components

        # Create minimal args
        args = Namespace()
        args.lr = 0.01
        args.optimizer = 'sgd'
        args.optim_wd = 0.0001
        args.optim_mom = 0.9
        args.optim_nesterov = False
        args.nowand = True
        args.label_perc = 1.0
        args.num_workers = 0

        with patch('models.dgr_model.DGRVAE'):
            model = DGRModel(backbone, loss, args, transform, dataset)

        # Check default values
        assert model.z_dim == 100
        assert model.vae_lr == 0.001
        assert model.replay_weight == 0.5
        assert model.vae_train_epochs == 1


class TestIntegrationCompliance:
    """Test integration with Mammoth training pipeline."""

    def test_get_model_device_handling(self, mock_args, mock_components):
        """Test GPM model device handling."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.get_model.GPMAdapter'):
            model = GPMModel(backbone, loss, mock_args, transform, dataset)

        # Test device movement
        device = torch.device('cpu')
        model = model.to(device)
        assert model.device == device

    def test_dgr_model_device_handling(self, mock_args, mock_components):
        """Test DGR model device handling."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.dgr_model.DGRVAE'):
            model = DGRModel(backbone, loss, mock_args, transform, dataset)

        # Test device movement
        device = torch.device('cpu')
        model = model.to(device)
        assert model.device == device

    def test_hybrid_model_device_handling(self, mock_args, mock_components):
        """Test hybrid model device handling."""
        backbone, loss, transform, dataset = mock_components

        with patch('models.gpm_dgr_hybrid_model.GPMAdapter'), \
             patch('models.gpm_dgr_hybrid_model.DGRVAE'):
            model = GPMDGRHybridModel(backbone, loss, mock_args, transform, dataset)

        # Test device movement
        device = torch.device('cpu')
        model = model.to(device)
        assert model.device == device


if __name__ == '__main__':
    pytest.main([__file__])
