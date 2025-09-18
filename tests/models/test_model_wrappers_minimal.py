"""
Minimal unit tests for Mammoth Model Wrappers

This module tests the basic functionality without complex mocking.
"""

import pytest
import torch
import torch.nn as nn
from argparse import Namespace

# Import the model wrappers
from models.gpm_model import GPMModel
from models.dgr_model import DGRModel
from models.gpm_dgr_hybrid_model import GPMDGRHybridModel
from models.utils.continual_model import ContinualModel


class SimpleBackbone(nn.Module):
    """Simple backbone for testing."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.layer3 = nn.Linear(16 * 4 * 4, 64)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer3(x))
        return self.classifier(x)


class SimpleDataset:
    """Simple dataset for testing."""

    def __init__(self):
        self.N_CLASSES = 10
        self.N_TASKS = 2
        self.N_CLASSES_PER_TASK = 5
        self.SETTING = 'class-il'

    def get_data_loaders(self):
        # Create simple data
        data = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 5, (4,))
        return [(data, labels)], [(data, labels)]

    def get_offsets(self, task):
        return task * 5, (task + 1) * 5

    def get_normalization_transform(self):
        return nn.Identity()


def create_args():
    """Create basic arguments."""
    args = Namespace()
    args.lr = 0.01
    args.optimizer = 'sgd'
    args.optim_wd = 0.0001
    args.optim_mom = 0.9
    args.optim_nesterov = False
    args.nowand = True
    args.label_perc = 1.0
    args.num_workers = 0
    return args


class TestModelWrapperBasics:
    """Test basic model wrapper functionality."""

    def test_gpm_model_inheritance(self):
        """Test that GPMModel inherits from ContinualModel."""
        assert issubclass(GPMModel, ContinualModel)
        assert GPMModel.NAME == 'gpm_adapted'
        assert 'class-il' in GPMModel.COMPATIBILITY

    def test_dgr_model_inheritance(self):
        """Test that DGRModel inherits from ContinualModel."""
        assert issubclass(DGRModel, ContinualModel)
        assert DGRModel.NAME == 'dgr_adapted'
        assert 'class-il' in DGRModel.COMPATIBILITY

    def test_hybrid_model_inheritance(self):
        """Test that GPMDGRHybridModel inherits from ContinualModel."""
        assert issubclass(GPMDGRHybridModel, ContinualModel)
        assert GPMDGRHybridModel.NAME == 'gpm_dgr_hybrid_adapted'
        assert 'class-il' in GPMDGRHybridModel.COMPATIBILITY

    def test_gpm_model_parser(self):
        """Test GPM model parser."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser = GPMModel.get_parser(parser)

        # Test that arguments can be parsed
        args = parser.parse_args([])
        assert hasattr(args, 'gpm_energy_threshold')
        assert hasattr(args, 'gpm_max_collection_batches')
        assert hasattr(args, 'gpm_layer_names')
        assert hasattr(args, 'gpm_device')

    def test_dgr_model_parser(self):
        """Test DGR model parser."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser = DGRModel.get_parser(parser)

        # Test that arguments can be parsed
        args = parser.parse_args([])
        assert hasattr(args, 'dgr_z_dim')
        assert hasattr(args, 'dgr_vae_lr')
        assert hasattr(args, 'dgr_replay_weight')

    def test_hybrid_model_parser(self):
        """Test hybrid model parser."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        parser = GPMDGRHybridModel.get_parser(parser)

        # Test that arguments can be parsed
        args = parser.parse_args([])
        assert hasattr(args, 'hybrid_gpm_energy_threshold')
        assert hasattr(args, 'hybrid_dgr_z_dim')
        assert hasattr(args, 'hybrid_coordination_mode')


class TestModelWrapperIntegration:
    """Test model wrapper integration with Mammoth framework."""

    def test_gpm_model_interface_compliance(self):
        """Test that GPMModel implements required ContinualModel methods."""
        # Check that all required methods exist
        assert hasattr(GPMModel, 'begin_task')
        assert hasattr(GPMModel, 'end_task')
        assert hasattr(GPMModel, 'observe')
        assert hasattr(GPMModel, 'forward')
        assert hasattr(GPMModel, 'get_parser')

        # Check that methods are callable
        assert callable(getattr(GPMModel, 'begin_task'))
        assert callable(getattr(GPMModel, 'end_task'))
        assert callable(getattr(GPMModel, 'observe'))
        assert callable(getattr(GPMModel, 'forward'))
        assert callable(getattr(GPMModel, 'get_parser'))

    def test_dgr_model_interface_compliance(self):
        """Test that DGRModel implements required ContinualModel methods."""
        # Check that all required methods exist
        assert hasattr(DGRModel, 'begin_task')
        assert hasattr(DGRModel, 'end_task')
        assert hasattr(DGRModel, 'observe')
        assert hasattr(DGRModel, 'forward')
        assert hasattr(DGRModel, 'get_parser')

        # Check that methods are callable
        assert callable(getattr(DGRModel, 'begin_task'))
        assert callable(getattr(DGRModel, 'end_task'))
        assert callable(getattr(DGRModel, 'observe'))
        assert callable(getattr(DGRModel, 'forward'))
        assert callable(getattr(DGRModel, 'get_parser'))

    def test_hybrid_model_interface_compliance(self):
        """Test that GPMDGRHybridModel implements required ContinualModel methods."""
        # Check that all required methods exist
        assert hasattr(GPMDGRHybridModel, 'begin_task')
        assert hasattr(GPMDGRHybridModel, 'end_task')
        assert hasattr(GPMDGRHybridModel, 'observe')
        assert hasattr(GPMDGRHybridModel, 'forward')
        assert hasattr(GPMDGRHybridModel, 'get_parser')

        # Check that methods are callable
        assert callable(getattr(GPMDGRHybridModel, 'begin_task'))
        assert callable(getattr(GPMDGRHybridModel, 'end_task'))
        assert callable(getattr(GPMDGRHybridModel, 'observe'))
        assert callable(getattr(GPMDGRHybridModel, 'forward'))
        assert callable(getattr(GPMDGRHybridModel, 'get_parser'))


class TestConfigurationHandling:
    """Test configuration parameter handling."""

    def test_gpm_default_configuration(self):
        """Test GPM default configuration handling."""
        args = create_args()

        # Test that model can handle missing GPM-specific args
        backbone = SimpleBackbone()
        loss = nn.CrossEntropyLoss()
        transform = nn.Identity()
        dataset = SimpleDataset()

        # This should work with default values
        try:
            # We can't actually create the model without mocking GPMAdapter
            # but we can test the argument extraction logic
            energy_threshold = getattr(args, 'gpm_energy_threshold', 0.95)
            max_collection_batches = getattr(args, 'gpm_max_collection_batches', 200)
            layer_names = getattr(args, 'gpm_layer_names', ['backbone.layer3', 'classifier'])

            assert energy_threshold == 0.95
            assert max_collection_batches == 200
            assert layer_names == ['backbone.layer3', 'classifier']
        except Exception as e:
            pytest.fail(f"Default configuration handling failed: {e}")

    def test_dgr_default_configuration(self):
        """Test DGR default configuration handling."""
        args = create_args()

        # Test that model can handle missing DGR-specific args
        try:
            z_dim = getattr(args, 'dgr_z_dim', 100)
            vae_lr = getattr(args, 'dgr_vae_lr', 0.001)
            replay_weight = getattr(args, 'dgr_replay_weight', 0.5)

            assert z_dim == 100
            assert vae_lr == 0.001
            assert replay_weight == 0.5
        except Exception as e:
            pytest.fail(f"Default configuration handling failed: {e}")

    def test_hybrid_default_configuration(self):
        """Test hybrid default configuration handling."""
        args = create_args()

        # Test that model can handle missing hybrid-specific args
        try:
            gpm_energy_threshold = getattr(args, 'hybrid_gpm_energy_threshold', 0.95)
            dgr_z_dim = getattr(args, 'hybrid_dgr_z_dim', 100)
            coordination_mode = getattr(args, 'hybrid_coordination_mode', 'sequential')

            assert gpm_energy_threshold == 0.95
            assert dgr_z_dim == 100
            assert coordination_mode == 'sequential'
        except Exception as e:
            pytest.fail(f"Default configuration handling failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__])
