"""
Test suite for DGR Mammoth Integration

This module tests the Deep Generative Replay (DGR) integration with the Mammoth
framework, ensuring that the VAE functionality and replay generation work correctly
and maintain compatibility with the original DGR behavior.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace
from unittest.mock import Mock, patch
import tempfile
import os

# Add the project root to the path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.dgr_mammoth_adapter import DGRMammothAdapter, DGRVAE
from backbone.ResNet32 import ResNet32
from datasets.seq_cifar100_einstellung_224 import SequentialCIFAR100Einstellung224


class TestDGRVAE:
    """Test the adapted DGR VAE functionality."""

    @pytest.fixture
    def vae_config(self):
        """Configuration for VAE testing."""
        return {
            'image_size': 32,
            'image_channels': 3,
            'z_dim': 64,
            'fc_layers': 3,
            'fc_units': 256,
            'recon_loss': 'BCE',
            'network_output': 'sigmoid'
        }

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def vae(self, vae_config, device):
        """Create a VAE instance for testing."""
        vae = DGRVAE(**vae_config, device=device)
        vae.to(device)
        vae.set_optimizer(lr=0.001)
        return vae

    def test_vae_initialization(self, vae, vae_config):
        """Test that VAE initializes correctly with original architecture."""
        assert vae.z_dim == vae_config['z_dim']
        assert vae.image_size == vae_config['image_size']
        assert vae.image_channels == vae_config['image_channels']
        assert vae.fc_layers == vae_config['fc_layers']
        assert vae.recon_loss == vae_config['recon_loss']
        assert vae.network_output == vae_config['network_output']

        # Check that optimizer is set
        assert vae.optimizer is not None
        assert isinstance(vae.optimizer, torch.optim.Adam)

    def test_vae_forward_pass(self, vae, vae_config, device):
        """Test VAE forward pass functionality."""
        batch_size = 8
        x = torch.randn(batch_size, vae_config['image_channels'],
                       vae_config['image_size'], vae_config['image_size']).to(device)

        # Test full forward pass
        x_recon, mu, logvar, z = vae(x, full=True, reparameterize=True)

        assert x_recon.shape == x.shape
        assert mu.shape == (batch_size, vae_config['z_dim'])
        assert logvar.shape == (batch_size, vae_config['z_dim'])
        assert z.shape == (batch_size, vae_config['z_dim'])

        # Test reconstruction-only forward pass
        x_recon_only = vae(x, full=False)
        assert x_recon_only.shape == x.shape

    def test_vae_encoding_decoding(self, vae, vae_config, device):
        """Test VAE encoding and decoding separately."""
        batch_size = 4
        x = torch.randn(batch_size, vae_config['image_channels'],
                       vae_config['image_size'], vae_config['image_size']).to(device)

        # Test encoding
        z_mean, z_logvar, hE, hidden_x = vae.encode(x)
        assert z_mean.shape == (batch_size, vae_config['z_dim'])
        assert z_logvar.shape == (batch_size, vae_config['z_dim'])

        # Test reparameterization
        z = vae.reparameterize(z_mean, z_logvar)
        assert z.shape == (batch_size, vae_config['z_dim'])

        # Test decoding
        x_recon = vae.decode(z)
        assert x_recon.shape == x.shape

    def test_vae_loss_computation(self, vae, vae_config, device):
        """Test VAE loss function computation."""
        batch_size = 4
        x = torch.randn(batch_size, vae_config['image_channels'],
                       vae_config['image_size'], vae_config['image_size']).to(device)

        x_recon, mu, logvar, z = vae(x, full=True, reparameterize=True)

        # Test loss computation
        reconL, variatL = vae.loss_function(x=x, x_recon=x_recon, mu=mu, z=z, logvar=logvar)

        assert isinstance(reconL, torch.Tensor)
        assert isinstance(variatL, torch.Tensor)
        assert reconL.dim() == 0  # Scalar loss
        assert variatL.dim() == 0  # Scalar loss
        assert reconL.item() >= 0  # Non-negative reconstruction loss
        assert variatL.item() >= 0  # Non-negative variational loss

    def test_vae_sample_generation(self, vae, vae_config, device):
        """Test VAE sample generation functionality."""
        n_samples = 10

        samples = vae.generate_samples(n_samples, device)

        assert samples.shape == (n_samples, vae_config['image_channels'],
                               vae_config['image_size'], vae_config['image_size'])
        assert samples.device == device

        # Test that samples are in valid range for sigmoid output
        if vae_config['network_output'] == 'sigmoid':
            assert torch.all(samples >= 0)
            assert torch.all(samples <= 1)

    def test_vae_training_step(self, vae, vae_config, device):
        """Test VAE training on a batch."""
        batch_size = 8
        x = torch.randn(batch_size, vae_config['image_channels'],
                       vae_config['image_size'], vae_config['image_size']).to(device)

        # Test training without replay
        loss_dict = vae.train_on_batch(x)

        assert 'total_loss' in loss_dict
        assert 'recon_current' in loss_dict
        assert 'variat_current' in loss_dict
        assert loss_dict['total_loss'] > 0

        # Test training with replay data
        x_replay = torch.randn_like(x)
        loss_dict_replay = vae.train_on_batch(x, x_replay, replay_weight=0.5)

        assert 'recon_replay' in loss_dict_replay
        assert 'variat_replay' in loss_dict_replay
        assert loss_dict_replay['total_loss'] > 0


class TestDGRMammothAdapter:
    """Test the DGR Mammoth adapter functionality."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for DGR adapter."""
        args = Namespace()
        args.dgr_z_dim = 64
        args.dgr_vae_lr = 0.001
        args.dgr_vae_fc_layers = 3
        args.dgr_vae_fc_units = 256
        args.dgr_replay_weight = 0.5
        args.dgr_vae_train_epochs = 1
        args.lr = 0.01
        args.optimizer = 'sgd'
        args.optim_wd = 0.0001
        args.optim_mom = 0.9
        args.optim_nesterov = False
        args.label_perc = 1.0  # Required by ContinualModel
        args.buffer_size = 0   # No buffer for DGR
        args.dataset = 'seq-cifar100'
        args.nowand = True     # Disable wandb for testing
        return args

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset for testing."""
        dataset = Mock()
        dataset.SIZE = [3, 32, 32]  # CIFAR-like dimensions
        dataset.N_CLASSES = 100
        dataset.N_TASKS = 10
        dataset.N_CLASSES_PER_TASK = 10
        dataset.SETTING = 'class-il'
        dataset.get_offsets = Mock(return_value=(0, 10))
        dataset.get_normalization_transform = Mock(return_value=nn.Identity())
        return dataset

    @pytest.fixture
    def backbone(self):
        """Create a backbone network for testing."""
        return ResNet32(num_classes=100)

    @pytest.fixture
    def loss_fn(self):
        """Create loss function for testing."""
        return nn.CrossEntropyLoss()

    @pytest.fixture
    def transform(self):
        """Create transform for testing."""
        return nn.Identity()

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def dgr_model(self, backbone, loss_fn, mock_args, transform, mock_dataset, device):
        """Create DGR model for testing."""
        model = DGRMammothAdapter(backbone, loss_fn, mock_args, transform, mock_dataset)
        model.to(device)
        return model

    def test_dgr_initialization(self, dgr_model, mock_args):
        """Test DGR adapter initialization."""
        assert dgr_model.NAME == 'dgr'
        assert 'class-il' in dgr_model.COMPATIBILITY
        assert dgr_model.z_dim == mock_args.dgr_z_dim
        assert dgr_model.vae_lr == mock_args.dgr_vae_lr
        assert dgr_model.replay_weight == mock_args.dgr_replay_weight

        # Check VAE initialization
        assert dgr_model.vae is not None
        assert isinstance(dgr_model.vae, DGRVAE)
        assert dgr_model.vae.z_dim == mock_args.dgr_z_dim

        # Check that previous VAE is initially None
        assert dgr_model.previous_vae is None

    def test_dgr_parser(self):
        """Test DGR argument parser."""
        from argparse import ArgumentParser
        parser = ArgumentParser()
        DGRMammothAdapter.get_parser(parser)

        # Check that DGR arguments are added
        args = parser.parse_args(['--dgr_z_dim', '128', '--dgr_vae_lr', '0.002'])
        assert args.dgr_z_dim == 128
        assert args.dgr_vae_lr == 0.002

    def test_dgr_begin_task(self, dgr_model, mock_dataset):
        """Test task beginning functionality."""
        # Add some dummy data to buffer
        dgr_model.current_task_buffer = [torch.randn(3, 32, 32)]

        dgr_model.begin_task(mock_dataset)

        # Buffer should be cleared
        assert len(dgr_model.current_task_buffer) == 0

    def test_dgr_observe_first_task(self, dgr_model, device):
        """Test observation on first task (no replay)."""
        batch_size = 4
        inputs = torch.randn(batch_size, 3, 32, 32).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)
        not_aug_inputs = inputs.clone()

        # First task should have no replay
        assert dgr_model.previous_vae is None

        loss = dgr_model.observe(inputs, labels, not_aug_inputs)

        assert isinstance(loss, float)
        assert loss > 0

        # Check that data was added to buffer
        assert len(dgr_model.current_task_buffer) == batch_size

    def test_dgr_end_task(self, dgr_model, mock_dataset, device):
        """Test task completion functionality."""
        # Add some data to buffer
        batch_size = 8
        dummy_data = [torch.randn(3, 32, 32) for _ in range(batch_size)]
        dgr_model.current_task_buffer = dummy_data

        # Mock VAE training to avoid actual training
        with patch.object(dgr_model, '_train_vae_on_current_task') as mock_train:
            dgr_model.end_task(mock_dataset)
            mock_train.assert_called_once()

        # Previous VAE should be set
        assert dgr_model.previous_vae is not None
        assert isinstance(dgr_model.previous_vae, DGRVAE)

    def test_dgr_observe_with_replay(self, dgr_model, device):
        """Test observation with replay data."""
        # Set up previous VAE
        dgr_model.previous_vae = dgr_model.vae
        dgr_model._current_task = 1  # Simulate second task

        batch_size = 4
        inputs = torch.randn(batch_size, 3, 32, 32).to(device)
        labels = torch.randint(0, 10, (batch_size,)).to(device)
        not_aug_inputs = inputs.clone()

        # Mock replay generation to avoid actual VAE sampling
        with patch.object(dgr_model, '_generate_replay_data') as mock_replay:
            replay_inputs = torch.randn_like(inputs)
            replay_labels = torch.randint(0, 10, (batch_size,)).to(device)
            mock_replay.return_value = (replay_inputs, replay_labels)

            loss = dgr_model.observe(inputs, labels, not_aug_inputs)

            assert isinstance(loss, float)
            assert loss > 0
            mock_replay.assert_called_once_with(batch_size)

    def test_dgr_replay_generation(self, dgr_model, device):
        """Test replay data generation."""
        # Set up previous VAE
        dgr_model.previous_vae = dgr_model.vae
        dgr_model._n_past_classes = 10

        batch_size = 6

        # Mock VAE sample generation
        with patch.object(dgr_model.previous_vae, 'generate_samples') as mock_gen:
            mock_samples = torch.randn(batch_size, 3, 32, 32).to(device)
            mock_gen.return_value = mock_samples

            replay_inputs, replay_labels = dgr_model._generate_replay_data(batch_size)

            assert replay_inputs.shape == (batch_size, 3, 32, 32)
            assert replay_labels.shape == (batch_size,)
            assert torch.all(replay_labels < 10)  # Labels should be from previous classes
            mock_gen.assert_called_once_with(batch_size, device)

    def test_dgr_vae_training(self, dgr_model, device):
        """Test VAE training on current task data."""
        # Add data to buffer
        batch_size = 16
        dummy_data = [torch.randn(3, 32, 32) for _ in range(batch_size)]
        dgr_model.current_task_buffer = dummy_data

        # Mock VAE training methods to avoid actual training
        with patch.object(dgr_model.vae, 'train_on_batch') as mock_train:
            mock_train.return_value = {'total_loss': 1.0}

            dgr_model._train_vae_on_current_task()

            # Should have called train_on_batch
            assert mock_train.called

    def test_dgr_forward_pass(self, dgr_model, device):
        """Test forward pass through the model."""
        batch_size = 4
        inputs = torch.randn(batch_size, 3, 32, 32).to(device)

        outputs = dgr_model.forward(inputs)

        assert outputs.shape == (batch_size, 100)  # 100 classes


class TestDGRIntegrationWithEinstellung:
    """Test DGR integration with Einstellung dataset."""

    @pytest.fixture
    def einstellung_args(self):
        """Create arguments for Einstellung integration test."""
        args = Namespace()
        args.dgr_z_dim = 64
        args.dgr_vae_lr = 0.001
        args.dgr_vae_fc_layers = 3
        args.dgr_vae_fc_units = 256
        args.dgr_replay_weight = 0.5
        args.dgr_vae_train_epochs = 1
        args.lr = 0.01
        args.optimizer = 'sgd'
        args.optim_wd = 0.0001
        args.optim_mom = 0.9
        args.optim_nesterov = False
        args.dataset = 'seq-cifar100-einstellung-224'
        args.buffer_size = 0  # No buffer for DGR
        args.label_perc = 1.0  # Required by ContinualModel
        args.nowand = True     # Disable wandb for testing
        return args

    @pytest.mark.slow
    def test_dgr_einstellung_compatibility(self, einstellung_args):
        """Test that DGR works with Einstellung dataset."""
        # This test requires the actual dataset, so we'll mock it
        with patch('datasets.seq_cifar100_einstellung_224.SequentialCIFAR100Einstellung224') as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset.SIZE = [3, 224, 224]
            mock_dataset.N_CLASSES = 100
            mock_dataset.N_TASKS = 10
            mock_dataset.N_CLASSES_PER_TASK = 10
            mock_dataset.SETTING = 'class-il'
            mock_dataset.get_offsets = Mock(return_value=(0, 10))
            mock_dataset.get_normalization_transform = Mock(return_value=nn.Identity())
            mock_dataset_class.return_value = mock_dataset

            backbone = ResNet32(num_classes=100)
            loss_fn = nn.CrossEntropyLoss()
            transform = nn.Identity()

            # Should initialize without errors
            dgr_model = DGRMammothAdapter(backbone, loss_fn, einstellung_args, transform, mock_dataset)

            assert dgr_model.image_size == 224  # Should adapt to Einstellung size
            assert dgr_model.image_channels == 3
            assert isinstance(dgr_model.vae, DGRVAE)


class TestDGRPerformance:
    """Test DGR performance characteristics."""

    @pytest.fixture
    def performance_model(self):
        """Create DGR model for performance testing."""
        args = Namespace()
        args.dgr_z_dim = 64
        args.dgr_vae_lr = 0.001
        args.dgr_vae_fc_layers = 3
        args.dgr_vae_fc_units = 256
        args.dgr_replay_weight = 0.5
        args.dgr_vae_train_epochs = 1
        args.lr = 0.01
        args.optimizer = 'sgd'
        args.optim_wd = 0.0001
        args.optim_mom = 0.9
        args.optim_nesterov = False
        args.label_perc = 1.0  # Required by ContinualModel
        args.buffer_size = 0   # No buffer for DGR
        args.dataset = 'seq-cifar100'
        args.nowand = True     # Disable wandb for testing

        dataset = Mock()
        dataset.SIZE = [3, 32, 32]
        dataset.N_CLASSES = 100
        dataset.N_TASKS = 10
        dataset.N_CLASSES_PER_TASK = 10
        dataset.SETTING = 'class-il'
        dataset.get_offsets = Mock(return_value=(0, 10))
        dataset.get_normalization_transform = Mock(return_value=nn.Identity())

        backbone = ResNet32(num_classes=100)
        loss_fn = nn.CrossEntropyLoss()
        transform = nn.Identity()

        return DGRMammothAdapter(backbone, loss_fn, args, transform, dataset)

    @pytest.mark.slow
    def test_dgr_training_time(self, performance_model):
        """Test that DGR training completes within reasonable time."""
        import time

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        performance_model.to(device)

        batch_size = 32
        n_batches = 10

        start_time = time.time()

        for i in range(n_batches):
            inputs = torch.randn(batch_size, 3, 32, 32).to(device)
            labels = torch.randint(0, 10, (batch_size,)).to(device)
            not_aug_inputs = inputs.clone()

            loss = performance_model.observe(inputs, labels, not_aug_inputs)
            assert loss > 0

        end_time = time.time()
        training_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert training_time < 30.0, f"Training took too long: {training_time:.2f}s"

    def test_dgr_memory_usage(self, performance_model):
        """Test that DGR doesn't use excessive memory."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        performance_model.to(device)

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Perform several training steps
            batch_size = 32
            for i in range(5):
                inputs = torch.randn(batch_size, 3, 32, 32).to(device)
                labels = torch.randint(0, 10, (batch_size,)).to(device)
                not_aug_inputs = inputs.clone()

                loss = performance_model.observe(inputs, labels, not_aug_inputs)

            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 1GB)
            assert memory_increase < 1e9, f"Memory usage too high: {memory_increase / 1e6:.2f}MB"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
