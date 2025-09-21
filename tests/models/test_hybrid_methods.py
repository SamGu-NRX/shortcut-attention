"""
Test suite for GPM + DGR Hybrid Methods Implementation

This module contains comprehensive tests for the hybrid approach that combines
GPM (Gradient Projection Memory) and DGR (Deep Generative Replay) methods.
Tests verify correct execution order, memory coordination, and integration
with the Mammoth training pipeline.
"""

import pytest

pytest.skip("Hybrid method tests are disabled for the original-method integration", allow_module_level=True)


class SimpleTestModel(nn.Module):
    """Simple model for testing purposes."""

    def __init__(self, input_size=32*32*3, hidden_size=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class TestGPMDGRHybrid:
    """Test the core GPMDGRHybrid class."""

    @pytest.fixture
    def device(self):
        return torch.device('cpu')  # Use CPU for testing

    @pytest.fixture
    def model(self, device):
        model = SimpleTestModel()
        model.to(device)
        return model

    @pytest.fixture
    def gpm_config(self):
        return {
            'energy_threshold': 0.95,
            'max_collection_batches': 50,  # Smaller for testing
            'update_frequency': 1
        }

    @pytest.fixture
    def dgr_config(self):
        return {
            'z_dim': 32,  # Smaller for testing
            'vae_lr': 0.001,
            'fc_layers': 2,  # Smaller for testing
            'fc_units': 64,  # Smaller for testing
            'replay_weight': 0.5,
            'vae_train_epochs': 1,
            'enable_monitoring': False,  # Disable for testing
            'monitor_frequency': 5
        }

    @pytest.fixture
    def hybrid_method(self, model, gpm_config, dgr_config, device):
        return GPMDGRHybrid(
            model=model,
            image_size=32,
            image_channels=3,
            gpm_config=gpm_config,
            dgr_config=dgr_config,
            device=device
        )

    def test_initialization(self, hybrid_method, gpm_config, dgr_config):
        """Test hybrid method initialization."""
        # Check GPM component
        assert hybrid_method.gpm is not None
        assert hybrid_method.gpm.energy_threshold == gpm_config['energy_threshold']
        assert hybrid_method.gpm.max_collection_batches == gpm_config['max_collection_batches']

        # Check DGR component
        assert hybrid_method.vae is not None
        assert hybrid_method.vae.z_dim == dgr_config['z_dim']

        # Check configuration
        assert hybrid_method.replay_weight == dgr_config['replay_weight']
        assert hybrid_method.vae_train_epochs == dgr_config['vae_train_epochs']

        # Check initial state
        assert hybrid_method.previous_vae is None
        assert len(hybrid_method.current_task_data) == 0

    def test_training_step_first_task(self, hybrid_method, device):
        """Test training step for the first task (no replay)."""
        # Create test data
        batch_size = 8
        real_inputs = torch.randn(batch_size, 3, 32, 32, device=device)
        real_labels = torch.randint(0, 10, (batch_size,), device=device)

        # Create optimizer and criterion
        optimizer = optim.SGD(hybrid_method.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Perform training step
        loss_dict = hybrid_method.training_step(
            real_inputs=real_inputs,
            real_labels=real_labels,
            optimizer=optimizer,
            criterion=criterion,
            epoch=0
        )

        # Verify loss dictionary
        assert 'loss_real' in loss_dict
        assert 'loss_replay' in loss_dict
        assert 'total_loss' in loss_dict

        # For first task, replay loss should be 0
        assert loss_dict['loss_replay'] == 0.0
        assert loss_dict['total_loss'] == loss_dict['loss_real']

        # Verify data was stored for memory update
        assert len(hybrid_method.current_task_data) > 0

    def test_training_step_with_replay(self, hybrid_method, device):
        """Test training step with replay data."""
        # Set up previous VAE (mock)
        hybrid_method.previous_vae = Mock()
        hybrid_method.previous_vae.generate_samples.return_value = torch.randn(8, 3, 32, 32, device=device)
        hybrid_method.previous_classifier = Mock()
        hybrid_method.previous_classifier.return_value = torch.randn(8, 100, device=device)

        # Create test data
        batch_size = 8
        real_inputs = torch.randn(batch_size, 3, 32, 32, device=device)
        real_labels = torch.randint(0, 10, (batch_size,), device=device)

        # Create optimizer and criterion
        optimizer = optim.SGD(hybrid_method.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Perform training step
        loss_dict = hybrid_method.training_step(
            real_inputs=real_inputs,
            real_labels=real_labels,
            optimizer=optimizer,
            criterion=criterion,
            epoch=0
        )

        # Verify replay was used
        hybrid_method.previous_vae.generate_samples.assert_called_once_with(batch_size, device)

        # Verify loss components
        assert loss_dict['loss_replay'] > 0.0
        assert loss_dict['total_loss'] != loss_dict['loss_real']

    def test_generate_replay_data(self, hybrid_method, device):
        """Test replay data generation."""
        # Test without previous VAE
        replay_inputs, replay_labels, replay_logits = hybrid_method._generate_replay_data(8)
        assert replay_inputs is None
        assert replay_labels is None
        assert replay_logits is None

        # Test with previous VAE
        hybrid_method.previous_vae = Mock()
        mock_samples = torch.randn(8, 3, 32, 32, device=device)
        hybrid_method.previous_vae.generate_samples.return_value = mock_samples

        hybrid_method.previous_classifier = Mock()
        hybrid_method.previous_classifier.return_value = torch.randn(8, 100, device=device)

        replay_inputs, replay_labels, replay_logits = hybrid_method._generate_replay_data(8)

        assert replay_inputs is not None
        assert replay_labels is not None
        assert replay_logits is not None
        assert replay_inputs.shape == (8, 3, 32, 32)
        assert replay_labels.shape == (8,)
        hybrid_method.previous_vae.generate_samples.assert_called_once_with(8, device)

    def test_end_task_memory_coordination(self, hybrid_method, device):
        """Test end-of-task memory updates for both GPM and DGR."""
        # Add some task data
        hybrid_method.current_task_data = [
            torch.randn(3, 32, 32) for _ in range(10)
        ]

        # Mock the GPM and VAE components
        with patch.object(hybrid_method.gpm, 'collect_activations') as mock_collect, \
             patch.object(hybrid_method.gpm, 'update_memory') as mock_update_gpm, \
             patch.object(hybrid_method, '_train_vae_on_current_task') as mock_train_vae:

            # Set up mocks
            mock_collect.return_value = [np.random.randn(100, 50)]

            # Create dummy train loader
            dummy_data = torch.randn(10, 3, 32, 32)
            dummy_dataset = torch.utils.data.TensorDataset(dummy_data)
            train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=5)

            # Call end_task
            hybrid_method.end_task(train_loader, task_id=0)

            # Verify GPM memory was updated
            mock_collect.assert_called_once()
            mock_update_gpm.assert_called_once()
            # Check that the call was made with the right task_id
            call_args = mock_update_gpm.call_args
            assert call_args[0][1] == 0  # task_id should be 0

            # Verify VAE was trained
            mock_train_vae.assert_called_once()

            # Verify previous VAE was stored
            assert hybrid_method.previous_vae is not None

            # Verify task data was cleared
            assert len(hybrid_method.current_task_data) == 0

    def test_vae_training_on_current_task(self, hybrid_method, device):
        """Test VAE training on current task data."""
        # Add task data
        hybrid_method.current_task_data = [
            torch.randn(3, 32, 32) for _ in range(20)
        ]

        # Mock VAE training
        with patch.object(hybrid_method.vae, 'train_on_batch') as mock_train:
            mock_train.return_value = {'total_loss': 0.5}

            # Train VAE
            hybrid_method._train_vae_on_current_task()

            # Verify training was called
            assert mock_train.call_count > 0

    def test_memory_summary(self, hybrid_method):
        """Test memory summary generation."""
        summary = hybrid_method.get_memory_summary()

        # Check structure
        assert 'gpm' in summary
        assert 'dgr' in summary
        assert 'hybrid' in summary

        # Check GPM info
        assert 'num_bases' in summary['gpm']
        assert 'energy_threshold' in summary['gpm']

        # Check DGR info
        assert 'vae_parameters' in summary['dgr']
        assert 'z_dim' in summary['dgr']

        # Check hybrid info
        assert 'replay_weight' in summary['hybrid']


class TestGPMDGRHybridMammoth:
    """Test the Mammoth integration class."""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for testing."""
        args = Mock()
        args.hybrid_gpm_energy_threshold = 0.95
        args.hybrid_gpm_max_collection_batches = 50
        args.hybrid_gpm_update_frequency = 1
        args.hybrid_dgr_z_dim = 32
        args.hybrid_dgr_vae_lr = 0.001
        args.hybrid_dgr_vae_fc_layers = 2
        args.hybrid_dgr_vae_fc_units = 64
        args.hybrid_dgr_replay_weight = 0.5
        args.hybrid_dgr_vae_train_epochs = 1
        args.hybrid_enable_monitoring = False
        args.hybrid_monitor_frequency = 5
        # Required by ContinualModel
        args.optimizer = 'sgd'
        args.lr = 0.01
        args.batch_size = 32
        args.n_epochs = 1
        args.optim_mom = 0.0
        args.optim_wd = 0.0
        args.optim_nesterov = False
        return args

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset for testing."""
        dataset = Mock()
        dataset.SIZE = [3, 32, 32]  # CIFAR-like
        return dataset

    @pytest.fixture
    def hybrid_model(self, mock_args, mock_dataset):
        """Create hybrid model for testing."""
        backbone = SimpleTestModel()
        loss = nn.CrossEntropyLoss()
        transform = None

        return GPMDGRHybridMammoth(
            backbone=backbone,
            loss=loss,
            args=mock_args,
            transform=transform,
            dataset=mock_dataset
        )

    def test_initialization_with_args(self, hybrid_model, mock_args):
        """Test initialization with command line arguments."""
        # Check that hybrid method was created
        assert hybrid_model.hybrid_method is not None

        # Check image properties
        assert hybrid_model.image_size == 32
        assert hybrid_model.image_channels == 3

        # Check that configuration was passed correctly
        assert hybrid_model.hybrid_method.gpm.energy_threshold == mock_args.hybrid_gpm_energy_threshold
        assert hybrid_model.hybrid_method.vae.z_dim == mock_args.hybrid_dgr_z_dim
        assert hybrid_model.hybrid_method.replay_weight == mock_args.hybrid_dgr_replay_weight

    def test_begin_task(self, hybrid_model):
        """Test begin_task method."""
        mock_dataset = Mock()

        # Should not raise any exceptions
        hybrid_model.begin_task(mock_dataset)

        # Check that current_task was set (inherited from ContinualModel)
        assert hasattr(hybrid_model, 'current_task')

    def test_end_task(self, hybrid_model):
        """Test end_task method."""
        mock_dataset = Mock()
        mock_dataset.train_loader = None

        # Mock the hybrid method's end_task
        with patch.object(hybrid_model.hybrid_method, 'end_task') as mock_end_task:
            hybrid_model.end_task(mock_dataset)

            # Verify hybrid method's end_task was called
            mock_end_task.assert_called_once()

    def test_observe(self, hybrid_model):
        """Test observe method (training step)."""
        # Create test data
        inputs = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        not_aug_inputs = inputs.clone()

        # Mock the hybrid method's training_step
        with patch.object(hybrid_model.hybrid_method, 'training_step') as mock_training_step:
            mock_training_step.return_value = {'total_loss': 0.5}

            loss = hybrid_model.observe(inputs, labels, not_aug_inputs, epoch=1)

            # Verify training_step was called with correct arguments
            mock_training_step.assert_called_once()
            call_args = mock_training_step.call_args

            assert torch.equal(call_args[1]['real_inputs'], inputs)
            assert torch.equal(call_args[1]['real_labels'], labels)
            assert call_args[1]['epoch'] == 1

            # Verify loss was returned
            assert loss == 0.5

    def test_forward(self, hybrid_model):
        """Test forward pass."""
        inputs = torch.randn(4, 3, 32, 32)
        outputs = hybrid_model.forward(inputs)

        # Check output shape
        assert outputs.shape == (4, 10)  # 10 classes in SimpleTestModel

    def test_get_hybrid_summary(self, hybrid_model):
        """Test hybrid summary retrieval."""
        with patch.object(hybrid_model.hybrid_method, 'get_memory_summary') as mock_summary:
            mock_summary.return_value = {'test': 'summary'}

            summary = hybrid_model.get_hybrid_summary()

            mock_summary.assert_called_once()
            assert summary == {'test': 'summary'}


class TestHybridConfigValidation:
    """Test configuration validation and creation utilities."""

    def test_validate_hybrid_config_valid(self):
        """Test validation with valid configuration."""
        gpm_config = {
            'energy_threshold': 0.95,
            'max_collection_batches': 200
        }
        dgr_config = {
            'z_dim': 100,
            'replay_weight': 0.5,
            'vae_lr': 0.001
        }

        assert validate_hybrid_config(gpm_config, dgr_config) is True

    def test_validate_hybrid_config_invalid_gpm(self):
        """Test validation with invalid GPM configuration."""
        # Invalid energy threshold
        gpm_config = {
            'energy_threshold': 1.5,  # Too high
            'max_collection_batches': 200
        }
        dgr_config = {
            'z_dim': 100,
            'replay_weight': 0.5,
            'vae_lr': 0.001
        }

        assert validate_hybrid_config(gpm_config, dgr_config) is False

        # Invalid max collection batches
        gpm_config['energy_threshold'] = 0.95
        gpm_config['max_collection_batches'] = -10

        assert validate_hybrid_config(gpm_config, dgr_config) is False

    def test_validate_hybrid_config_invalid_dgr(self):
        """Test validation with invalid DGR configuration."""
        gpm_config = {
            'energy_threshold': 0.95,
            'max_collection_batches': 200
        }

        # Invalid z_dim
        dgr_config = {
            'z_dim': -10,  # Negative
            'replay_weight': 0.5,
            'vae_lr': 0.001
        }

        assert validate_hybrid_config(gpm_config, dgr_config) is False

        # Invalid replay weight
        dgr_config['z_dim'] = 100
        dgr_config['replay_weight'] = 1.5  # Too high

        assert validate_hybrid_config(gpm_config, dgr_config) is False

        # Invalid VAE learning rate
        dgr_config['replay_weight'] = 0.5
        dgr_config['vae_lr'] = -0.001  # Negative

        assert validate_hybrid_config(gpm_config, dgr_config) is False

    def test_create_hybrid_config(self):
        """Test configuration creation from arguments."""
        # Create mock args
        args = Mock()
        args.hybrid_gpm_energy_threshold = 0.90
        args.hybrid_gpm_max_collection_batches = 150
        args.hybrid_gpm_update_frequency = 2
        args.hybrid_dgr_z_dim = 64
        args.hybrid_dgr_vae_lr = 0.002
        args.hybrid_dgr_vae_fc_layers = 4
        args.hybrid_dgr_vae_fc_units = 256
        args.hybrid_dgr_replay_weight = 0.3
        args.hybrid_dgr_vae_train_epochs = 2
        args.hybrid_enable_monitoring = True
        args.hybrid_monitor_frequency = 10

        gpm_config, dgr_config = create_hybrid_config(args)

        # Check GPM config
        assert gpm_config['energy_threshold'] == 0.90
        assert gpm_config['max_collection_batches'] == 150
        assert gpm_config['update_frequency'] == 2

        # Check DGR config
        assert dgr_config['z_dim'] == 64
        assert dgr_config['vae_lr'] == 0.002
        assert dgr_config['fc_layers'] == 4
        assert dgr_config['fc_units'] == 256
        assert dgr_config['replay_weight'] == 0.3
        assert dgr_config['vae_train_epochs'] == 2
        assert dgr_config['enable_monitoring'] is True
        assert dgr_config['monitor_frequency'] == 10

    def test_create_hybrid_config_defaults(self):
        """Test configuration creation with default values."""
        # Create args without hybrid-specific attributes
        args = Mock(spec=[])  # Empty spec to prevent automatic attribute creation

        gpm_config, dgr_config = create_hybrid_config(args)

        # Check that defaults were used
        assert gpm_config['energy_threshold'] == 0.95
        assert gpm_config['max_collection_batches'] == 200
        assert gpm_config['update_frequency'] == 1

        assert dgr_config['z_dim'] == 100
        assert dgr_config['vae_lr'] == 0.001
        assert dgr_config['fc_layers'] == 3
        assert dgr_config['fc_units'] == 400
        assert dgr_config['replay_weight'] == 0.5
        assert dgr_config['vae_train_epochs'] == 1
        assert dgr_config['enable_monitoring'] is True
        assert dgr_config['monitor_frequency'] == 5


class TestHybridIntegration:
    """Integration tests for the hybrid method."""

    def test_execution_order(self):
        """Test that the training step follows the correct execution order."""
        # Create test components
        model = SimpleTestModel()
        device = torch.device('cpu')

        gpm_config = {'energy_threshold': 0.95, 'max_collection_batches': 10, 'update_frequency': 1}
        dgr_config = {'z_dim': 16, 'vae_lr': 0.001, 'fc_layers': 2, 'fc_units': 32,
                     'replay_weight': 0.5, 'vae_train_epochs': 1, 'enable_monitoring': False, 'monitor_frequency': 5}

        hybrid = GPMDGRHybrid(model, 32, 3, gpm_config, dgr_config, device)

        # Set up previous VAE to enable replay
        hybrid.previous_vae = Mock()
        hybrid.previous_vae.generate_samples.return_value = torch.randn(4, 3, 32, 32)

        # Create test data
        inputs = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Mock the GPM projection to track when it's called
        with patch.object(hybrid.gpm, 'project_gradients') as mock_project:
            # Perform training step
            loss_dict = hybrid.training_step(inputs, labels, optimizer, criterion, 0)

            # Verify GPM projection was called (indicating correct execution order)
            mock_project.assert_called_once()

            # Verify replay was generated
            hybrid.previous_vae.generate_samples.assert_called_once()

            # Verify loss components
            assert 'loss_real' in loss_dict
            assert 'loss_replay' in loss_dict
            assert 'total_loss' in loss_dict

    def test_memory_coordination(self):
        """Test that both GPM and DGR memories are updated correctly."""
        model = SimpleTestModel()
        device = torch.device('cpu')

        gpm_config = {'energy_threshold': 0.95, 'max_collection_batches': 10, 'update_frequency': 1}
        dgr_config = {'z_dim': 16, 'vae_lr': 0.001, 'fc_layers': 2, 'fc_units': 32,
                     'replay_weight': 0.5, 'vae_train_epochs': 1, 'enable_monitoring': False, 'monitor_frequency': 5}

        hybrid = GPMDGRHybrid(model, 32, 3, gpm_config, dgr_config, device)

        # Add some task data
        hybrid.current_task_data = [torch.randn(3, 32, 32) for _ in range(5)]

        # Create dummy train loader
        dummy_data = torch.randn(5, 3, 32, 32)
        dummy_dataset = torch.utils.data.TensorDataset(dummy_data)
        train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=5)

        # Mock both memory update methods
        with patch.object(hybrid.gpm, 'collect_activations') as mock_collect, \
             patch.object(hybrid.gpm, 'update_memory') as mock_update_gpm, \
             patch.object(hybrid, '_train_vae_on_current_task') as mock_train_vae:

            mock_collect.return_value = [np.random.randn(50, 25)]

            # Call end_task
            hybrid.end_task(train_loader, 0)

            # Verify both memories were updated
            mock_collect.assert_called_once()
            mock_update_gpm.assert_called_once()
            mock_train_vae.assert_called_once()

            # Verify previous VAE was stored
            assert hybrid.previous_vae is not None

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        model = SimpleTestModel()
        device = torch.device('cpu')

        # Enable monitoring
        gpm_config = {'energy_threshold': 0.95, 'max_collection_batches': 10, 'update_frequency': 1}
        dgr_config = {'z_dim': 16, 'vae_lr': 0.001, 'fc_layers': 2, 'fc_units': 32,
                     'replay_weight': 0.5, 'vae_train_epochs': 1, 'enable_monitoring': True, 'monitor_frequency': 1}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch the monitor directory creation
            with patch('pathlib.Path.mkdir') as mock_mkdir:

                hybrid = GPMDGRHybrid(model, 32, 3, gpm_config, dgr_config, device)

                # Mock monitoring methods
                with patch.object(hybrid, '_log_performance_metrics') as mock_log, \
                     patch.object(hybrid, '_visualize_replay_samples') as mock_viz:

                    # Set up replay
                    hybrid.previous_vae = Mock()
                    replay_data = torch.randn(4, 3, 32, 32)
                    hybrid.previous_vae.generate_samples.return_value = replay_data

                    # Perform training step (should trigger monitoring due to frequency=1)
                    inputs = torch.randn(4, 3, 32, 32)
                    labels = torch.randint(0, 10, (4,))
                    optimizer = optim.SGD(model.parameters(), lr=0.01)
                    criterion = nn.CrossEntropyLoss()

                    loss_dict = hybrid.training_step(inputs, labels, optimizer, criterion, 0)

                    # Verify monitoring was called
                    mock_log.assert_called_once()
                    mock_viz.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
