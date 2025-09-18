#!/usr/bin/env python3
"""
DGR Complete Integration Test

This script tests the complete DGR integration with the Mammoth experiment pipeline.
"""

import os
import sys
import logging
from pathlib import Path
from argparse import Namespace

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_simple_backbone(input_size, num_classes):
    """Create a simple backbone that can handle any input size."""
    import torch.nn as nn

    if input_size == 32:
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

def test_dgr_model_availability():
    """Test that DGR model is available in Mammoth."""
    logger.info("Testing DGR model availability...")

    from models import get_model_names
    models = get_model_names()

    assert 'dgr' in models, "DGR model not found in available models"

    from models.dgr_mammoth_adapter import DGRMammothAdapter
    assert models['dgr'] == DGRMammothAdapter, "Wrong DGR class registered"

    logger.info("✅ DGR model is properly registered")

def test_dgr_training_simulation():
    """Test DGR training simulation."""
    logger.info("Testing DGR training simulation...")

    import torch
    import torch.nn as nn
    from models.dgr_mammoth_adapter import DGRMammothAdapter
    from unittest.mock import Mock

    device = torch.device('cpu')

    # Create test arguments
    args = Namespace()
    args.dgr_z_dim = 32
    args.dgr_vae_lr = 0.001
    args.dgr_vae_fc_layers = 2
    args.dgr_vae_fc_units = 128
    args.dgr_replay_weight = 0.5
    args.dgr_vae_train_epochs = 1
    args.dgr_disable_replay_monitoring = True
    args.lr = 0.01
    args.optimizer = 'adam'
    args.optim_wd = 0.0001
    args.optim_mom = 0.9
    args.optim_nesterov = False
    args.label_perc = 1.0
    args.buffer_size = 0
    args.dataset = 'seq-cifar100'
    args.nowand = True

    # Mock dataset
    dataset = Mock()
    dataset.SIZE = [3, 32, 32]
    dataset.N_CLASSES = 100
    dataset.N_TASKS = 2
    dataset.N_CLASSES_PER_TASK = 50
    dataset.SETTING = 'class-il'
    dataset.get_offsets = Mock(side_effect=lambda task: (task * 50, (task + 1) * 50))
    dataset.get_normalization_transform = Mock(return_value=nn.Identity())

    # Create model
    backbone = create_simple_backbone(32, 100)
    loss_fn = nn.CrossEntropyLoss()
    transform = nn.Identity()

    model = DGRMammothAdapter(backbone, loss_fn, args, transform, dataset)
    model.to(device)

    # Simulate 2-task training
    for task_id in range(2):
        logger.info(f"  Simulating task {task_id + 1}")

        # Begin task
        model.begin_task(dataset)

        # Training steps
        for step in range(5):
            batch_size = 8
            inputs = torch.randn(batch_size, 3, 32, 32).to(device)
            labels = torch.randint(task_id * 50, (task_id + 1) * 50, (batch_size,)).to(device)
            not_aug_inputs = inputs.clone()

            loss = model.observe(inputs, labels, not_aug_inputs)
            assert isinstance(loss, float)
            assert loss > 0

        # End task
        model.end_task(dataset)

        # After first task, should have previous VAE
        if task_id == 0:
            assert model.previous_vae is not None

            # Test replay generation
            replay_samples = model.previous_vae.generate_samples(4, device)
            assert replay_samples.shape == (4, 3, 32, 32)

    logger.info("✅ DGR training simulation successful")

def test_dgr_einstellung_compatibility():
    """Test DGR compatibility with Einstellung evaluation."""
    logger.info("Testing DGR Einstellung compatibility...")

    import torch
    import torch.nn as nn
    from models.dgr_mammoth_adapter import DGRMammothAdapter
    from unittest.mock import Mock

    device = torch.device('cpu')

    # Create test arguments for Einstellung
    args = Namespace()
    args.dgr_z_dim = 32
    args.dgr_vae_lr = 0.001
    args.dgr_vae_fc_layers = 2
    args.dgr_vae_fc_units = 128
    args.dgr_replay_weight = 0.5
    args.dgr_vae_train_epochs = 1
    args.dgr_disable_replay_monitoring = True
    args.lr = 0.01
    args.optimizer = 'adam'
    args.optim_wd = 0.0001
    args.optim_mom = 0.9
    args.optim_nesterov = False
    args.label_perc = 1.0
    args.buffer_size = 0
    args.dataset = 'seq-cifar100-einstellung-224'
    args.nowand = True

    # Mock Einstellung dataset
    dataset = Mock()
    dataset.SIZE = [3, 224, 224]
    dataset.N_CLASSES = 100
    dataset.N_TASKS = 2
    dataset.N_CLASSES_PER_TASK = 50
    dataset.SETTING = 'class-il'
    dataset.get_offsets = Mock(side_effect=lambda task: (task * 50, (task + 1) * 50))
    dataset.get_normalization_transform = Mock(return_value=nn.Identity())

    # Create model with backbone that can handle 224x224
    backbone = create_simple_backbone(224, 100)
    loss_fn = nn.CrossEntropyLoss()
    transform = nn.Identity()

    model = DGRMammothAdapter(backbone, loss_fn, args, transform, dataset)
    model.to(device)

    # Test that model adapts to Einstellung image size
    assert model.image_size == 224
    assert model.image_channels == 3

    # Test forward pass with Einstellung-sized inputs
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        assert outputs.shape == (batch_size, 100)

    # Test training step
    model.train()
    labels = torch.randint(0, 50, (batch_size,)).to(device)
    not_aug_inputs = inputs.clone()

    loss = model.observe(inputs, labels, not_aug_inputs)
    assert isinstance(loss, float)
    assert loss > 0

    logger.info("✅ DGR Einstellung compatibility verified")

def test_dgr_command_line_compatibility():
    """Test DGR command line compatibility."""
    logger.info("Testing DGR command line compatibility...")

    from argparse import ArgumentParser
    from models.dgr_mammoth_adapter import DGRMammothAdapter

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='sgd')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)

    # Add DGR arguments
    parser = DGRMammothAdapter.get_parser(parser)

    # Test command
    test_args = [
        '--model', 'dgr',
        '--backbone', 'resnet18',
        '--dgr_z_dim', '100',
        '--dgr_vae_lr', '0.001',
        '--dgr_replay_weight', '0.5',
        '--dgr_disable_replay_monitoring',
        '--n_epochs', '2',
        '--batch_size', '32'
    ]

    args = parser.parse_args(test_args)

    assert args.model == 'dgr'
    assert args.backbone == 'resnet18'
    assert args.dgr_z_dim == 100
    assert args.dgr_vae_lr == 0.001
    assert args.dgr_replay_weight == 0.5
    assert args.dgr_disable_replay_monitoring == True
    assert args.n_epochs == 2
    assert args.batch_size == 32

    logger.info("✅ DGR command line compatibility verified")

def main():
    """Run all DGR integration tests."""
    logger.info("🚀 Starting DGR Complete Integration Tests")
    logger.info("=" * 60)

    try:
        # Test model availability
        test_dgr_model_availability()

        # Test training simulation
        test_dgr_training_simulation()

        # Test Einstellung compatibility
        test_dgr_einstellung_compatibility()

        # Test command line compatibility
        test_dgr_command_line_compatibility()

        logger.info("=" * 60)
        logger.info("🎉 ALL DGR INTEGRATION TESTS PASSED!")
        logger.info("")
        logger.info("DGR is ready for use with the experiment runner:")
        logger.info("  python run_einstellung_experiment.py --model dgr --backbone resnet18 --auto_checkpoint")
        logger.info("")
        logger.info("Available DGR-specific arguments:")
        logger.info("  --dgr_z_dim: Latent dimension (default: 100)")
        logger.info("  --dgr_vae_lr: VAE learning rate (default: 0.001)")
        logger.info("  --dgr_replay_weight: Replay weight (default: 0.5)")
        logger.info("  --dgr_disable_replay_monitoring: Disable replay visualization")
        logger.info("  --dgr_vae_train_epochs: VAE training epochs per task (default: 1)")

        return True

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
