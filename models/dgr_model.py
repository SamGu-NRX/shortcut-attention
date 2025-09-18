"""
DGR Model Wrapper for Mammoth Framework

This module provides a ContinualModel wrapper for the adapted DGR implementation,
ensuring seamless integration with Mammoth's training pipeline and evaluation framework.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from argparse import ArgumentParser
from typing import TYPE_CHECKING, Tuple, Optional

from models.utils.continual_model import ContinualModel
from models.dgr_mammoth_adapter import DGRVAE
from utils.args import add_rehearsal_args

if TYPE_CHECKING:
    from datasets.utils.continual_dataset import ContinualDataset


class DGRModel(ContinualModel):
    """
    DGR (Deep Generative Replay) model wrapper for Mammoth framework.

    This wrapper integrates the adapted DGR implementation with Mammoth's ContinualModel
    interface, providing standard hooks and configuration parameter handling.
    """

    NAME = 'dgr_adapted'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        """Add DGR-specific arguments to the parser."""
        parser.add_argument('--dgr_z_dim', type=int, default=100,
                          help='Latent dimension for DGR VAE (default: 100)')
        parser.add_argument('--dgr_vae_lr', type=float, default=0.001,
                          help='Learning rate for VAE training (default: 0.001)')
        parser.add_argument('--dgr_vae_fc_layers', type=int, default=3,
                          help='Number of FC layers in VAE (default: 3)')
        parser.add_argument('--dgr_vae_fc_units', type=int, default=400,
                          help='Number of units in VAE FC layers (default: 400)')
        parser.add_argument('--dgr_replay_weight', type=float, default=0.5,
                          help='Weight for replay data in training (default: 0.5)')
        parser.add_argument('--dgr_vae_train_epochs', type=int, default=1,
                          help='Number of epochs to train VAE per task (default: 1)')
        parser.add_argument('--dgr_buffer_size', type=int, default=1000,
                          help='Size of task data buffer for VAE training (default: 1000)')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """Initialize DGR model wrapper."""
        super().__init__(backbone, loss, args, transform, dataset)

        # Extract DGR configuration from args
        self.z_dim = getattr(args, 'dgr_z_dim', 100)
        self.vae_lr = getattr(args, 'dgr_vae_lr', 0.001)
        self.vae_fc_layers = getattr(args, 'dgr_vae_fc_layers', 3)
        self.vae_fc_units = getattr(args, 'dgr_vae_fc_units', 400)
        self.replay_weight = getattr(args, 'dgr_replay_weight', 0.5)
        self.vae_train_epochs = getattr(args, 'dgr_vae_train_epochs', 1)
        self.buffer_size = getattr(args, 'dgr_buffer_size', 1000)

        # Get image properties from dataset
        # Get image properties from dataset
        if hasattr(dataset, 'get_data_loaders'):
            # Try to get image dimensions from dataset
            train_loader, _ = dataset.get_data_loaders()
            # Peek at first batch without consuming it
            for sample_batch in train_loader:
                self.image_shape = sample_batch[0].shape[1:]  # (C, H, W)
                break
        else:
            # Default ImageNet-like dimensions
            self.image_shape = (3, 224, 224)
            
        # Initialize VAE components
        self.current_vae = None
        self.previous_vae = None
        self.current_task_buffer = []

        logging.info(f"DGR model initialized with z_dim: {self.z_dim}, image_shape: {self.image_shape}")
        logging.info(f"DGR replay weight: {self.replay_weight}, VAE train epochs: {self.vae_train_epochs}")

    def begin_task(self, dataset: 'ContinualDataset') -> None:
        """Prepare for new task."""
        super().begin_task(dataset)

        # Move previous VAE to storage
        if self.current_vae is not None:
            self.previous_vae = self.current_vae
            self.previous_vae.eval()  # Set to evaluation mode

        # Initialize new VAE for current task
        image_channels, image_height, image_width = self.image_shape
        image_size = max(image_height, image_width)  # Assume square images
        self.current_vae = DGRVAE(
            image_size=image_size,
            image_channels=image_channels,
            z_dim=self.z_dim,
            fc_layers=self.vae_fc_layers,
            fc_units=self.vae_fc_units
        ).to(self.device)

        # Clear task buffer
        self.current_task_buffer.clear()

        logging.info(f"DGR: Beginning task {self.current_task}")

    def end_task(self, dataset: 'ContinualDataset') -> None:
        """Train VAE on current task data after task completion."""
        super().end_task(dataset)

        if len(self.current_task_buffer) > 0:
            logging.info(f"DGR: Training VAE on {len(self.current_task_buffer)} samples")
            self._train_vae_on_current_task()

        logging.info(f"DGR: Completed task {self.current_task}")

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, epoch: int = None) -> float:
        """
        Compute a training step with DGR generative replay.

        Args:
            inputs: batch of examples
            labels: ground-truth labels
            not_aug_inputs: batch of inputs without augmentation
            epoch: current epoch number

        Returns:
            the value of the loss function
        """
        # Store data for VAE training
        if len(self.current_task_buffer) < self.buffer_size:
            self.current_task_buffer.extend([x.cpu() for x in not_aug_inputs])

        # Generate replay data if previous VAE exists
        replay_inputs, replay_labels = self._generate_replay_data(inputs.size(0))

        # Forward pass on current data
        outputs = self.net(inputs)
        loss_current = self.loss(outputs, labels)

        total_loss = loss_current

        # Add replay loss if available
        if replay_inputs is not None:
            replay_outputs = self.net(replay_inputs)
            loss_replay = self.loss(replay_outputs, replay_labels)
            total_loss = (1 - self.replay_weight) * loss_current + self.replay_weight * loss_replay

        # Backward pass and optimization
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        return total_loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(x)

    def _generate_replay_data(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate replay data using the previous VAE.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tuple of (replay_inputs, replay_labels) or (None, None) if no previous VAE
        """
        if self.previous_vae is None:
            return None, None

        # Generate samples from previous VAE
        replay_inputs = self.previous_vae.generate_samples(batch_size, self.device)

        # For simplicity, generate random labels from previous tasks
        # In a more sophisticated implementation, you might use the classifier
        # to generate pseudo-labels or store label information
        n_previous_classes = self.n_past_classes
        if n_previous_classes > 0:
            replay_labels = torch.randint(0, n_previous_classes, (batch_size,), device=self.device)
        else:
            replay_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        return replay_inputs, replay_labels

    def _train_vae_on_current_task(self):
        """Train the VAE on current task data."""
        if len(self.current_task_buffer) == 0:
            return

        # Convert buffer to tensor
        task_data = torch.stack(self.current_task_buffer).to(self.device)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(task_data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=min(128, len(task_data)), shuffle=True
        )

        # Train VAE
        self.current_vae.train_vae(dataloader, epochs=self.vae_train_epochs, lr=self.vae_lr)

        logging.info(f"DGR: VAE training completed for task {self.current_task}")
