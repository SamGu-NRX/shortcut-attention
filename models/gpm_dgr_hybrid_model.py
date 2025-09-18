"""
GPM + DGR Hybrid Model Wrapper for Mammoth Framework

This module provides a ContinualModel wrapper for the hybrid GPM + DGR implementation,
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
from models.gpm import GPMAdapter
from models.dgr_mammoth_adapter import DGRVAE
from utils.args import add_rehearsal_args

if TYPE_CHECKING:
    from datasets.utils.continual_dataset import ContinualDataset


class GPMDGRHybridModel(ContinualModel):
    """
    GPM + DGR Hybrid model wrapper for Mammoth framework.

    This wrapper integrates the hybrid approach with Mammoth's ContinualModel interface,
    coordinating both GPM gradient projection and DGR generative replay in a single
    training pipeline.
    """

    NAME = 'gpm_dgr_hybrid_adapted'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        """Add hybrid method arguments to the parser."""
        # GPM arguments
        parser.add_argument('--hybrid_gmp_energy_threshold', type=float, default=0.95,
                          help='Energy threshold for GPM basis selection (default: 0.95)')
        parser.add_argument('--hybrid_gmp_max_collection_batches', type=int, default=200,
                          help='Maximum batches for GPM activation collection (default: 200)')
        parser.add_argument('--hybrid_gmp_layer_names', type=str, nargs='+',
                          default=['backbone.layer3', 'classifier'],
                          help='Layer names for GPM projection (default: backbone.layer3 classifier)')

        # DGR arguments
        parser.add_argument('--hybrid_dgr_z_dim', type=int, default=100,
                          help='Latent dimension for DGR VAE (default: 100)')
        parser.add_argument('--hybrid_dgr_vae_lr', type=float, default=0.001,
                          help='Learning rate for VAE training (default: 0.001)')
        parser.add_argument('--hybrid_dgr_vae_fc_layers', type=int, default=3,
                          help='Number of FC layers in VAE (default: 3)')
        parser.add_argument('--hybrid_dgr_vae_fc_units', type=int, default=400,
                          help='Number of units in VAE FC layers (default: 400)')
        parser.add_argument('--hybrid_dgr_replay_weight', type=float, default=0.5,
                          help='Weight for replay data in training (default: 0.5)')
        parser.add_argument('--hybrid_dgr_vae_train_epochs', type=int, default=1,
                          help='Number of epochs to train VAE per task (default: 1)')
        parser.add_argument('--hybrid_dgr_buffer_size', type=int, default=1000,
                          help='Size of task data buffer for VAE training (default: 1000)')

        # Coordination arguments
        parser.add_argument('--hybrid_coordination_mode', type=str, default='sequential',
                          choices=['sequential', 'parallel'],
                          help='Coordination mode for GPM and DGR updates (default: sequential)')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """Initialize hybrid model wrapper."""
        super().__init__(backbone, loss, args, transform, dataset)

        # Extract GPM configuration from args
        self.gmp_energy_threshold = getattr(args, 'hybrid_gmp_energy_threshold', 0.95)
        self.gmp_max_collection_batches = getattr(args, 'hybrid_gmp_max_collection_batches', 200)
        self.gmp_layer_names = getattr(args, 'hybrid_gmp_layer_names', ['backbone.layer3', 'classifier'])

        # Extract DGR configuration from args
        self.dgr_z_dim = getattr(args, 'hybrid_dgr_z_dim', 100)
        self.dgr_vae_lr = getattr(args, 'hybrid_dgr_vae_lr', 0.001)
        self.dgr_vae_fc_layers = getattr(args, 'hybrid_dgr_vae_fc_layers', 3)
        self.dgr_vae_fc_units = getattr(args, 'hybrid_dgr_vae_fc_units', 400)
        self.dgr_replay_weight = getattr(args, 'hybrid_dgr_replay_weight', 0.5)
        self.dgr_vae_train_epochs = getattr(args, 'hybrid_dgr_vae_train_epochs', 1)
        self.dgr_buffer_size = getattr(args, 'hybrid_dgr_buffer_size', 1000)

        # Coordination configuration
        self.coordination_mode = getattr(args, 'hybrid_coordination_mode', 'sequential')

        # Get image properties from dataset
        if hasattr(dataset, 'get_data_loaders'):
            # Try to get image dimensions from dataset
            train_loader, _ = dataset.get_data_loaders()
            sample_batch = next(iter(train_loader))
            self.image_shape = sample_batch[0].shape[1:]  # (C, H, W)
        else:
            # Default CIFAR-100 dimensions
            self.image_shape = (3, 224, 224)

        # Initialize GPM adapter
        self.gmp_adapter = GPMAdapter(
            model=self.net,
            layer_names=self.gmp_layer_names,
            energy_threshold=self.gmp_energy_threshold,
            device=self.device
        )

        # Initialize DGR components
        self.current_vae = None
        self.previous_vae = None
        self.current_task_buffer = []
        self.current_task_data = []  # For GPM memory update

        logging.info(f"Hybrid model initialized with GPM energy threshold: {self.gmp_energy_threshold}")
        logging.info(f"Hybrid model initialized with DGR z_dim: {self.dgr_z_dim}, image_shape: {self.image_shape}")
        logging.info(f"Coordination mode: {self.coordination_mode}")

    def begin_task(self, dataset: 'ContinualDataset') -> None:
        """Prepare for new task."""
        super().begin_task(dataset)

        # GPM preparation
        self.current_task_data.clear()

        # DGR preparation
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
            z_dim=self.dgr_z_dim,
            fc_layers=self.dgr_vae_fc_layers,
            fc_units=self.dgr_vae_fc_units
        ).to(self.device)

        # Clear task buffer
        self.current_task_buffer.clear()

        logging.info(f"Hybrid: Beginning task {self.current_task}")

    def end_task(self, dataset: 'ContinualDataset') -> None:
        """Update both GPM memory and train DGR VAE after task completion."""
        super().end_task(dataset)

        if self.coordination_mode == 'sequential':
            # Update GPM memory first, then train DGR VAE
            self._update_gmp_memory()
            self._train_dgr_vae()
        elif self.coordination_mode == 'parallel':
            # Update both simultaneously (simplified parallel approach)
            self._update_gmp_memory()
            self._train_dgr_vae()

        logging.info(f"Hybrid: Completed task {self.current_task}")

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, epoch: int = None) -> float:
        """
        Compute a training step with both GPM projection and DGR replay.

        Training sequence:
        1. Generate DGR replay data (if available)
        2. Combine real and replay data
        3. Forward pass and loss computation
        4. Backward pass
        5. Apply GPM gradient projection
        6. Optimizer step

        Args:
            inputs: batch of examples
            labels: ground-truth labels
            not_aug_inputs: batch of inputs without augmentation
            epoch: current epoch number

        Returns:
            the value of the loss function
        """
        # Store data for both GPM and DGR updates
        if len(self.current_task_data) < 1000:  # Limit memory usage for GPM
            self.current_task_data.extend([(x.cpu(), y.cpu()) for x, y in zip(not_aug_inputs, labels)])

        if len(self.current_task_buffer) < self.dgr_buffer_size:  # Store for DGR VAE training
            self.current_task_buffer.extend([x.cpu() for x in not_aug_inputs])

        # Step 1: Generate DGR replay data
        replay_inputs, replay_labels = self._generate_replay_data(inputs.size(0))

        # Step 2: Forward pass on current data
        outputs = self.net(inputs)
        loss_current = self.loss(outputs, labels)

        total_loss = loss_current

        # Step 3: Add replay loss if available
        if replay_inputs is not None:
            replay_outputs = self.net(replay_inputs)
            loss_replay = self.loss(replay_outputs, replay_labels)
            total_loss = (1 - self.dgr_replay_weight) * loss_current + self.dgr_replay_weight * loss_replay

        # Step 4: Backward pass
        self.opt.zero_grad()
        total_loss.backward()

        # Step 5: Apply GPM gradient projection
        self.gmp_adapter.project_gradients()

        # Step 6: Optimizer step
        self.opt.step()

        return total_loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(x)

    def _update_gmp_memory(self):
        """Update GPM memory with current task data."""
        if self.current_task_data:
            # Create data loader from collected task data
            task_dataset = torch.utils.data.TensorDataset(
                torch.stack([x for x, _ in self.current_task_data]),
                torch.stack([y for _, y in self.current_task_data])
            )
            task_loader = torch.utils.data.DataLoader(
                task_dataset,
                batch_size=min(128, len(task_dataset)),
                shuffle=False
            )

            # Update GPM memory
            logging.info(f"Hybrid: Updating GPM memory with {len(self.current_task_data)} samples")
            self.gmp_adapter.update_memory(task_loader, self.gmp_max_collection_batches)

    def _train_dgr_vae(self):
        """Train DGR VAE on current task data."""
        if len(self.current_task_buffer) > 0:
            logging.info(f"Hybrid: Training DGR VAE on {len(self.current_task_buffer)} samples")

            # Convert buffer to tensor
            task_data = torch.stack(self.current_task_buffer).to(self.device)

            # Create data loader
            dataset = torch.utils.data.TensorDataset(task_data)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=min(128, len(task_data)), shuffle=True
            )

            # Train VAE
            self.current_vae.train_vae(dataloader, epochs=self.dgr_vae_train_epochs, lr=self.dgr_vae_lr)

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
