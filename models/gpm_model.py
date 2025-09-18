"""
GPM Model Wrapper for Mammoth Framework

This module provides a ContinualModel wrapper for the adapted GPM implementation,
ensuring seamless integration with Mammoth's training pipeline and evaluation framework.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from argparse import ArgumentParser
from typing import TYPE_CHECKING

from models.utils.continual_model import ContinualModel
from models.gpm import GPMAdapter
from utils.args import add_rehearsal_args

if TYPE_CHECKING:
    from datasets.utils.continual_dataset import ContinualDataset


class GPMModel(ContinualModel):
    """
    GPM (Gradient Projection Memory) model wrapper for Mammoth framework.

    This wrapper integrates the adapted GPM implementation with Mammoth's ContinualModel
    interface, providing standard hooks and configuration parameter handling.
    """

    NAME = 'gpm_adapted'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        """Add GPM-specific arguments to the parser."""
        parser.add_argument('--gpm_energy_threshold', type=float, default=0.95,
                          help='Energy threshold for GPM basis selection (default: 0.95)')
        parser.add_argument('--gpm_max_collection_batches', type=int, default=200,
                          help='Maximum batches for activation collection (default: 200)')
        parser.add_argument('--gpm_layer_names', type=str, nargs='+',
                          default=['backbone.layer3', 'classifier'],
                          help='Layer names for GPM projection (default: backbone.layer3 classifier)')
        parser.add_argument('--gpm_device', type=str, default='auto',
                          help='Device for GPM computations (auto/cpu/cuda) (default: auto)')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """Initialize GPM model wrapper."""
        super().__init__(backbone, loss, args, transform, dataset)

        # Extract GPM configuration from args
        self.energy_threshold = getattr(args, 'gpm_energy_threshold', 0.95)
        self.max_collection_batches = getattr(args, 'gmp_max_collection_batches', 200)
        self.layer_names = getattr(args, 'gmp_layer_names', ['backbone.layer3', 'classifier'])

        # Determine device
        gmp_device = getattr(args, 'gmp_device', 'auto')
        if gpm_device == 'auto':
            self.gpm_device = self.device
        else:
            self.gmp_device = torch.device(gmp_device)

        # Initialize GPM adapter
        self.gmp_adapter = GPMAdapter(
            model=self.net,
            layer_names=self.layer_names,
            energy_threshold=self.energy_threshold,
            device=self.gmp_device
        )

        # Task data collection
        self.current_task_data = []

        logging.info(f"GPM model initialized with energy threshold: {self.energy_threshold}")
        logging.info(f"GPM layer names: {self.layer_names}")

    def begin_task(self, dataset: 'ContinualDataset') -> None:
        """Prepare for new task."""
        super().begin_task(dataset)
        self.current_task_data.clear()
        logging.info(f"GPM: Beginning task {self.current_task}")

    def end_task(self, dataset: 'ContinualDataset') -> None:
        """Update GPM memory after task completion."""
        super().end_task(dataset)

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
            logging.info(f"GPM: Updating memory with {len(self.current_task_data)} samples")
            self.gmp_adapter.update_memory(task_loader, self.max_collection_batches)

        logging.info(f"GPM: Completed task {self.current_task}")

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, epoch: int = None) -> float:
        """
        Compute a training step with GPM gradient projection.

        Args:
            inputs: batch of examples
            labels: ground-truth labels
            not_aug_inputs: batch of inputs without augmentation
            epoch: current epoch number

        Returns:
            the value of the loss function
        """
        # Store data for memory update
        if len(self.current_task_data) < 1000:  # Limit memory usage
            self.current_task_data.extend([(x.cpu(), y.cpu()) for x, y in zip(not_aug_inputs, labels)])

        # Forward pass
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        # Backward pass
        self.opt.zero_grad()
        loss.backward()

        # Apply GPM gradient projection
        self.gmp_adapter.project_gradients()

        # Optimizer step
        self.opt.step()

        return loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(x)
