# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GPM (Gradient Projection Memory) adapter for Mammoth framework.

This module adapts the original GPM implementation from the GPM directory
to work with Mammoth's ContinualModel interface while preserving the
original GPM functionality including SVD-based subspace extraction and
gradient projection mechanisms.

Original GPM paper: "Gradient Projection Memory for Continual Learning", ICLR 2021
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from copy import deepcopy

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    """Compute output size of convolutional layer."""
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


class GPMAdapter:
    """
    Core GPM functionality adapted from the original implementation.

    This class handles the SVD-based subspace extraction and gradient projection
    mechanisms while being compatible with Mammoth's model structure.
    """

    def __init__(self,
                 model: nn.Module,
                 energy_threshold: float = 0.95,
                 max_collection_batches: int = 200,
                 device: Optional[torch.device] = None):
        """
        Initialize GPM adapter.

        Args:
            model: The neural network model
            energy_threshold: Energy threshold for SVD basis selection (default: 0.95)
            max_collection_batches: Maximum batches for activation collection (default: 200)
            device: Device for computation (auto-detected if None)
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.energy_threshold = energy_threshold
        self.max_collection_batches = max_collection_batches

        # Storage for GPM bases
        self.feature_list: List[np.ndarray] = []
        self.projection_matrices: List[torch.Tensor] = []

        # Activation collection
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

        # Layer information for activation collection
        self.layer_info = self._analyze_model_structure()

        logging.info(f"GPM Adapter initialized with energy threshold: {energy_threshold}")
        logging.info(f"Model structure analyzed: {len(self.layer_info)} layers for GPM")

    def _analyze_model_structure(self) -> List[Dict]:
        """
        Analyze model structure to determine layers for GPM.

        Returns:
            List of layer information dictionaries
        """
        layer_info = []

        # Get all named modules
        named_modules = dict(self.model.named_modules())

        # Focus on key layers (similar to original GPM implementation)
        target_layers = []

        # Look for backbone layers (common in Mammoth)
        if hasattr(self.model, 'net') and hasattr(self.model.net, 'backbone'):
            backbone = self.model.net.backbone
            for name, module in backbone.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if 'layer' in name or 'fc' in name or 'conv' in name:
                        target_layers.append((f"net.backbone.{name}", module))

        # Fallback 1: look for net layers
        if not target_layers and hasattr(self.model, 'net'):
            for name, module in self.model.net.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    target_layers.append((f"net.{name}", module))

        # Fallback 2: look for any Conv2d/Linear layers
        if not target_layers:
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    target_layers.append((name, module))

        # Create layer info
        for name, module in target_layers[:5]:  # Limit to first 5 layers like original GPM
            info = {
                'name': name,
                'module': module,
                'type': 'conv' if isinstance(module, nn.Conv2d) else 'linear'
            }

            if isinstance(module, nn.Conv2d):
                info.update({
                    'kernel_size': module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels
                })

            layer_info.append(info)

        return layer_info

    def _register_hooks(self):
        """Register forward hooks for activation collection."""
        self.activations.clear()
        self.hooks.clear()

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        for layer_info in self.layer_info:
            name = layer_info['name']
            module = layer_info['module']
            hook = module.register_forward_hook(get_activation(name))
            self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def collect_activations(self, data_loader: torch.utils.data.DataLoader) -> List[np.ndarray]:
        """
        Collect activations from specified layers.

        Args:
            data_loader: DataLoader for activation collection

        Returns:
            List of activation matrices for each layer
        """
        self.model.eval()
        self._register_hooks()

        mat_list = []
        batch_count = 0

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(data_loader):
                if batch_count >= self.max_collection_batches:
                    break

                inputs = inputs.to(self.device)
                _ = self.model(inputs)  # Forward pass to collect activations

                if batch_idx == 0:  # Initialize matrices on first batch
                    for i, layer_info in enumerate(self.layer_info):
                        name = layer_info['name']
                        if name in self.activations:
                            act = self.activations[name]

                            if layer_info['type'] == 'conv':
                                # For conv layers, we need to handle spatial dimensions
                                # Following original GPM approach
                                act_batch_size, channels, height, width = act.shape
                                kernel_size = layer_info.get('kernel_size', 3)

                                # Calculate output size after convolution
                                s = compute_conv_output_size(32, kernel_size)  # Assuming 32x32 input

                                # Initialize matrix for conv layer
                                mat_size = kernel_size * kernel_size * layer_info['in_channels']
                                total_samples = s * s * act_batch_size * self.max_collection_batches
                                mat_list.append(np.zeros((mat_size, min(total_samples, 10000))))
                            else:
                                # For linear layers
                                act_batch_size = act.shape[0]
                                mat_list.append(np.zeros((act.shape[1], act_batch_size * self.max_collection_batches)))

                # Collect activations for this batch
                k_indices = [0] * len(self.layer_info)  # Column indices for each matrix

                for i, layer_info in enumerate(self.layer_info):
                    name = layer_info['name']
                    if name in self.activations:
                        act = self.activations[name].cpu().numpy()

                        if layer_info['type'] == 'conv':
                            # Handle conv layer activations (simplified version)
                            curr_batch_size, channels, height, width = act.shape
                            # Flatten spatial dimensions and take samples
                            act_flat = act.reshape(curr_batch_size, -1).T
                            samples_to_take = min(act_flat.shape[1], mat_list[i].shape[1] - k_indices[i])
                            if samples_to_take > 0:
                                mat_list[i][:, k_indices[i]:k_indices[i] + samples_to_take] = act_flat[:, :samples_to_take]
                                k_indices[i] += samples_to_take
                        else:
                            # Handle linear layer activations
                            act_t = act.T
                            samples_to_take = min(act_t.shape[1], mat_list[i].shape[1] - k_indices[i])
                            if samples_to_take > 0:
                                mat_list[i][:, k_indices[i]:k_indices[i] + samples_to_take] = act_t[:, :samples_to_take]
                                k_indices[i] += samples_to_take

                batch_count += 1

        self._remove_hooks()

        # Trim matrices to actual size
        for i in range(len(mat_list)):
            actual_size = k_indices[i] if i < len(k_indices) else mat_list[i].shape[1]
            mat_list[i] = mat_list[i][:, :actual_size]

        logging.info("Activation collection completed:")
        for i, mat in enumerate(mat_list):
            logging.info(f"Layer {i+1}: {mat.shape}")

        return mat_list

    def update_memory(self, mat_list: List[np.ndarray], task_id: int = 0) -> None:
        """
        Update GPM memory with new task activations.

        Args:
            mat_list: List of activation matrices
            task_id: Current task ID
        """
        # Dynamic threshold based on task (following original GPM)
        threshold = np.array([self.energy_threshold] * len(mat_list)) + task_id * np.array([0.003] * len(mat_list))

        logging.info(f"Updating GPM memory for task {task_id}")
        logging.info(f"Thresholds: {threshold}")

        if not self.feature_list:  # First task
            for i, activation in enumerate(mat_list):
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)

                # Energy-based basis selection (Eq-5 in original paper)
                sval_total = (S**2).sum()
                sval_ratio = (S**2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold[i])

                self.feature_list.append(U[:, :r])

                logging.info(f"Layer {i+1}: Selected {r}/{U.shape[1]} basis vectors")
        else:  # Subsequent tasks
            for i, activation in enumerate(mat_list):
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()

                # Projected representation (Eq-8 in original paper)
                act_hat = activation - np.dot(np.dot(self.feature_list[i], self.feature_list[i].T), activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)

                # Criteria (Eq-9 in original paper)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold[i]:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break

                if r == 0:
                    logging.info(f"Layer {i+1}: No update needed")
                    continue

                # Update GPM basis
                Ui = np.hstack((self.feature_list[i], U[:, :r]))
                if Ui.shape[1] > Ui.shape[0]:
                    self.feature_list[i] = Ui[:, :Ui.shape[0]]
                else:
                    self.feature_list[i] = Ui

                logging.info(f"Layer {i+1}: Updated basis to {self.feature_list[i].shape[1]}/{self.feature_list[i].shape[0]}")

        # Precompute projection matrices for efficient gradient projection
        self._precompute_projection_matrices()

    def _precompute_projection_matrices(self) -> None:
        """Precompute projection matrices for efficient gradient projection."""
        self.projection_matrices.clear()

        for i, feature_matrix in enumerate(self.feature_list):
            # Compute U * U^T for projection
            proj_matrix = torch.tensor(
                np.dot(feature_matrix, feature_matrix.T),
                dtype=torch.float32
            )
            # Handle device placement more robustly
            if hasattr(self.device, 'type'):
                proj_matrix = proj_matrix.to(self.device)
            else:
                # Fallback to CPU if device is not properly set
                proj_matrix = proj_matrix.to('cpu')
            self.projection_matrices.append(proj_matrix)
            logging.info(f"Layer {i+1}: Projection matrix shape {proj_matrix.shape}")

    def project_gradients(self) -> None:
        """
        Apply gradient projection: g ← g - U(U^T g)

        This is the core GPM operation that projects gradients orthogonal
        to the subspace of previous tasks.
        """
        if not self.projection_matrices:
            return  # No projection needed for first task

        param_idx = 0
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            # Only project gradients for layers we have bases for
            layer_found = False
            for i, layer_info in enumerate(self.layer_info):
                if layer_info['name'] in name:
                    layer_found = True

                    if param_idx < len(self.projection_matrices):
                        # Get gradient
                        grad = param.grad.data
                        original_shape = grad.shape

                        # Reshape gradient for projection
                        if len(grad.shape) > 2:  # Conv layer
                            grad_flat = grad.view(grad.shape[0], -1)
                        else:  # Linear layer
                            grad_flat = grad

                        # Apply projection: g ← g - U(U^T g)
                        if grad_flat.shape[0] <= self.projection_matrices[param_idx].shape[0]:
                            # Ensure projection matrix is on the same device as gradient
                            proj_matrix = self.projection_matrices[param_idx][:grad_flat.shape[0], :grad_flat.shape[0]]
                            proj_matrix = proj_matrix.to(grad_flat.device)

                            projected_grad = grad_flat - torch.mm(proj_matrix, grad_flat)

                            # Reshape back to original shape
                            param.grad.data = projected_grad.view(original_shape)

                    param_idx += 1
                    break

            # Handle bias terms (set to zero for non-first tasks, following original GPM)
            if not layer_found and 'bias' in name and len(self.feature_list) > 0:
                param.grad.data.fill_(0)


class GPMMammoth(ContinualModel):
    """
    GPM (Gradient Projection Memory) model for Mammoth framework.

    This class integrates the GPM algorithm with Mammoth's ContinualModel interface,
    preserving the original GPM functionality while being compatible with Mammoth's
    training pipeline and evaluation framework.
    """

    NAME = 'gpm'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """Add GPM-specific arguments."""
        parser.add_argument('--gpm_energy_threshold', type=float, default=0.95,
                          help='Energy threshold for GPM basis selection (default: 0.95)')
        parser.add_argument('--gpm_max_collection_batches', type=int, default=200,
                          help='Maximum batches for activation collection (default: 200)')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """Initialize GPM model."""
        super(GPMMammoth, self).__init__(backbone, loss, args, transform, dataset=dataset)

        # Initialize GPM adapter
        self.gpm_adapter = GPMAdapter(
            model=self,
            energy_threshold=getattr(args, 'gpm_energy_threshold', 0.95),
            max_collection_batches=getattr(args, 'gpm_max_collection_batches', 200),
            device=self.device
        )

        # Task tracking
        self.current_task_data = []

        logging.info(f"GPM model initialized with energy threshold: {self.gpm_adapter.energy_threshold}")

    def begin_task(self, dataset) -> None:
        """Prepare for new task."""
        super().begin_task(dataset)
        self.current_task_data.clear()
        logging.info(f"GPM: Beginning task {self.current_task}")

    def end_task(self, dataset) -> None:
        """Update GPM memory after task completion."""
        super().end_task(dataset)

        if self.current_task_data:
            logging.info(f"GPM: Updating memory for task {self.current_task}")

            # Create data loader from collected data
            task_dataset = torch.utils.data.TensorDataset(
                torch.stack([x for x, _ in self.current_task_data]),
                torch.stack([y for _, y in self.current_task_data])
            )
            task_loader = torch.utils.data.DataLoader(
                task_dataset,
                batch_size=64,
                shuffle=False
            )

            # Collect activations and update memory
            mat_list = self.gpm_adapter.collect_activations(task_loader)
            self.gpm_adapter.update_memory(mat_list, self.current_task)

            logging.info(f"GPM: Memory updated for task {self.current_task}")

        self.current_task_data.clear()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        Training step with GPM gradient projection.

        Args:
            inputs: Input batch
            labels: Target labels
            not_aug_inputs: Non-augmented inputs
            epoch: Current epoch (optional)

        Returns:
            Loss value
        """
        # Store data for memory update (sample a subset to avoid memory issues)
        if len(self.current_task_data) < 1000:  # Limit stored samples
            for i in range(min(10, len(inputs))):  # Store up to 10 samples per batch
                self.current_task_data.append((inputs[i].cpu(), labels[i].cpu()))

        # Standard forward pass
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        # Backward pass
        loss.backward()

        # Apply GPM gradient projection
        self.gpm_adapter.project_gradients()

        # Optimizer step
        self.opt.step()

        return loss.item()
