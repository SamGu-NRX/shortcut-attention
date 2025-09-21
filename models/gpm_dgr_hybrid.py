"""
GPM + DGR Hybrid Methods Implementation

This module implements hybrid approaches that combine the adapted GPM (Gradient Projection Memory)
and DGR (Deep Generative Replay) methods in a single training loop. The hybrid methods coordinate
GPM gradient projection with DGR generative replay to leverage both gradient-based memory protection
and generative replay mechanisms.

The implementation follows the sequence: DGR replay generation → loss computation → GPM projection → optimizer step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import copy
from pathlib import Path

from models.utils.continual_model import ContinualModel
from models.gpm import GPMAdapter
from models.dgr_mammoth_adapter import DGRVAE
from utils.args import ArgumentParser

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class GPMDGRHybrid:
    """
    Core hybrid method combining GPM gradient projection with DGR generative replay.

    This class coordinates both methods in a single training loop, managing memory updates
    for both GPM bases and DGR VAE training after each task.
    """

    def __init__(self,
                 model: nn.Module,
                 image_size: int,
                 image_channels: int,
                 gpm_config: Dict[str, Any],
                 dgr_config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initialize the hybrid method.

        Args:
            model: The neural network model
            image_size: Size of input images (assumed square)
            image_channels: Number of input channels
            gpm_config: Configuration for GPM component
            dgr_config: Configuration for DGR component
            device: Device for computation
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.image_size = image_size
        self.image_channels = image_channels

        # Initialize GPM component
        self.gpm = GPMAdapter(
            model=model,
            energy_threshold=gpm_config.get('energy_threshold', 0.95),
            max_collection_batches=gpm_config.get('max_collection_batches', 200),
            device=device
        )

        # Initialize DGR VAE component
        self.vae = DGRVAE(
            image_size=image_size,
            image_channels=image_channels,
            z_dim=dgr_config.get('z_dim', 100),
            fc_layers=dgr_config.get('fc_layers', 3),
            fc_units=dgr_config.get('fc_units', 400),
            recon_loss=dgr_config.get('recon_loss', 'BCE'),
            network_output=dgr_config.get('network_output', 'sigmoid'),
            lr=dgr_config.get('vae_lr', 0.001),
            device=device
        )
        self.vae.to(self.device)

        # Store previous VAE for replay generation
        self.previous_vae = None
        self.previous_classifier = None

        # Configuration
        self.replay_weight = dgr_config.get('replay_weight', 0.5)
        self.vae_train_epochs = dgr_config.get('vae_train_epochs', 1)
        self.update_frequency = gpm_config.get('update_frequency', 1)  # How often to update memories
        self.replay_targets = dgr_config.get('replay_targets', 'hard')
        self.distill_temperature = dgr_config.get('distill_temperature', 2.0)
        self.replay_batch_size = dgr_config.get('replay_batch_size', 0)

        # Data storage for memory updates
        self.current_task_data = []

        # Performance monitoring
        self.enable_monitoring = dgr_config.get('enable_monitoring', True)
        self.monitor_frequency = dgr_config.get('monitor_frequency', 5)
        self.monitor_dir = None

        if self.enable_monitoring and VISUALIZATION_AVAILABLE:
            self.monitor_dir = Path("outputs") / "hybrid_monitoring"
            self.monitor_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Hybrid method monitoring enabled: {self.monitor_dir}")

        self.last_vae_stats: Dict[str, float] = {}

        logging.info(f"GPM+DGR Hybrid initialized: GPM threshold={self.gpm.energy_threshold}, "
                    f"DGR z_dim={self.vae.z_dim}, replay_weight={self.replay_weight}")

    def training_step(self,
                     real_inputs: torch.Tensor,
                     real_labels: torch.Tensor,
                     optimizer: torch.optim.Optimizer,
                     criterion: nn.Module,
                     epoch: int = 0) -> Dict[str, float]:
        """
        Combined training step with replay and GPM projection.

        Sequence: DGR replay generation → loss computation → GPM projection → optimizer step

        Args:
            real_inputs: Real input batch
            real_labels: Real label batch
            optimizer: Model optimizer
            criterion: Loss criterion
            epoch: Current epoch

        Returns:
            Dictionary with loss components
        """
        # Store data for memory updates (sample to avoid memory issues)
        if len(self.current_task_data) < 1000:  # Limit stored samples
            for i in range(min(10, len(real_inputs))):  # Store up to 10 samples per batch
                self.current_task_data.append(real_inputs[i].cpu())

        # Step 1: DGR replay generation
        replay_inputs = None
        replay_labels = None
        replay_logits = None

        current_task = getattr(self.model, 'current_task', 0)
        rnt = 1.0 / float(max(1, current_task + 1))

        if self.previous_vae is not None and self.previous_classifier is not None:
            replay_inputs, replay_labels, replay_logits = self._generate_replay_data(real_inputs.size(0))

        # Step 2: Loss computation
        optimizer.zero_grad()

        # Forward pass on real data
        real_outputs = self.model(real_inputs)
        loss_real = criterion(real_outputs, real_labels)

        total_loss = loss_real
        loss_dict = {
            'loss_real': loss_real.item(),
            'loss_replay': 0.0,
            'total_loss': loss_real.item()
        }

        # Add replay loss if available
        if replay_inputs is not None:
            replay_outputs = self.model(replay_inputs)
            loss_replay = self._compute_replay_loss(replay_outputs, replay_labels, replay_logits, criterion)

            total_loss = rnt * loss_real + (1 - rnt) * loss_replay

            loss_dict['loss_replay'] = float(loss_replay.item())
            loss_dict['total_loss'] = float(total_loss.item())

        # Step 3: Backward pass
        total_loss.backward()

        # Step 4: GPM gradient projection
        self.gpm.project_gradients()

        # Step 5: Optimizer step
        optimizer.step()

        # Step 6: Train VAE with current and replay data
        vae_current = real_inputs.detach()
        vae_replay = replay_inputs.detach() if replay_inputs is not None else None
        self.last_vae_stats = self.vae.train_batch(vae_current, vae_replay, rnt)

        # Monitor performance periodically
        if self.enable_monitoring and epoch % self.monitor_frequency == 0:
            self._monitor_hybrid_performance(epoch, loss_dict, replay_inputs)

        return loss_dict

    def _generate_replay_data(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate replay data using the previous VAE.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tuple of (replay_inputs, replay_labels, replay_logits) or (None, None, None)
        """
        if self.previous_vae is None or self.previous_classifier is None:
            return None, None, None

        try:
            # Generate samples from previous VAE
            replay_inputs = self.previous_vae.generate_samples(
                self.replay_batch_size or batch_size,
                device=self.device
            )

            with torch.no_grad():
                logits = self.previous_classifier(replay_inputs)
                replay_labels = logits.argmax(dim=1)

            return replay_inputs, replay_labels, logits.detach()

        except Exception as e:
            logging.warning(f"Failed to generate replay data: {e}")
            return None, None, None

    def _compute_replay_loss(self,
                              replay_outputs: torch.Tensor,
                              replay_labels: Optional[torch.Tensor],
                              replay_logits: Optional[torch.Tensor],
                              criterion: nn.Module) -> torch.Tensor:
        if replay_outputs is None:
            return torch.tensor(0.0, device=self.device)

        if self.replay_targets == 'soft':
            assert replay_logits is not None
            temperature = float(self.distill_temperature)
            kd_loss = F.kl_div(
                F.log_softmax(replay_outputs / temperature, dim=1),
                F.softmax(replay_logits / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)
            return kd_loss

        assert replay_labels is not None
        return criterion(replay_outputs, replay_labels)

    def end_task(self, train_loader: torch.utils.data.DataLoader, task_id: int) -> None:
        """
        Update both GPM bases and DGR VAE after task completion.

        Args:
            train_loader: DataLoader for the completed task
            task_id: ID of the completed task
        """
        logging.info(f"Hybrid method: Updating memory for task {task_id}")

        # Step 1: Update GPM memory
        try:
            if self.current_task_data:
                # Create data loader from collected data
                task_dataset = torch.utils.data.TensorDataset(
                    torch.stack(self.current_task_data)
                )
                task_loader = torch.utils.data.DataLoader(
                    task_dataset,
                    batch_size=64,
                    shuffle=False
                )

                # Collect activations and update GPM memory
                mat_list = self.gpm.collect_activations(task_loader)
                self.gpm.update_memory(mat_list, task_id)

                logging.info(f"GPM memory updated for task {task_id}")

        except Exception as e:
            logging.error(f"Failed to update GPM memory: {e}")

        # Step 2: Update DGR VAE
        try:
            if self.current_task_data:
                self._train_vae_on_current_task()

                # Store current VAE as previous for next task
                self.previous_vae = self.vae.clone_frozen()

                logging.info(f"DGR VAE updated for task {task_id}")

            # Store frozen classifier for label generation
            self.previous_classifier = copy.deepcopy(self.model).to(self.device)
            self.previous_classifier.eval()
            for param in self.previous_classifier.parameters():
                if isinstance(param, torch.Tensor):
                    param.requires_grad_(False)

        except Exception as e:
            logging.error(f"Failed to update DGR VAE: {e}")

        # Step 3: Clear current task data
        self.current_task_data.clear()

        # Step 4: Monitor memory state
        self._monitor_memory_state(task_id)

        logging.info(f"Hybrid method: Memory update completed for task {task_id}")

    def _train_vae_on_current_task(self) -> None:
        """Train the VAE on current task data."""
        if len(self.current_task_data) == 0:
            return

        # Convert buffer to tensor
        task_data = torch.stack(self.current_task_data).to(self.device)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(task_data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=min(128, len(task_data)), shuffle=True
        )

        # Train VAE
        self.vae.train()
        total_losses = []

        for epoch in range(self.vae_train_epochs):
            epoch_losses = []

            for batch_data, in dataloader:
                # Generate replay data for VAE training if available
                replay_data = None
                if self.previous_vae is not None:
                    replay_data = self.previous_vae.generate_samples(
                        batch_data.size(0), self.device
                    )

                # Train VAE
                loss_dict = self.vae.train_on_batch(
                    batch_data, replay_data, self.replay_weight
                )
                epoch_losses.append(loss_dict['total_loss'])

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            total_losses.append(avg_loss)

            logging.debug(f"VAE training epoch {epoch + 1}/{self.vae_train_epochs}, "
                         f"avg_loss: {avg_loss:.4f}")

        avg_total_loss = sum(total_losses) / len(total_losses) if total_losses else 0
        logging.info(f"VAE training completed on {len(task_data)} samples, "
                    f"avg_loss: {avg_total_loss:.4f}")

    def _monitor_hybrid_performance(self, epoch: int, loss_dict: Dict[str, float],
                                  replay_inputs: Optional[torch.Tensor]) -> None:
        """Monitor hybrid method performance."""
        if not self.enable_monitoring or not VISUALIZATION_AVAILABLE:
            return

        try:
            # Log performance metrics
            self._log_performance_metrics(epoch, loss_dict)

            # Visualize replay samples if available
            if replay_inputs is not None:
                self._visualize_replay_samples(replay_inputs, epoch)

        except Exception as e:
            logging.warning(f"Failed to monitor hybrid performance: {e}")

    def _monitor_memory_state(self, task_id: int) -> None:
        """Monitor the state of both GPM and DGR memories."""
        if not self.enable_monitoring:
            return

        try:
            # Monitor GPM memory state
            gpm_info = {
                'task_id': task_id,
                'num_bases': len(self.gpm.feature_list),
                'basis_sizes': [basis.shape[1] for basis in self.gpm.feature_list] if self.gpm.feature_list else [],
                'projection_matrices': len(self.gpm.projection_matrices)
            }

            # Monitor DGR memory state
            dgr_info = {
                'task_id': task_id,
                'vae_parameters': sum(p.numel() for p in self.vae.parameters()),
                'has_previous_vae': self.previous_vae is not None,
                'has_previous_classifier': self.previous_classifier is not None,
                'z_dim': self.vae.z_dim
            }

            # Log memory state
            memory_log_file = self.monitor_dir / 'memory_state.txt'
            with open(memory_log_file, 'a') as f:
                f.write(f"Task {task_id}: GPM={gpm_info}, DGR={dgr_info}\n")

            logging.info(f"Memory state logged for task {task_id}")

        except Exception as e:
            logging.warning(f"Failed to monitor memory state: {e}")

    def _log_performance_metrics(self, epoch: int, loss_dict: Dict[str, float]) -> None:
        """Log performance metrics to file."""
        if self.monitor_dir is None:
            return

        metrics_file = self.monitor_dir / 'performance_metrics.txt'
        with open(metrics_file, 'a') as f:
            f.write(f"Epoch {epoch}: {loss_dict}\n")

    def _visualize_replay_samples(self, replay_inputs: torch.Tensor, epoch: int) -> None:
        """Visualize replay samples."""
        if not VISUALIZATION_AVAILABLE or self.monitor_dir is None:
            return

        try:
            samples_np = replay_inputs.detach().cpu().numpy()
            n_samples = min(8, samples_np.shape[0])

            # Create grid visualization
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            fig.suptitle(f'Hybrid Method Replay Samples - Epoch {epoch}')

            for i in range(n_samples):
                row = i // 4
                col = i % 4

                # Convert from CHW to HWC for display
                img = samples_np[i].transpose(1, 2, 0)

                # Handle different image formats
                if img.shape[2] == 1:
                    img = img.squeeze(2)
                    axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
                else:
                    # Ensure RGB values are in [0, 1]
                    img = np.clip(img, 0, 1)
                    axes[row, col].imshow(img)

                axes[row, col].axis('off')
                axes[row, col].set_title(f'Sample {i+1}')

            # Save visualization
            filename = f'hybrid_replay_epoch{epoch:03d}.png'
            filepath = self.monitor_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logging.warning(f"Failed to visualize replay samples: {e}")

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory usage for both components."""
        summary = {
            'gpm': {
                'num_bases': len(self.gpm.feature_list),
                'basis_sizes': [basis.shape for basis in self.gpm.feature_list] if self.gpm.feature_list else [],
                'energy_threshold': self.gpm.energy_threshold
            },
            'dgr': {
                'vae_parameters': sum(p.numel() for p in self.vae.parameters()),
                'z_dim': self.vae.z_dim,
                'has_previous_vae': self.previous_vae is not None
            },
            'hybrid': {
                'replay_weight': self.replay_weight,
                'update_frequency': self.update_frequency,
                'monitoring_enabled': self.enable_monitoring
            }
        }

        return summary


class GPMDGRHybridMammoth(ContinualModel):
    """
    GPM + DGR Hybrid model for Mammoth framework.

    This class integrates the hybrid approach with Mammoth's ContinualModel interface,
    coordinating both GPM gradient projection and DGR generative replay in a single
    training pipeline.
    """

    NAME = 'gpm_dgr_hybrid'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """Add hybrid method arguments."""
        # GPM arguments
        parser.add_argument('--hybrid_gpm_energy_threshold', type=float, default=0.95,
                          help='Energy threshold for GPM basis selection (default: 0.95)')
        parser.add_argument('--hybrid_gpm_max_collection_batches', type=int, default=200,
                          help='Maximum batches for GPM activation collection (default: 200)')
        parser.add_argument('--hybrid_gpm_update_frequency', type=int, default=1,
                          help='Frequency of GPM memory updates (default: 1)')

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

        # Monitoring arguments
        parser.add_argument('--hybrid_enable_monitoring', action='store_true', default=False,
                          help='Enable monitoring and visualization')
        parser.add_argument('--hybrid_monitor_frequency', type=int, default=5,
                          help='Frequency of performance monitoring (default: 5)')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """Initialize the hybrid model."""
        super().__init__(backbone, loss, args, transform, dataset)

        # Get image properties from dataset
        if hasattr(dataset, 'SIZE'):
            self.image_size = dataset.SIZE[-1]  # Assume square images
            self.image_channels = dataset.SIZE[0]
        else:
            # Default values for CIFAR-100
            self.image_size = 32
            self.image_channels = 3

        # Configure GPM component
        gpm_config = {
            'energy_threshold': getattr(args, 'hybrid_gpm_energy_threshold', 0.95),
            'max_collection_batches': getattr(args, 'hybrid_gpm_max_collection_batches', 200),
            'update_frequency': getattr(args, 'hybrid_gpm_update_frequency', 1)
        }

        # Configure DGR component
        dgr_config = {
            'z_dim': getattr(args, 'hybrid_dgr_z_dim', 100),
            'vae_lr': getattr(args, 'hybrid_dgr_vae_lr', 0.001),
            'fc_layers': getattr(args, 'hybrid_dgr_vae_fc_layers', 3),
            'fc_units': getattr(args, 'hybrid_dgr_vae_fc_units', 400),
            'replay_weight': getattr(args, 'hybrid_dgr_replay_weight', 0.5),
            'vae_train_epochs': getattr(args, 'hybrid_dgr_vae_train_epochs', 1),
            'enable_monitoring': getattr(args, 'hybrid_enable_monitoring', False),
            'monitor_frequency': getattr(args, 'hybrid_monitor_frequency', 5)
        }

        # Initialize hybrid method
        self.hybrid_method = GPMDGRHybrid(
            model=self,
            image_size=self.image_size,
            image_channels=self.image_channels,
            gpm_config=gpm_config,
            dgr_config=dgr_config,
            device=self.device
        )

        logging.info(f"GPM+DGR Hybrid model initialized: "
                    f"image_size={self.image_size}x{self.image_channels}, "
                    f"GPM_threshold={gpm_config['energy_threshold']}, "
                    f"DGR_z_dim={dgr_config['z_dim']}")

    def begin_task(self, dataset) -> None:
        """Prepare for new task."""
        super().begin_task(dataset)
        logging.info(f"Hybrid method: Beginning task {self.current_task}")

    def end_task(self, dataset) -> None:
        """Update hybrid memory after task completion."""
        super().end_task(dataset)

        # Create a dummy data loader for memory update
        # In practice, you might want to store actual task data
        if hasattr(dataset, 'train_loader'):
            train_loader = dataset.train_loader
        else:
            # Create a minimal loader if not available
            dummy_data = torch.randn(10, self.image_channels, self.image_size, self.image_size)
            dummy_dataset = torch.utils.data.TensorDataset(dummy_data)
            train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=10)

        self.hybrid_method.end_task(train_loader, self.current_task)
        logging.info(f"Hybrid method: Completed task {self.current_task}")

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        Training step with hybrid GPM+DGR approach.

        Args:
            inputs: Input batch
            labels: Target labels
            not_aug_inputs: Non-augmented inputs
            epoch: Current epoch

        Returns:
            Loss value
        """
        # Use hybrid training step
        loss_dict = self.hybrid_method.training_step(
            real_inputs=inputs,
            real_labels=labels,
            optimizer=self.opt,
            criterion=self.loss,
            epoch=epoch or 0
        )

        return loss_dict['total_loss']

    def forward(self, x):
        """Forward pass through the model."""
        return self.net(x)

    def get_hybrid_summary(self) -> Dict[str, Any]:
        """Get summary of hybrid method state."""
        return self.hybrid_method.get_memory_summary()


# Utility functions for hybrid method configuration and validation

def validate_hybrid_config(gpm_config: Dict[str, Any], dgr_config: Dict[str, Any]) -> bool:
    """
    Validate hybrid method configuration.

    Args:
        gpm_config: GPM configuration dictionary
        dgr_config: DGR configuration dictionary

    Returns:
        True if configuration is valid, False otherwise
    """
    # Validate GPM config
    if not (0.8 <= gpm_config.get('energy_threshold', 0.95) <= 0.99):
        logging.error("GPM energy threshold must be between 0.8 and 0.99")
        return False

    if gpm_config.get('max_collection_batches', 200) <= 0:
        logging.error("GPM max collection batches must be positive")
        return False

    # Validate DGR config
    if dgr_config.get('z_dim', 100) <= 0:
        logging.error("DGR latent dimension must be positive")
        return False

    if not (0.0 <= dgr_config.get('replay_weight', 0.5) <= 1.0):
        logging.error("DGR replay weight must be between 0.0 and 1.0")
        return False

    if dgr_config.get('vae_lr', 0.001) <= 0:
        logging.error("DGR VAE learning rate must be positive")
        return False

    return True


def create_hybrid_config(args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create hybrid method configuration from command line arguments.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (gpm_config, dgr_config)
    """
    gpm_config = {
        'energy_threshold': getattr(args, 'hybrid_gpm_energy_threshold', 0.95),
        'max_collection_batches': getattr(args, 'hybrid_gpm_max_collection_batches', 200),
        'update_frequency': getattr(args, 'hybrid_gpm_update_frequency', 1)
    }

    dgr_config = {
        'z_dim': getattr(args, 'hybrid_dgr_z_dim', 100),
        'vae_lr': getattr(args, 'hybrid_dgr_vae_lr', 0.001),
        'fc_layers': getattr(args, 'hybrid_dgr_vae_fc_layers', 3),
        'fc_units': getattr(args, 'hybrid_dgr_vae_fc_units', 400),
        'replay_weight': getattr(args, 'hybrid_dgr_replay_weight', 0.5),
        'vae_train_epochs': getattr(args, 'hybrid_dgr_vae_train_epochs', 1),
        'enable_monitoring': getattr(args, 'hybrid_enable_monitoring', False),
        'monitor_frequency': getattr(args, 'hybrid_monitor_frequency', 5)
    }

    return gpm_config, dgr_config
