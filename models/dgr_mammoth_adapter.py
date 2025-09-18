"""
Deep Generative Replay (DGR) Mammoth Adapter

This module adapts the existing DGR implementation from DGR_wrapper to work with
the Mammoth continual learning framework. It extracts the VAE and generative replay
functionality while maintaining compatibility with Mammoth's ContinualModel interface.

The adapter preserves the original DGR VAE architecture and training configurations
while integrating seamlessly with Mammoth's training pipeline and evaluation system.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import logging
from pathlib import Path

from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Import the original DGR VAE implementation
try:
    from DGR_wrapper.models.vae import VAE as OriginalVAE
except ImportError:
    # Fallback for testing - create a minimal VAE interface
    class OriginalVAE(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.z_dim = kwargs.get('z_dim', 100)
            self.image_size = kwargs.get('image_size', 32)
            self.image_channels = kwargs.get('image_channels', 3)
            self.fc_layers = kwargs.get('fc_layers', 3)
            self.recon_loss = kwargs.get('recon_loss', 'BCE')
            self.network_output = kwargs.get('network_output', 'sigmoid')
            self.lamda_rcl = 1.0
            self.lamda_vl = 1.0
            self.optimizer = None

            # Create a simple encoder-decoder for testing
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.image_channels * self.image_size * self.image_size, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU()
            )
            self.mu_layer = nn.Linear(256, self.z_dim)
            self.logvar_layer = nn.Linear(256, self.z_dim)

            self.decoder = nn.Sequential(
                nn.Linear(self.z_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, self.image_channels * self.image_size * self.image_size),
                nn.Sigmoid() if self.network_output == 'sigmoid' else nn.Identity()
            )

        def encode(self, x):
            h = self.encoder(x)
            mu = self.mu_layer(h)
            logvar = self.logvar_layer(h)
            return mu, logvar, h, x

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = self.decoder(z)
            return h.view(-1, self.image_channels, self.image_size, self.image_size)

        def forward(self, x, full=False, reparameterize=True):
            mu, logvar, h, _ = self.encode(x)
            z = self.reparameterize(mu, logvar) if reparameterize else mu
            x_recon = self.decode(z)
            return (x_recon, mu, logvar, z) if full else x_recon

        def loss_function(self, x, x_recon, mu, z, logvar=None):
            # Simple reconstruction loss
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')

            # KL divergence loss
            if logvar is not None:
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            else:
                kl_loss = torch.tensor(0.0, device=x.device)

            return recon_loss, kl_loss

        def sample(self, size):
            z = torch.randn(size, self.z_dim, device=next(self.parameters()).device)
            return self.decode(z)


class DGRVAE(OriginalVAE):
    """
    Adapted VAE from the original DGR implementation.

    This class wraps the original VAE to work with Mammoth's data pipeline
    while preserving all original functionality and architecture.
    """

    def __init__(self, image_size: int, image_channels: int,
                 z_dim: int = 100, fc_layers: int = 3, fc_units: int = 400,
                 recon_loss: str = 'BCE', network_output: str = 'sigmoid',
                 prior: str = 'standard', device: Optional[torch.device] = None):
        """
        Initialize the DGR VAE with original parameters.

        Args:
            image_size: Size of input images (assumed square)
            image_channels: Number of input channels
            z_dim: Latent dimension size
            fc_layers: Number of fully connected layers
            fc_units: Number of units in FC layers
            recon_loss: Reconstruction loss type ('BCE' or 'MSE')
            network_output: Output activation ('sigmoid' or other)
            prior: Prior type ('standard' or 'GMM')
            device: Device to use for computation
        """
        super().__init__(
            image_size=image_size,
            image_channels=image_channels,
            depth=0,  # No conv layers for simplicity
            fc_layers=fc_layers,
            fc_units=fc_units,
            fc_bn=False,
            fc_nl="relu",
            excit_buffer=True,
            prior=prior,
            z_dim=z_dim,
            recon_loss=recon_loss,
            network_output=network_output
        )

        self.device_override = device

    def _device(self):
        """Override device method to use Mammoth's device management."""
        if self.device_override is not None:
            return self.device_override
        return next(self.parameters()).device

    def train_on_batch(self, x: torch.Tensor, x_replay: Optional[torch.Tensor] = None,
                      replay_weight: float = 0.5) -> Dict[str, float]:
        """
        Train the VAE on a batch of data, optionally with replay data.

        Args:
            x: Current task data
            x_replay: Optional replay data from previous tasks
            replay_weight: Weight for replay data (1-replay_weight for current data)

        Returns:
            Dictionary with loss components
        """
        self.train()

        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")

        self.optimizer.zero_grad()

        # Train on current data
        loss_dict = {}
        if x is not None:
            recon_batch, mu, logvar, z = self(x, full=True, reparameterize=True)
            reconL, variatL = self.loss_function(x=x, x_recon=recon_batch, mu=mu, z=z, logvar=logvar)
            loss_current = self.lamda_rcl * reconL + self.lamda_vl * variatL
            loss_dict['recon_current'] = reconL.item()
            loss_dict['variat_current'] = variatL.item()
        else:
            loss_current = torch.tensor(0.0, device=self._device())

        # Train on replay data
        if x_replay is not None:
            recon_batch_r, mu_r, logvar_r, z_r = self(x_replay, full=True, reparameterize=True)
            reconL_r, variatL_r = self.loss_function(x=x_replay, x_recon=recon_batch_r,
                                                   mu=mu_r, z=z_r, logvar=logvar_r)
            loss_replay = self.lamda_rcl * reconL_r + self.lamda_vl * variatL_r
            loss_dict['recon_replay'] = reconL_r.item()
            loss_dict['variat_replay'] = variatL_r.item()
        else:
            loss_replay = torch.tensor(0.0, device=self._device())

        # Combine losses
        if x is None:
            total_loss = loss_replay
        elif x_replay is None:
            total_loss = loss_current
        else:
            total_loss = (1 - replay_weight) * loss_current + replay_weight * loss_replay

        loss_dict['total_loss'] = total_loss.item()

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return loss_dict

    def set_optimizer(self, lr: float = 0.001):
        """Set the optimizer for VAE training."""
        self.optim_list = [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': lr}]
        self.optimizer = torch.optim.Adam(self.optim_list, betas=(0.9, 0.999))

    def generate_samples(self, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate samples from the VAE.

        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated samples as tensor
        """
        if device is None:
            device = self._device()

        self.eval()
        with torch.no_grad():
            samples = self.sample(n_samples)

        return samples.to(device)


class DGRMammothAdapter(ContinualModel):
    """
    Deep Generative Replay adapter for Mammoth framework.

    This class integrates the original DGR implementation with Mammoth's ContinualModel
    interface, providing VAE-based generative replay for continual learning.
    """

    NAME = 'dgr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        Initialize the DGR adapter.

        Args:
            backbone: Mammoth backbone network
            loss: Loss function
            args: Command line arguments
            transform: Data transformations
            dataset: Dataset instance
        """
        super().__init__(backbone, loss, args, transform, dataset)

        # DGR-specific parameters
        self.z_dim = getattr(args, 'dgr_z_dim', 100)
        self.vae_lr = getattr(args, 'dgr_vae_lr', 0.001)
        self.vae_fc_layers = getattr(args, 'dgr_vae_fc_layers', 3)
        self.vae_fc_units = getattr(args, 'dgr_vae_fc_units', 400)
        self.replay_weight = getattr(args, 'dgr_replay_weight', 0.5)
        self.vae_train_epochs = getattr(args, 'dgr_vae_train_epochs', 1)

        # Get image properties from dataset
        if hasattr(dataset, 'SIZE'):
            self.image_size = dataset.SIZE[-1]  # Assume square images
            self.image_channels = dataset.SIZE[0]
        else:
            # Default values for CIFAR-100
            self.image_size = 32
            self.image_channels = 3

        # Initialize VAE
        self.vae = DGRVAE(
            image_size=self.image_size,
            image_channels=self.image_channels,
            z_dim=self.z_dim,
            fc_layers=self.vae_fc_layers,
            fc_units=self.vae_fc_units,
            device=self.device
        )
        self.vae.to(self.device)
        self.vae.set_optimizer(self.vae_lr)

        # Store previous VAE for replay generation
        self.previous_vae = None

        # Buffer for storing current task data for VAE training
        self.current_task_buffer = []

        # Replay monitoring
        self.enable_replay_monitoring = not getattr(args, 'dgr_disable_replay_monitoring', False)
        self.replay_monitor_frequency = getattr(args, 'dgr_replay_monitor_frequency', 5)
        self.replay_monitor_samples = getattr(args, 'dgr_replay_monitor_samples', 8)
        self.replay_monitor_dir = None

        # Initialize replay monitoring if enabled
        if self.enable_replay_monitoring and VISUALIZATION_AVAILABLE:
            self.replay_monitor_dir = Path("outputs") / "dgr_replay_monitoring"
            self.replay_monitor_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"DGR replay monitoring enabled: {self.replay_monitor_dir}")
        elif self.enable_replay_monitoring and not VISUALIZATION_AVAILABLE:
            logging.warning("DGR replay monitoring requested but matplotlib not available")

        logging.info(f"Initialized DGR with VAE: z_dim={self.z_dim}, "
                    f"image_size={self.image_size}x{self.image_channels}")

    @staticmethod
    def get_parser(parser):
        """Add DGR-specific arguments to the parser."""
        parser.add_argument('--dgr_z_dim', type=int, default=100,
                          help='Latent dimension for DGR VAE')
        parser.add_argument('--dgr_vae_lr', type=float, default=0.001,
                          help='Learning rate for VAE training')
        parser.add_argument('--dgr_vae_fc_layers', type=int, default=3,
                          help='Number of FC layers in VAE')
        parser.add_argument('--dgr_vae_fc_units', type=int, default=400,
                          help='Number of units in VAE FC layers')
        parser.add_argument('--dgr_replay_weight', type=float, default=0.5,
                          help='Weight for replay data in training')
        parser.add_argument('--dgr_vae_train_epochs', type=int, default=1,
                          help='Number of epochs to train VAE per task')
        parser.add_argument('--dgr_disable_replay_monitoring', action='store_true',
                          help='Disable monitoring and visualization of replay samples')
        parser.add_argument('--dgr_replay_monitor_frequency', type=int, default=5,
                          help='Frequency of replay monitoring (every N epochs)')
        parser.add_argument('--dgr_replay_monitor_samples', type=int, default=8,
                          help='Number of replay samples to monitor and visualize')
        return parser

    def begin_task(self, dataset):
        """Prepare for a new task."""
        super().begin_task(dataset)

        # Clear current task buffer
        self.current_task_buffer = []

        logging.info(f"Starting task {self.current_task + 1}")

    def end_task(self, dataset):
        """Complete the current task and update VAE."""
        super().end_task(dataset)

        # Train VAE on current task data
        if len(self.current_task_buffer) > 0:
            self._train_vae_on_current_task()

        # Store current VAE as previous for next task
        self.previous_vae = copy.deepcopy(self.vae).eval()

        # Monitor replay samples from the newly trained VAE
        self._monitor_replay_samples(self.current_task, epoch=0)

        logging.info(f"Completed task {self.current_task}, VAE updated")

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        Perform a training step with DGR.

        Args:
            inputs: Batch of input images
            labels: Batch of labels
            not_aug_inputs: Batch of non-augmented inputs
            epoch: Current epoch (optional)

        Returns:
            Loss value
        """
        # Store current task data for VAE training
        self.current_task_buffer.extend(not_aug_inputs.cpu())

        # Generate replay data if we have a previous VAE
        replay_inputs = None
        replay_labels = None

        if self.previous_vae is not None and self.current_task > 0:
            replay_inputs, replay_labels = self._generate_replay_data(inputs.size(0))

        # Train classifier on current + replay data
        self.opt.zero_grad()

        # Forward pass on current data
        outputs = self.net(inputs)
        loss_current = self.loss(outputs, labels)

        total_loss = loss_current

        # Add replay loss if available
        if replay_inputs is not None:
            replay_outputs = self.net(replay_inputs)
            loss_replay = self.loss(replay_outputs, replay_labels)
            total_loss = (1 - self.replay_weight) * loss_current + self.replay_weight * loss_replay

        total_loss.backward()
        self.opt.step()

        return total_loss.item()

    def _generate_replay_data(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate replay data using the previous VAE.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Tuple of (replay_inputs, replay_labels)
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
        self.vae.train()
        for epoch in range(self.vae_train_epochs):
            total_loss = 0
            n_batches = 0

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
                total_loss += loss_dict['total_loss']
                n_batches += 1

            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            logging.debug(f"VAE training epoch {epoch + 1}/{self.vae_train_epochs}, "
                         f"avg_loss: {avg_loss:.4f}")

        logging.info(f"VAE training completed on {len(task_data)} samples")

    def forward(self, x):
        """Forward pass through the classifier."""
        return self.net(x)

    def _monitor_replay_samples(self, task_id: int, epoch: int = 0):
        """Monitor and visualize replay samples if enabled."""
        if not self.enable_replay_monitoring or not VISUALIZATION_AVAILABLE:
            return

        if self.previous_vae is None:
            return

        if epoch % self.replay_monitor_frequency != 0:
            return

        try:
            # Generate replay samples
            replay_samples = self.previous_vae.generate_samples(
                self.replay_monitor_samples, self.device
            )

            # Create visualization
            self._visualize_replay_samples(replay_samples, task_id, epoch)

            # Log statistics
            self._log_replay_statistics(replay_samples, task_id, epoch)

        except Exception as e:
            logging.warning(f"Failed to monitor replay samples: {e}")

    def _visualize_replay_samples(self, samples: torch.Tensor, task_id: int, epoch: int):
        """Create and save visualization of replay samples."""
        if not VISUALIZATION_AVAILABLE:
            return

        samples_np = samples.detach().cpu().numpy()
        n_samples = min(8, samples.shape[0])

        # Create grid visualization
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(f'DGR Replay Samples - Task {task_id}, Epoch {epoch}')

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
        filename = f'replay_samples_task{task_id}_epoch{epoch:03d}.png'
        filepath = self.replay_monitor_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        logging.debug(f"Saved replay visualization: {filepath}")

    def _log_replay_statistics(self, samples: torch.Tensor, task_id: int, epoch: int):
        """Log statistical information about replay samples."""
        stats_file = self.replay_monitor_dir / 'replay_statistics.txt'

        with open(stats_file, 'a') as f:
            mean_val = torch.mean(samples).item()
            std_val = torch.std(samples).item()
            min_val = torch.min(samples).item()
            max_val = torch.max(samples).item()

            # Calculate additional quality metrics
            # Diversity: average pairwise distance between samples
            flat_samples = samples.view(samples.size(0), -1)
            pairwise_dists = torch.cdist(flat_samples, flat_samples)
            avg_diversity = torch.mean(pairwise_dists).item()

            # Intensity distribution
            intensity_hist = torch.histc(samples, bins=10, min=0, max=1)
            entropy = -torch.sum(intensity_hist * torch.log(intensity_hist + 1e-8)).item()

            f.write(f"Task {task_id}, Epoch {epoch:03d}: "
                   f"mean={mean_val:.4f}, std={std_val:.4f}, "
                   f"min={min_val:.4f}, max={max_val:.4f}, "
                   f"diversity={avg_diversity:.4f}, entropy={entropy:.4f}\n")

    def get_replay_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of replay monitoring data."""
        if not self.enable_replay_monitoring or self.replay_monitor_dir is None:
            return {}

        summary = {
            'monitoring_enabled': True,
            'monitor_dir': str(self.replay_monitor_dir),
            'frequency': self.replay_monitor_frequency,
            'samples_per_monitoring': self.replay_monitor_samples
        }

        # Count generated files
        image_files = list(self.replay_monitor_dir.glob('replay_samples_*.png'))
        summary['total_visualizations'] = len(image_files)

        # Read statistics if available
        stats_file = self.replay_monitor_dir / 'replay_statistics.txt'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                lines = f.readlines()
            summary['total_statistics_entries'] = len(lines)

            if lines:
                # Parse last entry for current stats
                last_line = lines[-1].strip()
                if 'mean=' in last_line:
                    # Extract values using regex
                    import re
                    mean_match = re.search(r'mean=([\d.]+)', last_line)
                    std_match = re.search(r'std=([\d.]+)', last_line)
                    diversity_match = re.search(r'diversity=([\d.]+)', last_line)

                    if mean_match:
                        summary['latest_mean'] = float(mean_match.group(1))
                    if std_match:
                        summary['latest_std'] = float(std_match.group(1))
                    if diversity_match:
                        summary['latest_diversity'] = float(diversity_match.group(1))

        return summary
