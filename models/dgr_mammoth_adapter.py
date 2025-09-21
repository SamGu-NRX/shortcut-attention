"""
Deep Generative Replay (DGR) Mammoth Adapter

This module bridges the original DGR implementation with the Mammoth continual
learning framework. It loads the released VAE implementation without namespace
collisions and coordinates replay, classifier distillation, and VAE training
inside Mammoth's training loop.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    VISUALIZATION_AVAILABLE = False


class DGRVAE(nn.Module):
    """VAE model used for Deep Generative Replay."""

    def __init__(
        self,
        image_size: int,
        image_channels: int,
        z_dim: int,
        fc_layers: int,
        fc_units: int,
        recon_loss: str = "BCE",
        network_output: str = "sigmoid",
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        device: Optional[torch.device] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.image_height = image_height or image_size
        self.image_width = image_width or image_size
        self.image_channels = image_channels
        self.z_dim = z_dim
        self.fc_layers = max(1, fc_layers)
        self.fc_units = fc_units
        self.recon_loss_type = recon_loss
        self.network_output = network_output
        self.device_override = device or torch.device("cpu")

        input_dim = image_channels * self.image_height * self.image_width
        encoder_layers: List[nn.Module] = []
        prev_dim = input_dim
        for layer_idx in range(self.fc_layers):
            next_dim = fc_units if layer_idx < self.fc_layers - 1 else fc_units
            encoder_layers.append(nn.Linear(prev_dim, next_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = next_dim
        if encoder_layers:
            encoder_layers = encoder_layers[:-1]  # remove last activation to keep output linear
        self.encoder = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()
        encoder_out_dim = prev_dim if encoder_layers else input_dim
        self.mu_layer = nn.Linear(encoder_out_dim, z_dim)
        self.logvar_layer = nn.Linear(encoder_out_dim, z_dim)

        decoder_layers: List[nn.Module] = []
        prev_dim = z_dim
        for layer_idx in range(self.fc_layers):
            next_dim = fc_units if layer_idx < self.fc_layers - 1 else input_dim
            decoder_layers.append(nn.Linear(prev_dim, next_dim))
            if layer_idx < self.fc_layers - 1:
                decoder_layers.append(nn.ReLU(inplace=True))
            prev_dim = next_dim
        self.decoder = nn.Sequential(*decoder_layers) if decoder_layers else nn.Identity()

        if self.network_output == "sigmoid":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()

        self.to(self.device_override)
        self.set_optimizer(lr=lr, betas=betas)
        self.lamda_rcl = 1.0
        self.lamda_vl = 1.0

    def set_optimizer(self, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999)) -> None:
        params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=betas)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(start_dim=1)
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder(z)
        output = self.output_activation(hidden)
        return output.view(z.size(0), self.image_channels, self.image_height, self.image_width)

    def forward(
        self, x: torch.Tensor, full: bool = False, reparameterize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if reparameterize else mu
        recon = self.decode(z)
        if full:
            return recon, mu, logvar, z
        return recon

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.recon_loss_type.upper() == "MSE":
            recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        else:
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kld

    def train_batch(
        self,
        current_inputs: Optional[torch.Tensor],
        replay_inputs: Optional[torch.Tensor],
        rnt: float,
    ) -> Dict[str, float]:
        self.train()
        device = self.device_override
        total_loss = torch.tensor(0.0, device=device)
        recon_cur = variat_cur = recon_rep = variat_rep = torch.tensor(0.0, device=device)

        self.optimizer.zero_grad()

        if current_inputs is not None:
            x = current_inputs.to(device)
            recon, mu, logvar, _ = self(x, full=True)
            recon_cur, variat_cur = self.loss_function(x, recon, mu, logvar)
            loss_cur = self.lamda_rcl * recon_cur + self.lamda_vl * variat_cur
        else:
            loss_cur = torch.tensor(0.0, device=device)

        if replay_inputs is not None:
            x_rep = replay_inputs.to(device)
            recon_rep_t, mu_rep, logvar_rep, _ = self(x_rep, full=True)
            recon_rep, variat_rep = self.loss_function(x_rep, recon_rep_t, mu_rep, logvar_rep)
            loss_rep = self.lamda_rcl * recon_rep + self.lamda_vl * variat_rep
        else:
            loss_rep = torch.tensor(0.0, device=device)

        if current_inputs is None:
            total_loss = loss_rep
        elif replay_inputs is None:
            total_loss = loss_cur
        else:
            total_loss = rnt * loss_cur + (1.0 - rnt) * loss_rep

        total_loss.backward()
        self.optimizer.step()

        return {
            "loss_total": float(total_loss.item()),
            "recon": float(recon_cur.item()),
            "variat": float(variat_cur.item()),
            "recon_r": float(recon_rep.item()),
            "variat_r": float(variat_rep.item()),
        }

    def train_on_batch(
        self,
        x: Optional[torch.Tensor],
        x_replay: Optional[torch.Tensor] = None,
        replay_weight: float = 0.5,
    ) -> Dict[str, float]:
        rnt = 1.0 - float(replay_weight)
        stats = self.train_batch(x, x_replay, rnt)
        return {
            "total_loss": stats["loss_total"],
            "recon_current": stats["recon"],
            "variat_current": stats["variat"],
            "recon_replay": stats["recon_r"],
            "variat_replay": stats["variat_r"],
        }

    def generate_samples(self, n_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device_override
        z = torch.randn(n_samples, self.z_dim, device=device)
        self.eval()
        with torch.no_grad():
            samples = self.decode(z)
        return samples

    def clone_frozen(self) -> "DGRVAE":
        clone: DGRVAE = copy.deepcopy(self).to(self.device_override)
        clone.eval()
        for param in clone.parameters():
            param.requires_grad_(False)
        return clone


class DGRMammothAdapter(ContinualModel):
    """Deep Generative Replay adapted for Mammoth."""

    NAME = "dgr"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--dgr_z_dim", type=int, default=100,
                            help="Latent dimension for the VAE generator")
        parser.add_argument("--dgr_vae_lr", type=float, default=1e-3,
                            help="Learning rate for the VAE optimizer")
        parser.add_argument("--dgr_vae_fc_layers", type=int, default=3,
                            help="Number of fully connected layers in the VAE")
        parser.add_argument("--dgr_vae_fc_units", type=int, default=400,
                            help="Hidden units in each VAE FC layer")
        parser.add_argument("--dgr_replay_targets", type=str, default="hard",
                            choices=["hard", "soft"],
                            help="Use hard labels or distillation targets for replay")
        parser.add_argument("--dgr_distill_temperature", type=float, default=2.0,
                            help="Temperature for knowledge distillation when using soft targets")
        parser.add_argument("--dgr_replay_batch_size", type=int, default=0,
                            help="Batch size for replay samples (0 = match current batch)")
        parser.add_argument("--dgr_monitor_frequency", type=int, default=5,
                            help="Epoch frequency for replay visualizations")
        parser.add_argument("--dgr_disable_monitoring", action="store_true",
                            help="Disable saving replay sample visualizations")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)

        self.image_channels, self.image_size = 3, 32
        self.image_shape: Tuple[int, int, int] = (self.image_channels, self.image_size, self.image_size)

        if dataset is not None:
            if hasattr(dataset, "SIZE"):
                size = dataset.SIZE
                if len(size) >= 3:
                    self.image_channels = int(size[0])
                    height = int(size[-2])
                    width = int(size[-1])
                    self.image_size = max(height, width)
                    self.image_shape = (self.image_channels, height, width)
            elif hasattr(dataset, "get_data_loaders"):
                try:
                    loader, _ = dataset.get_data_loaders()
                    batch = next(iter(loader))
                    sample = batch[0] if isinstance(batch, (list, tuple)) else batch
                    if sample.ndim >= 4:
                        self.image_channels = int(sample.shape[1])
                        height = int(sample.shape[2])
                        width = int(sample.shape[3]) if sample.ndim > 3 else height
                        self.image_size = max(height, width)
                        self.image_shape = (self.image_channels, height, width)
                except Exception:
                    pass

        self.z_dim = getattr(args, "dgr_z_dim", 100)
        self.vae_lr = getattr(args, "dgr_vae_lr", 1e-3)
        self.vae_fc_layers = getattr(args, "dgr_vae_fc_layers", 3)
        self.vae_fc_units = getattr(args, "dgr_vae_fc_units", 400)
        self.replay_targets = getattr(args, "dgr_replay_targets", "hard")
        self.distill_temperature = getattr(args, "dgr_distill_temperature", 2.0)
        self.replay_batch_size = getattr(args, "dgr_replay_batch_size", 0)

        self.replay_weight = getattr(args, "dgr_replay_weight", 0.5)
        self.vae_train_epochs = getattr(args, "dgr_vae_train_epochs", 1)
        self.buffer_size = getattr(args, "dgr_buffer_size", getattr(args, "buffer_size", 0))

        height, width = self.image_shape[1], self.image_shape[2]
        self.generator = DGRVAE(
            image_size=self.image_size,
            image_channels=self.image_channels,
            z_dim=self.z_dim,
            fc_layers=self.vae_fc_layers,
            fc_units=self.vae_fc_units,
            lr=self.vae_lr,
            device=self.device,
            image_height=height,
            image_width=width,
        )
        self.prev_generator: Optional[DGRVAE] = None
        self.prev_classifier: Optional[nn.Module] = None
        # Backwards compatibility aliases expected by existing tests/utilities.
        self.vae = self.generator
        self.previous_vae: Optional[DGRVAE] = None
        self.current_vae: Optional[DGRVAE] = self.generator
        self.current_task_buffer: List[torch.Tensor] = []

        self.enable_monitoring = not getattr(args, "dgr_disable_monitoring", False)
        self.monitor_frequency = getattr(args, "dgr_monitor_frequency", 5)
        self.monitor_dir: Optional[Path] = None
        if self.enable_monitoring and VISUALIZATION_AVAILABLE:
            self.monitor_dir = Path("outputs") / "dgr_monitoring"
            self.monitor_dir.mkdir(parents=True, exist_ok=True)
            logging.info("DGR monitoring enabled: %s", self.monitor_dir)

        self._last_generator_losses: Dict[str, float] = {}

        logging.info(
            "Initialized DGR (z_dim=%d, image=%dx%d, replay_targets=%s)",
            self.z_dim,
            self.image_size,
            self.image_channels,
            self.replay_targets,
        )

    # ------------------------------------------------------------------
    # Training flow
    # ------------------------------------------------------------------

    def begin_task(self, dataset) -> None:
        previous = getattr(self, "current_vae", None)
        if previous is not None:
            if hasattr(previous, "clone_frozen"):
                self.prev_generator = previous.clone_frozen()
                self.previous_vae = self.prev_generator
            else:
                self.previous_vae = previous

        height, width = self.image_shape[1], self.image_shape[2]
        self.generator = DGRVAE(
            image_size=self.image_size,
            image_channels=self.image_channels,
            z_dim=self.z_dim,
            fc_layers=self.vae_fc_layers,
            fc_units=self.vae_fc_units,
            lr=self.vae_lr,
            device=self.device,
            image_height=height,
            image_width=width,
        ).to(self.device)
        self.current_vae = self.generator
        self.vae = self.generator
        self.vae = self.generator

        super().begin_task(dataset)
        self.generator.train()

    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        not_aug_inputs: torch.Tensor,
        epoch: Optional[int] = None,
        **unused,
    ) -> float:
        batch_size = inputs.size(0)
        rnt = 1.0 / float(max(1, self.current_task + 1))

        replay_inputs, replay_labels, replay_logits = self._generate_replay_batch(batch_size)

        self.opt.zero_grad()
        current_logits = self.net(inputs)
        loss_current = self.loss(current_logits, labels)
        total_loss = loss_current

        if replay_inputs is not None:
            replay_logits_current = self.net(replay_inputs)
            loss_replay = self._compute_replay_loss(
                replay_logits_current, replay_labels, replay_logits
            )
            total_loss = rnt * loss_current + (1.0 - rnt) * loss_replay
        total_loss.backward()
        self.opt.step()

        if self.buffer_size > 0 and not_aug_inputs is not None:
            remaining = self.buffer_size - len(self.current_task_buffer)
            if remaining > 0:
                self.current_task_buffer.extend([x.cpu() for x in not_aug_inputs[:remaining]])

        generator_current = not_aug_inputs if not_aug_inputs is not None else inputs.detach()
        generator_current = generator_current.detach()
        generator_replay = replay_inputs.detach() if replay_inputs is not None else None
        self._last_generator_losses = self.generator.train_batch(generator_current, generator_replay, rnt)

        if self.enable_monitoring and epoch is not None:
            self._maybe_monitor(epoch, replay_inputs)

        return float(total_loss.item())

    def end_task(self, dataset) -> None:
        super().end_task(dataset)
        if self.current_task_buffer:
            self._train_vae_on_current_task()

        self.prev_classifier = copy.deepcopy(self.net).to(self.device)
        self.prev_classifier.eval()
        for param in self.prev_classifier.parameters():
            param.requires_grad_(False)

        self.prev_generator = self.generator.clone_frozen()
        self.previous_vae = self.prev_generator
        logging.info("DGR: stored frozen generator and classifier for replay")

    # ------------------------------------------------------------------
    # Replay helpers
    # ------------------------------------------------------------------

    def _generate_replay_batch(
        self,
        batch_size: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.prev_generator is None or self.prev_classifier is None:
            return None, None, None

        replay_bs = batch_size if self.replay_batch_size <= 0 else self.replay_batch_size
        with torch.no_grad():
            replay_inputs = self.prev_generator.generate_samples(replay_bs, device=self.device)
            prev_outputs = self.prev_classifier(replay_inputs)
            replay_logits = prev_outputs.detach()
            replay_labels = prev_outputs.argmax(dim=1)
        return replay_inputs, replay_labels, replay_logits

    def _compute_replay_loss(
        self,
        replay_logits_current: torch.Tensor,
        replay_labels: Optional[torch.Tensor],
        replay_logits_reference: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.replay_targets == "soft":
            assert replay_logits_reference is not None
            temperature = float(self.distill_temperature)
            kd_loss = F.kl_div(
                F.log_softmax(replay_logits_current / temperature, dim=1),
                F.softmax(replay_logits_reference / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
            return kd_loss

        assert replay_labels is not None
        return self.loss(replay_logits_current, replay_labels)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def _train_vae_on_current_task(self) -> None:
        if not self.current_task_buffer:
            return

        tensor_data = torch.stack(self.current_task_buffer).to(self.device)
        dataset = data.TensorDataset(tensor_data)
        dataloader = data.DataLoader(dataset, batch_size=min(128, len(dataset)), shuffle=True)

        for _ in range(self.vae_train_epochs):
            for batch, in dataloader:
                replay = None
                if self.previous_vae is not None:
                    replay = self.previous_vae.generate_samples(batch.size(0), device=self.device)
                self.vae.train_on_batch(batch, replay, self.replay_weight)

        self.current_task_buffer.clear()

    def _maybe_monitor(self, epoch: int, replay_inputs: Optional[torch.Tensor]) -> None:
        if not VISUALIZATION_AVAILABLE or self.monitor_dir is None:
            return
        if epoch % self.monitor_frequency != 0:
            return
        if replay_inputs is None:
            return
        try:
            samples_np = replay_inputs.detach().cpu().numpy()
            n_samples = min(8, samples_np.shape[0])
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            fig.suptitle(f"DGR Replay Samples - Epoch {epoch}")
            for idx in range(n_samples):
                row, col = divmod(idx, 4)
                img = samples_np[idx].transpose(1, 2, 0)
                if img.shape[2] == 1:
                    axes[row, col].imshow(img.squeeze(2), cmap="gray", vmin=0, vmax=1)
                else:
                    axes[row, col].imshow(np.clip(img, 0, 1))
                axes[row, col].axis("off")
                axes[row, col].set_title(f"Sample {idx + 1}")
            for idx in range(n_samples, 8):
                row, col = divmod(idx, 4)
                axes[row, col].axis("off")
            fig.tight_layout()
            outfile = self.monitor_dir / f"replay_epoch_{epoch:04d}.png"
            plt.savefig(outfile, dpi=150)
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - monitoring is best effort
            logging.warning("Failed to visualize DGR replay samples: %s", exc)

    # ------------------------------------------------------------------
    # Auxiliary information
    # ------------------------------------------------------------------

    def get_generator_stats(self) -> Dict[str, Any]:
        return {
            "last_loss": self._last_generator_losses.get("loss_total", 0.0),
            "recon": self._last_generator_losses.get("recon", 0.0),
            "recon_replay": self._last_generator_losses.get("recon_r", 0.0),
            "has_prev_generator": self.prev_generator is not None,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["DGRMammothAdapter", "DGRVAE"]
