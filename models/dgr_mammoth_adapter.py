"""
Deep Generative Replay (DGR) Mammoth Adapter

This module bridges the original DGR implementation with the Mammoth continual
learning framework. It loads the released VAE implementation without namespace
collisions and coordinates replay, classifier distillation, and VAE training
inside Mammoth's training loop.
"""

from __future__ import annotations

import copy
import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType
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


def _ensure_dgr_imports() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    dgr_root = repo_root / "DGR"
    if not dgr_root.exists():  # pragma: no cover - sanity check
        raise ImportError(f"Original DGR repository not found at {dgr_root}")
    if str(dgr_root) not in sys.path:
        sys.path.insert(0, str(dgr_root))
    return dgr_root


def _import_dgr_module(module_name: str):
    repo_root = _ensure_dgr_imports()

    preserved = {name: sys.modules[name] for name in list(sys.modules)
                 if name == "models" or name.startswith("models.") or name == "utils" or name.startswith("utils.")}
    for name in preserved:
        sys.modules.pop(name)

    shim_utils = ModuleType("utils")

    def _checkattr(args, attr):
        return hasattr(args, attr) and isinstance(getattr(args, attr), bool) and getattr(args, attr)

    def _get_data_loader(dataset, batch_size, cuda=False, drop_last=False, augment=False):
        loader_kwargs = {}
        if cuda:
            loader_kwargs.update({'num_workers': 0, 'pin_memory': True})
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            **loader_kwargs,
        )

    shim_utils.checkattr = _checkattr
    shim_utils.get_data_loader = _get_data_loader

    sys.modules['utils'] = shim_utils

    sys.path.insert(0, str(repo_root))
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.path.remove(str(repo_root))
        sys.modules.pop('utils', None)
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                sys.modules.pop(name)
        for name, mod in preserved.items():
            sys.modules[name] = mod

    return module


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
        parser.add_argument(
            "--dgr_z_dim",
            "--dgr-z-dim",
            type=int,
            default=100,
            help="Latent dimension for the VAE generator",
        )
        parser.add_argument(
            "--dgr_vae_lr",
            "--dgr-vae-lr",
            type=float,
            default=1e-3,
            help="Learning rate for the VAE optimizer",
        )
        parser.add_argument(
            "--dgr_vae_fc_layers",
            "--dgr-vae-fc-layers",
            type=int,
            default=3,
            help="Number of fully connected layers in the VAE",
        )
        parser.add_argument(
            "--dgr_vae_fc_units",
            "--dgr-vae-fc-units",
            type=int,
            default=400,
            help="Hidden units in each VAE FC layer",
        )
        parser.add_argument(
            "--dgr_replay_weight",
            "--dgr-replay-ratio",
            type=float,
            default=0.5,
            dest="dgr_replay_weight",
            help="Replay mixing weight (alias: --dgr-replay-ratio)",
        )
        parser.add_argument(
            "--dgr_replay_targets",
            "--dgr-replay-targets",
            type=str,
            default="hard",
            choices=["hard", "soft"],
            help="Use hard labels or distillation targets for replay",
        )
        parser.add_argument(
            "--dgr_distill_temperature",
            "--dgr-temperature",
            type=float,
            default=2.0,
            help="Temperature for knowledge distillation when using soft targets",
        )
        parser.add_argument(
            "--dgr_replay_batch_size",
            "--dgr-replay-batch-size",
            type=int,
            default=0,
            help="Batch size for replay samples (0 = match current batch)",
        )
        parser.add_argument(
            "--dgr_monitor_frequency",
            "--dgr-monitor-frequency",
            type=int,
            default=5,
            help="Epoch frequency for replay visualizations",
        )
        parser.add_argument(
            "--dgr_disable_monitoring",
            "--dgr-disable-monitoring",
            action="store_true",
            help="Disable saving replay sample visualizations",
        )
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)

        self.total_classes = self.dataset.N_CLASSES
        if isinstance(self.dataset.N_CLASSES_PER_TASK, int):
            self._classes_per_task_seq = [self.dataset.N_CLASSES_PER_TASK for _ in range(self.dataset.N_TASKS)]
        else:
            self._classes_per_task_seq = list(self.dataset.N_CLASSES_PER_TASK)
        self._class_offsets: List[Tuple[int, int]] = []
        start = 0
        for count in self._classes_per_task_seq:
            self._class_offsets.append((start, start + count))
            start += count

        self.image_channels, self.image_size = 3, 32
        self.image_shape: Tuple[int, int, int] = (self.image_channels, self.image_size, self.image_size)

        if dataset is not None and hasattr(dataset, "SIZE"):
            size = tuple(dataset.SIZE)
            if len(size) >= 3:
                self.image_channels = int(size[0])
                height = int(size[-2])
                width = int(size[-1])
                self.image_size = max(height, width)
                self.image_shape = (self.image_channels, height, width)

        if hasattr(self.dataset, "MEAN") and hasattr(self.dataset, "STD"):
            mean = torch.tensor(self.dataset.MEAN, dtype=torch.float32).view(1, self.image_channels, 1, 1)
            std = torch.tensor(self.dataset.STD, dtype=torch.float32).view(1, self.image_channels, 1, 1)
            self.register_buffer("_data_mean", mean)
            self.register_buffer("_data_std", std)
        else:
            self._data_mean = None  # type: ignore[assignment]
            self._data_std = None   # type: ignore[assignment]

        self.z_dim = getattr(args, "dgr_z_dim", 100)
        self.vae_lr = getattr(args, "dgr_vae_lr", 1e-3)
        self.vae_fc_layers = getattr(args, "dgr_vae_fc_layers", 3)
        self.vae_fc_units = getattr(args, "dgr_vae_fc_units", 400)
        self.replay_targets = getattr(args, "dgr_replay_targets", "hard")
        self.distill_temperature = getattr(args, "dgr_distill_temperature", 2.0)
        self.replay_batch_size = getattr(args, "dgr_replay_batch_size", 0)

        replay_weight = getattr(args, "dgr_replay_weight", None)
        if replay_weight is None:
            replay_weight = getattr(args, "dgr_replay_ratio", 0.5)
        else:
            setattr(args, "dgr_replay_ratio", replay_weight)
        self.replay_weight = float(replay_weight)
        self.vae_train_epochs = getattr(args, "dgr_vae_train_epochs", 1)
        self.buffer_size = getattr(args, "dgr_buffer_size", getattr(args, "buffer_size", 0))

        if self.__class__._COND_VAE_CLS is None:
            cond_module = _import_dgr_module("models.cond_vae")
            self.__class__._COND_VAE_CLS = cond_module.CondVAE

        self.generator = self._build_generator()

        self.prev_generator: Optional[nn.Module] = None
        self.prev_classifier: Optional[nn.Module] = None
        # Backwards compatibility aliases expected by existing tests/utilities.
        self.vae = self.generator
        self.previous_vae: Optional[nn.Module] = None
        self.current_vae: Optional[nn.Module] = self.generator
        self.current_task_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.seen_classes: List[int] = []

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

    def _has_normalization(self) -> bool:
        return isinstance(getattr(self, "_data_mean", None), torch.Tensor) and isinstance(
            getattr(self, "_data_std", None), torch.Tensor
        )

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._has_normalization():
            return tensor
        mean = self._data_mean.to(tensor.device)  # type: ignore[union-attr]
        std = self._data_std.to(tensor.device)    # type: ignore[union-attr]
        return (tensor - mean) / std

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._has_normalization():
            return tensor
        mean = self._data_mean.to(tensor.device)  # type: ignore[union-attr]
        std = self._data_std.to(tensor.device)    # type: ignore[union-attr]
        return torch.clamp(tensor * std + mean, 0.0, 1.0)

    def _active_classes(self) -> List[List[int]]:
        active: List[List[int]] = []
        for idx in range(self.current_task + 1):
            start, end = self._class_offsets[idx]
            active.append(list(range(start, end)))
        return active

    def _update_seen_classes(self, labels: torch.Tensor) -> None:
        unique = torch.unique(labels).tolist()
        for cls in unique:
            if cls not in self.seen_classes:
                self.seen_classes.append(cls)
        self.seen_classes.sort()

    def begin_task(self, dataset) -> None:
        previous = getattr(self, "current_vae", None)
        if previous is not None:
            self.prev_generator = self._freeze_generator(previous)
            self.previous_vae = self.prev_generator

        self.generator = self._build_generator()
        self.current_vae = self.generator
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

        inputs = inputs.to(self.device)
        labels = labels.to(self.device, dtype=torch.long)
        inputs_raw = self._denormalize(inputs.detach())

        replay_raw, replay_norm, replay_labels, replay_logits = self._generate_replay_batch(batch_size)

        self.opt.zero_grad()
        current_logits = self.net(inputs)
        loss_current = self.loss(current_logits, labels)
        total_loss = loss_current

        if replay_norm is not None:
            replay_logits_current = self.net(replay_norm)
            loss_replay = self._compute_replay_loss(
                replay_logits_current, replay_labels, replay_logits
            )
            total_loss = rnt * loss_current + (1.0 - rnt) * loss_replay
        total_loss.backward()
        self.opt.step()

        self._update_seen_classes(labels)

        if self.buffer_size > 0 and not_aug_inputs is not None:
            remaining = self.buffer_size - len(self.current_task_buffer)
            if remaining > 0:
                imgs = not_aug_inputs[:remaining].detach().cpu()
                labs = labels[:remaining].detach().cpu()
                self.current_task_buffer.extend([(img, lab) for img, lab in zip(imgs, labs)])

        generator_current = inputs_raw
        generator_replay = replay_raw if replay_raw is not None else None
        generator_replay_labels = replay_labels if replay_labels is not None else None
        generator_scores = replay_logits if (self.replay_targets == "soft") else None
        self._last_generator_losses = self.generator.train_a_batch(
            generator_current,
            y=labels.detach(),
            x_=generator_replay,
            y_=generator_replay_labels,
            scores_=generator_scores,
            rnt=rnt,
            context=self.current_task + 1,
        )

        if self.enable_monitoring and epoch is not None:
            self._maybe_monitor(epoch, replay_raw)

        return float(total_loss.item())

    def end_task(self, dataset) -> None:
        super().end_task(dataset)
        if self.current_task_buffer:
            self._train_vae_on_current_task()

        self.prev_classifier = copy.deepcopy(self.net).to(self.device)
        self.prev_classifier.eval()
        for param in self.prev_classifier.parameters():
            param.requires_grad_(False)

        self.prev_generator = self._freeze_generator(self.generator)
        self.previous_vae = self.prev_generator
        logging.info("DGR: stored frozen generator and classifier for replay")

    # ------------------------------------------------------------------
    # Replay helpers
    # ------------------------------------------------------------------

    def _generate_replay_batch(
        self,
        batch_size: int,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if self.prev_generator is None or self.prev_classifier is None:
            return None, None, None, None

        replay_bs = batch_size if self.replay_batch_size <= 0 else self.replay_batch_size
        allowed = self.seen_classes if self.seen_classes else None
        with torch.no_grad():
            sample = self.prev_generator.sample(replay_bs, allowed_classes=allowed, only_x=False)
        if isinstance(sample, tuple):
            replay_raw, y_used, _ = sample
        else:
            replay_raw = sample
            y_used = None

        replay_raw = replay_raw.to(self.device)
        if y_used is not None:
            replay_labels = torch.tensor(y_used, device=self.device, dtype=torch.long)
        else:
            replay_labels = None

        replay_norm = self._normalize(replay_raw)
        with torch.no_grad():
            replay_logits = self.prev_classifier(replay_norm)
        if replay_labels is None:
            replay_labels = replay_logits.argmax(dim=1)

        return replay_raw.detach(), replay_norm.detach(), replay_labels.detach(), replay_logits.detach()

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

    def _build_generator(self) -> nn.Module:
        generator = self.__class__._COND_VAE_CLS(
            image_size=self.image_size,
            image_channels=self.image_channels,
            classes=self.total_classes,
            fc_layers=self.vae_fc_layers,
            fc_units=self.vae_fc_units,
            z_dim=self.z_dim,
            recon_loss="BCE",
            network_output="sigmoid",
            device=str(self.device),
            scenario="class",
            contexts=getattr(self.dataset, "N_TASKS", self.current_task + 1),
        ).to(self.device)
        generator.scenario = "class"
        generator.classes_per_context = self.classes_per_task
        generator.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, generator.parameters()), lr=self.vae_lr
        )
        generator.lamda_pl = 1.0
        return generator

    def _freeze_generator(self, generator: nn.Module) -> nn.Module:
        frozen = copy.deepcopy(generator).to(self.device)
        frozen.eval()
        for param in frozen.parameters():
            param.requires_grad_(False)
        return frozen

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def _train_vae_on_current_task(self) -> None:
        if not self.current_task_buffer:
            return

        images = torch.stack([item[0] for item in self.current_task_buffer]).to(self.device)
        labels = torch.stack([item[1] for item in self.current_task_buffer]).to(self.device)

        dataset = data.TensorDataset(images, labels)
        dataloader = data.DataLoader(dataset, batch_size=min(128, len(dataset)), shuffle=True)

        for _ in range(self.vae_train_epochs):
            for batch_imgs, batch_labels in dataloader:
                replay_inputs = replay_labels = None
                if self.previous_vae is not None:
                    sample = self.previous_vae.sample(
                        batch_imgs.size(0), allowed_classes=self.seen_classes or None, only_x=False
                    )
                    if isinstance(sample, tuple):
                        replay_raw, y_used, _ = sample
                    else:
                        replay_raw = sample
                        y_used = None
                    replay_inputs = replay_raw.to(self.device)
                    if y_used is not None:
                        replay_labels = torch.tensor(y_used, device=self.device, dtype=torch.long)

                self.vae.train_a_batch(
                    batch_imgs,
                    y=batch_labels,
                    x_=replay_inputs,
                    y_=replay_labels,
                    scores_=None,
                    rnt=self.replay_weight,
                    context=self.current_task + 1,
                )

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
