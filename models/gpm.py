"""Original GPM integration for Mammoth.

This module exposes a :class:`ContinualModel` wrapper that drives the original
Gradient Projection Memory (GPM) implementation released with the paper
"Gradient Projection Memory for Continual Learning" (ICLR 2021).

Rather than re-implementing the algorithm, the wrapper loads the reference
repository under ``./GPM`` and calls its functions directly, adapting only the
minimum glue required to satisfy Mammoth's training pipeline.
"""

from __future__ import annotations

import importlib
import logging
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.utils.continual_model import ContinualModel

logger = logging.getLogger(__name__)


def _load_original_gpm_module():
    """Ensure the original GPM repository is importable and return its module."""

    repo_root = Path(__file__).resolve().parent.parent
    gpm_root = repo_root / "GPM"
    if not gpm_root.exists():  # pragma: no cover - sanity check
        raise ImportError(f"Original GPM repository not found at {gpm_root}")

    if str(gpm_root) not in sys.path:
        sys.path.insert(0, str(gpm_root))

    module = importlib.import_module("main_cifar100")
    return module


class Gpm(ContinualModel):
    """Wrapper around the original GPM implementation for Mammoth."""

    NAME = "gpm"
    COMPATIBILITY = ["class-il", "task-il"]

    _ORIGINAL_MODULE = None

    @classmethod
    def _ensure_original_module(cls):
        if cls._ORIGINAL_MODULE is None:
            cls._ORIGINAL_MODULE = _load_original_gpm_module()
        return cls._ORIGINAL_MODULE

    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        if parser is None:
            parser = ArgumentParser(description="Gradient Projection Memory (original implementation)")

        group = parser.add_argument_group("GPM (original)")
        group.add_argument(
            "--gpm-threshold-base",
            type=float,
            default=0.97,
            help="Base energy threshold used for subspace selection (default: 0.97)",
        )
        group.add_argument(
            "--gpm-threshold-increment",
            type=float,
            default=0.003,
            help="Increment applied to the threshold after each task (default: 0.003)",
        )
        group.add_argument(
            "--gpm-activation-samples",
            type=int,
            default=512,
            help="Number of samples retained for subspace estimation (default: 512)",
        )
        group.add_argument(
            "--gpm-max-proj-layers",
            type=int,
            default=5,
            help="Maximum number of layers considered for projection (default: 5)",
        )

        # Align optimizer defaults with the reference implementation.
        parser.set_defaults(lr=0.01, optim="sgd", momentum=0.9)
        return parser

    def __init__(self, backbone: nn.Module, loss: nn.Module, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)

        module = self._ensure_original_module()
        self._alexnet_cls = module.AlexNet
        self._train_projected = module.train_projected
        self._get_representation_matrix = module.get_representation_matrix
        self._update_gpm = module.update_GPM

        self.taskcla = self._build_taskcla()
        self.task_offsets = self._compute_task_offsets()
        self.total_classes = self.task_offsets[-1][1]

        # Replace Mammoth backbone with the original AlexNet configured for all tasks
        self.net = self._alexnet_cls(self.taskcla).to(self.device)
        self.net.train()

        # Optimizer & argument namespace matching the reference implementation signature
        self.opt = optim.SGD(
            self.net.parameters(),
            lr=self.args.lr,
            momentum=getattr(self.args, "momentum", 0.9),
        )
        self.criterion = loss
        default_batch_size = getattr(self.args, "batch_size", getattr(self.args, "batch_size_train", 32))
        self.original_args = SimpleNamespace(
            batch_size_train=default_batch_size,
            batch_size_test=default_batch_size,
        )

        # GPM state
        self.feature_list: List[np.ndarray] = []
        self.feature_mat: List[torch.Tensor] = []
        self.threshold_base = getattr(self.args, "gpm_threshold_base", 0.97)
        self.threshold_increment = getattr(self.args, "gpm_threshold_increment", 0.003)
        self.max_proj_layers = getattr(self.args, "gpm_max_proj_layers", 5)
        self.activation_capacity = getattr(self.args, "gpm_activation_samples", 512)
        self.activation_buffer: List[Tuple[torch.Tensor, int]] = []
        self.samples_seen = 0

        logger.info("Initialized original GPM wrapper with %d tasks", len(self.taskcla))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_taskcla(self) -> List[Tuple[int, int]]:
        if isinstance(self.dataset.N_CLASSES_PER_TASK, int):
            sizes: Sequence[int] = [self.dataset.N_CLASSES_PER_TASK] * self.dataset.N_TASKS
        else:
            sizes = list(self.dataset.N_CLASSES_PER_TASK)
        return [(task_id, n_classes) for task_id, n_classes in enumerate(sizes)]

    def _compute_task_offsets(self) -> List[Tuple[int, int]]:
        offsets: List[Tuple[int, int]] = []
        current = 0
        for _, n_classes in self.taskcla:
            offsets.append((current, current + n_classes))
            current += n_classes
        return offsets

    def _reservoir_add(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        if self.activation_capacity <= 0:
            return

        inputs_cpu = inputs.detach().cpu()
        labels_cpu = labels.detach().cpu()
        for idx in range(inputs_cpu.size(0)):
            data = inputs_cpu[idx]
            label = int(labels_cpu[idx].item())
            self.samples_seen += 1
            if len(self.activation_buffer) < self.activation_capacity:
                self.activation_buffer.append((data, label))
            else:
                replace_idx = random.randint(0, self.samples_seen - 1)
                if replace_idx < self.activation_capacity:
                    self.activation_buffer[replace_idx] = (data, label)

    def _build_projection_matrices(self) -> None:
        self.feature_mat = []
        device = self.device
        for matrix in self.feature_list[: self.max_proj_layers]:
            proj = torch.from_numpy(matrix @ matrix.transpose()).float().to(device)
            self.feature_mat.append(proj)
        logger.debug("Constructed %d projection matrices", len(self.feature_mat))

    def _current_threshold(self) -> np.ndarray:
        base = np.array([self.threshold_base] * self.max_proj_layers)
        increment = np.array([self.threshold_increment] * self.max_proj_layers) * self.current_task
        return base + increment

    # ------------------------------------------------------------------
    # ContinualModel interface
    # ------------------------------------------------------------------
    def begin_task(self, dataset) -> None:
        super().begin_task(dataset)
        self.activation_buffer = []
        self.samples_seen = 0

        if self.current_task > 0 and self.feature_list:
            self._build_projection_matrices()
        else:
            self.feature_mat = []

        self.net.train()
        logger.info("Begin task %d", self.current_task)

    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        not_aug_inputs: torch.Tensor,
        epoch: int = None,
    ) -> float:
        self.net.train()
        self._reservoir_add(not_aug_inputs, labels)

        start_offset, _ = self.task_offsets[self.current_task]
        task_labels = labels - start_offset

        if self.current_task == 0 or not self.feature_mat:
            self.opt.zero_grad()
            outputs = self.net(inputs)
            logits = outputs[self.current_task]
            loss = self.criterion(logits, task_labels)
            loss.backward()
            self.opt.step()
            return loss.item()

        batch_size = inputs.size(0)
        self.original_args.batch_size_train = batch_size
        self.original_args.batch_size_test = batch_size

        self._train_projected(
            self.original_args,
            self.net,
            self.device,
            inputs.detach(),
            task_labels.detach(),
            self.opt,
            self.criterion,
            self.feature_mat,
            self.current_task,
        )

        with torch.no_grad():
            outputs = self.net(inputs)
            logits = outputs[self.current_task]
            loss_val = self.criterion(logits, task_labels).item()
        return loss_val

    def end_task(self, dataset) -> None:
        if not self.activation_buffer:
            logger.warning("No activations collected for task %d, skipping GPM update", self.current_task)
            super().end_task(dataset)
            return

        samples = self.activation_buffer[: self.activation_capacity]
        x_tensor = torch.stack([item[0] for item in samples]).to(self.device)
        y_tensor = torch.tensor([item[1] for item in samples], dtype=torch.long, device=self.device)

        mat_list = self._get_representation_matrix(self.net, self.device, x_tensor, y_tensor)
        threshold = self._current_threshold()
        self.feature_list = self._update_gpm(self.net, mat_list, threshold, self.feature_list)
        self._build_projection_matrices()

        logger.info("Updated GPM memory after task %d", self.current_task)
        super().end_task(dataset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.net(x)
        if isinstance(outputs, list):
            batch_size = x.size(0)
            logits = torch.zeros(batch_size, self.total_classes, device=x.device)
            for task_idx, (start, end) in enumerate(self.task_offsets):
                logits[:, start:end] = outputs[task_idx]
            return logits
        return outputs


__all__ = ["Gpm"]
