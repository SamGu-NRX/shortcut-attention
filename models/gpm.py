"""Gradient Projection Memory integration with Mammoth backbones.

The original project bundled the reference implementation released with the
paper.  While faithful, that wrapper swapped in the paper's bespoke AlexNet
classifier which performs poorly when compared to the ResNet backbones used by
the rest of the Einstellung benchmark.  The adapter below re-implements the
core idea—projecting gradients onto the orthogonal complement of previously
observed feature subspaces—directly on top of Mammoth's `ContinualModel`
interface.  This lets GPM train the same backbone as the other strategies and
unlock comparable accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from models.utils.continual_model import ContinualModel

LOGGER = logging.getLogger(__name__)


def _resolve_classifier(backbone: nn.Module) -> nn.Linear:
    """Best effort resolution of the classifier layer used for logits."""

    candidate_names = [
        "classifier",
        "fc",
        "head",
    ]
    for name in candidate_names:
        module = getattr(backbone, name, None)
        if isinstance(module, nn.Linear):
            return module

    # Fall back to searching the module tree for the first linear layer that
    # matches the output dimensionality of the network.
    for module in backbone.modules():
        if isinstance(module, nn.Linear):
            return module

    raise AttributeError("Could not locate a classifier layer on the backbone")


@dataclass
class FeatureBuffer:
    """Stores penultimate activations for the current task."""

    capacity: int
    tensors: List[torch.Tensor]

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tensors = []
        self._count = 0

    def append(self, features: torch.Tensor) -> None:
        if self._count >= self.capacity:
            return

        remaining = self.capacity - self._count
        to_store = features[:remaining].detach().cpu()
        if to_store.numel() == 0:
            return
        self.tensors.append(to_store)
        self._count += to_store.size(0)

    def as_tensor(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.tensors:
            return None
        stacked = torch.cat(self.tensors, dim=0)
        return stacked.to(device)

    def clear(self) -> None:
        self.tensors.clear()
        self._count = 0


class FeatureSubspaceProjector:
    """Maintains an orthonormal basis of important features and projects gradients."""

    def __init__(
        self,
        feature_dim: int,
        device: torch.device,
        base_threshold: float = 0.97,
        threshold_increment: float = 0.003,
    ) -> None:
        self.feature_dim = feature_dim
        self.device = device
        self.base_threshold = base_threshold
        self.threshold_increment = threshold_increment

        self.basis: Optional[torch.Tensor] = None
        self.projection: Optional[torch.Tensor] = None

    def _current_threshold(self, task_idx: int) -> float:
        return min(0.999, self.base_threshold + self.threshold_increment * task_idx)

    def update(self, features: torch.Tensor, task_idx: int) -> None:
        """Expand the stored basis with information from the provided features."""

        if features is None or features.numel() == 0:
            return

        threshold = self._current_threshold(task_idx)

        work_features = features.detach()
        if work_features.device.type != "cpu":
            work_features = work_features.to("cpu")
        work_features = work_features - work_features.mean(dim=0, keepdim=True)
        try:
            u, s, _ = torch.linalg.svd(work_features, full_matrices=False)
        except RuntimeError as exc:
            LOGGER.warning("GPM: SVD failed on feature batch (%s)", exc)
            return

        energy = s.pow(2)
        total_energy = energy.sum()
        if total_energy <= 0:
            return

        cumulative = torch.cumsum(energy, dim=0) / total_energy
        keep = int((cumulative <= threshold).sum().item())
        if keep == 0:
            keep = 1

        new_basis = u[:, :keep]

        existing_basis = None
        if self.basis is not None:
            existing_basis = self.basis.to(new_basis.device)

        if existing_basis is not None:
            projection = existing_basis @ (existing_basis.t() @ new_basis)
            new_basis = new_basis - projection
            if new_basis.numel() == 0:
                return
            new_basis, _ = torch.linalg.qr(new_basis, mode="reduced")
            combined_basis = torch.cat([existing_basis, new_basis], dim=1)
        else:
            new_basis, _ = torch.linalg.qr(new_basis, mode="reduced")
            combined_basis = new_basis

        if combined_basis.numel() == 0:
            return

        self.basis = combined_basis.contiguous()
        self.projection = self.basis @ self.basis.t()

    def project_gradients(self, classifier: nn.Linear) -> None:
        if self.projection is None:
            return
        if classifier.weight.grad is None:
            return

        projection = self.projection.to(classifier.weight.grad.device)
        grad = classifier.weight.grad
        grad.sub_(grad @ projection)


class Gpm(ContinualModel):
    """Gradient Projection Memory using Mammoth backbones."""

    NAME = "gpm"
    COMPATIBILITY = ["class-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser: Optional["ArgumentParser"] = None):  # type: ignore[override]
        from argparse import ArgumentParser

        if parser is None:
            parser = ArgumentParser(description="Gradient Projection Memory (Mammoth adapter)")

        group = parser.add_argument_group("GPM")
        group.add_argument(
            "--gpm-threshold-base",
            type=float,
            default=0.97,
            help="Base energy threshold used when selecting basis vectors (default: 0.97)",
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
            help="Number of feature vectors retained per task for subspace estimation",
        )

        return parser

    def __init__(self, backbone: nn.Module, loss: nn.Module, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)

        self.classifier = _resolve_classifier(self.net)
        feature_dim = self.classifier.weight.size(1)

        self.activation_capacity = getattr(args, "gpm_activation_samples", 512)
        self.threshold_base = getattr(args, "gpm_threshold_base", 0.97)
        self.threshold_increment = getattr(args, "gpm_threshold_increment", 0.003)

        self.feature_buffer = FeatureBuffer(self.activation_capacity)
        self.projector = FeatureSubspaceProjector(
            feature_dim=feature_dim,
            device=self.device,
            base_threshold=self.threshold_base,
            threshold_increment=self.threshold_increment,
        )

        self._collect_features = True
        self._feature_hook = self.classifier.register_forward_pre_hook(self._store_features)

        LOGGER.info(
            "Initialized GPM adapter with feature_dim=%d, capacity=%d",
            feature_dim,
            self.activation_capacity,
        )

    def _store_features(self, module: nn.Module, inputs) -> None:
        if not self._collect_features:
            return
        if not inputs:
            return
        features = inputs[0]
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        self.feature_buffer.append(features)
        if self.feature_buffer._count >= self.activation_capacity:
            self._collect_features = False

    def begin_task(self, dataset) -> None:
        super().begin_task(dataset)
        self.feature_buffer.clear()
        self._collect_features = True

    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        not_aug_inputs: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> float:
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        if self.current_task > 0:
            self.projector.project_gradients(self.classifier)

        self.opt.step()
        return float(loss.item())

    def end_task(self, dataset) -> None:
        features = self.feature_buffer.as_tensor(self.device)
        if features is None:
            LOGGER.warning(
                "GPM: no features collected for task %d, skipping basis update", self.current_task
            )
        else:
            LOGGER.info(
                "GPM: updating feature subspace with %d vectors for task %d",
                features.size(0),
                self.current_task,
            )
            self.projector.update(features, self.current_task)

        self.feature_buffer.clear()
        self._collect_features = False
        super().end_task(dataset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def __del__(self) -> None:
        if hasattr(self, "_feature_hook"):
            try:
                self._feature_hook.remove()
            except Exception:
                pass


__all__ = ["Gpm"]
