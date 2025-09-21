"""Original DGR integration for Mammoth.

This module wraps the reference Deep Generative Replay implementation to expose
it as a :class:`models.utils.continual_model.ContinualModel` inside the Mammoth
pipeline. The wrapper keeps the original classifier and VAE generator intact,
calling their ``train_a_batch`` routines during Mammoth's ``observe`` calls.
"""

from __future__ import annotations

import copy
import importlib
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.utils.continual_model import ContinualModel

logger = logging.getLogger(__name__)


def _ensure_dgr_imports():
    repo_root = Path(__file__).resolve().parent.parent
    dgr_root = repo_root / "DGR"
    if not dgr_root.exists():  # pragma: no cover - sanity check
        raise ImportError(f"Original DGR repository not found at {dgr_root}")
    if str(dgr_root) not in sys.path:
        sys.path.insert(0, str(dgr_root))
    return dgr_root


def _import_dgr_module(module_name: str):
    """Import module from original DGR repository while preserving Mammoth modules."""

    repo_root = _ensure_dgr_imports()

    # Preserve Mammoth modules using the 'models' namespace
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


class Dgr(ContinualModel):
    """Wrapper around the original Deep Generative Replay implementation."""

    NAME = "dgr"
    COMPATIBILITY = ["class-il"]

    @staticmethod
    def _infer_image_channels(dataset) -> int:
        """Best effort inference of input channels when dataset.SIZE omits them."""
        for attr in ("IMAGE_CHANNELS", "CHANNELS", "channels", "in_channels"):
            value = getattr(dataset, attr, None)
            if isinstance(value, int) and value > 0:
                return value

        for stats_attr in ("MEAN", "STD"):
            stats = getattr(dataset, stats_attr, None)
            if isinstance(stats, (tuple, list)) and stats:
                return len(stats)

        return 3

    def __init__(self, backbone: nn.Module, loss: nn.Module, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset)

        if not hasattr(self.__class__, "_CLASSIFIER_CLS"):
            self.__class__._CLASSIFIER_CLS = None
            self.__class__._COND_VAE_CLS = None

        if self.__class__._CLASSIFIER_CLS is None:
            classifier_module = _import_dgr_module("models.classifier")
            self.__class__._CLASSIFIER_CLS = classifier_module.Classifier

        if self.__class__._COND_VAE_CLS is None:
            condvae_module = _import_dgr_module("models.cond_vae")
            self.__class__._COND_VAE_CLS = condvae_module.CondVAE

        self._Classifier = self.__class__._CLASSIFIER_CLS
        self._CondVAE = self.__class__._COND_VAE_CLS

        size_tuple = tuple(self.dataset.SIZE)
        if len(size_tuple) == 3:
            self.image_channels, self.image_height, self.image_width = size_tuple
        elif len(size_tuple) == 2:
            # Legacy datasets expose only height/width; infer channels from stats when needed.
            self.image_height, self.image_width = size_tuple
            self.image_channels = self._infer_image_channels(self.dataset)
        else:
            raise ValueError(
                f"Unsupported dataset SIZE format: expected 2 or 3 values, got {size_tuple}"
            )
        self.total_classes = self.dataset.N_CLASSES
        if isinstance(self.dataset.N_CLASSES_PER_TASK, int):
            class_counts: Sequence[int] = [
                self.dataset.N_CLASSES_PER_TASK for _ in range(self.dataset.N_TASKS)
            ]
        else:
            class_counts = list(self.dataset.N_CLASSES_PER_TASK)
        self._classes_per_task_seq = class_counts
        self._cpt = class_counts

        # Instantiate original classifier
        self.classifier = self._Classifier(
            image_size=self.image_height,
            image_channels=self.image_channels,
            classes=self.total_classes,
            depth=0,
            fc_layers=2,
            fc_units=400,
            fc_drop=0.0,
            fc_bn=True,
            fc_nl="relu",
        ).to(self.device)
        self.classifier.scenario = "class"
        self.classifier.classes_per_context = self.classes_per_task
        self.classifier.singlehead = True
        self.classifier.replay_mode = "generative"
        self.classifier.replay_targets = "soft"
        self.classifier.use_replay = "normal"
        self.classifier.neg_samples = "all-so-far"
        self.classifier.KD_temp = getattr(self.args, "dgr_temperature", 2.0)
        self.classifier.optimizer = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.args.lr,
            momentum=getattr(self.args, "momentum", 0.9),
        )
        self.opt = self.classifier.optimizer

        # Instantiate VAE generator
        self.generator = self._CondVAE(
            image_size=self.image_height,
            image_channels=self.image_channels,
            classes=self.total_classes,
            fc_layers=2,
            fc_units=400,
            z_dim=getattr(self.args, "dgr_z_dim", 100),
            recon_loss="BCE",
            network_output="sigmoid",
        ).to(self.device)
        self.generator.scenario = "class"
        self.generator.optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=getattr(self.args, "dgr_vae_lr", 1e-3),
        )
        self.generator.lamda_rcl = 1.0
        self.generator.lamda_vl = 1.0
        self.generator.lamda_pl = 1.0

        # Replace Mammoth backbone with original classifier
        self.net = self.classifier

        # Replay state
        self.prev_classifier: Optional[nn.Module] = None
        self.prev_generator: Optional[nn.Module] = None
        self.seen_classes: List[int] = []

        logger.info("Initialized original DGR wrapper with %d classes", self.total_classes)

    # ------------------------------------------------------------------
    @staticmethod
    def get_parser(parser: ArgumentParser) -> ArgumentParser:
        if parser is None:
            parser = ArgumentParser(description="Deep Generative Replay (original implementation)")
        group = parser.add_argument_group("DGR (original)")
        group.add_argument("--dgr-vae-lr", type=float, default=1e-3, help="Learning rate for the VAE generator")
        group.add_argument("--dgr-z-dim", type=int, default=100, help="Latent dimensionality for the VAE")
        group.add_argument("--dgr-temperature", type=float, default=2.0, help="Distillation temperature")
        group.add_argument(
            "--dgr-replay-ratio",
            type=float,
            default=0.5,
            help="Relative importance of current batch when mixing with replay",
        )
        group.add_argument(
            "--dgr-replay-weight",
            type=float,
            dest="dgr_replay_ratio",
            default=None,
            help="Legacy synonym for --dgr-replay-ratio",
        )
        parser.set_defaults(lr=0.01, optim="sgd", momentum=0.9)
        return parser

    # ------------------------------------------------------------------
    def _update_seen_classes(self, labels: torch.Tensor) -> None:
        new_classes = torch.unique(labels).tolist()
        for cls in new_classes:
            if cls not in self.seen_classes:
                self.seen_classes.append(cls)
        self.seen_classes.sort()

    def _replay_ratio(self) -> float:
        if hasattr(self.args, "dgr_replay_ratio") and getattr(self.args, "dgr_replay_ratio") is not None:
            return float(getattr(self.args, "dgr_replay_ratio"))
        return float(getattr(self.args, "dgr_replay_weight", 0.5))

    def _sample_replay(self, batch_size: int):
        if self.prev_generator is None or not self.seen_classes:
            return None, None, None
        replay_inputs, y_np, _ = self.prev_generator.sample(
            batch_size,
            allowed_classes=self.seen_classes,
            only_x=False,
        )
        replay_inputs = replay_inputs.to(self.device)

        # Handle case where y_np is None (when VAE doesn't use GMM prior or decoder gates)
        if y_np is None:
            # Generate random labels from seen classes for replay samples
            import numpy as np
            y_np = np.random.choice(self.seen_classes, size=batch_size)

        replay_labels = torch.from_numpy(y_np).long().to(self.device)
        with torch.no_grad():
            replay_scores = self.prev_classifier.classify(replay_inputs)
        return replay_inputs, replay_labels, replay_scores

    # ------------------------------------------------------------------
    def begin_task(self, dataset) -> None:
        super().begin_task(dataset)
        self.net.train()

    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        not_aug_inputs: torch.Tensor,
        epoch: int = None,
    ) -> float:
        inputs = inputs.to(self.device)
        labels = labels.to(self.device, dtype=torch.long)

        replay_inputs, replay_labels, replay_scores = self._sample_replay(inputs.size(0))

        stats = self.classifier.train_a_batch(
            inputs,
            labels,
            x_=replay_inputs,
            y_=replay_labels,
            scores_=replay_scores,
            rnt=self._replay_ratio(),
        )

        self.generator.train_a_batch(
            inputs,
            y=labels,
            x_=replay_inputs,
            y_=replay_labels,
            scores_=replay_scores,
            rnt=self._replay_ratio(),
        )

        self._update_seen_classes(labels)

        return float(stats.get("loss_total", 0.0))

    def end_task(self, dataset) -> None:
        self.prev_classifier = copy.deepcopy(self.classifier).to(self.device)
        self.prev_classifier.eval()
        self.prev_generator = copy.deepcopy(self.generator).to(self.device)
        self.prev_generator.eval()
        super().end_task(dataset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


__all__ = ["Dgr"]
