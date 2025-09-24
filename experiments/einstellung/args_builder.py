"""Helpers to construct Mammoth CLI arguments for Einstellung runs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import ExperimentConfig

_DATASETS = {
    "resnet18": "seq-cifar100-einstellung",
    "vit": "seq-cifar100-einstellung-224",
}

_DEFAULT_EPOCHS = {
    "resnet18": 50,
    "vit": 20,
}

_DEBUG_EPOCHS = {
    "resnet18": 10,
    "vit": 5,
}

_DEFAULT_BATCH_SIZE = {
    "resnet18": 32,
    "vit": 32,
}

_DEFAULT_LR = 0.01


def determine_dataset(backbone: str) -> str:
    backbone_lower = backbone.lower()
    if backbone_lower in _DATASETS:
        return _DATASETS[backbone_lower]
    if "vit" in backbone_lower:
        return _DATASETS["vit"]
    return _DATASETS["resnet18"]


def resolve_epochs(config: ExperimentConfig) -> int:
    if config.epochs is not None:
        return int(config.epochs)

    backbone_key = "vit" if "vit" in config.backbone.lower() else "resnet18"
    base = _DEBUG_EPOCHS[backbone_key] if config.debug else _DEFAULT_EPOCHS[backbone_key]

    if config.strategy == "interleaved":
        return base * 2

    return base


def resolve_batch_size(config: ExperimentConfig) -> int:
    if config.batch_size is not None:
        return int(config.batch_size)
    backbone_key = "vit" if "vit" in config.backbone.lower() else "resnet18"
    return _DEFAULT_BATCH_SIZE[backbone_key]


def resolve_learning_rate(config: ExperimentConfig) -> float:
    return float(config.learning_rate if config.learning_rate is not None else _DEFAULT_LR)


def build_mammoth_args(
    config: ExperimentConfig,
    *,
    results_path: Path,
    evaluation_only: bool,
    checkpoint_path: Optional[Path],
) -> List[str]:
    """Assemble the argument list passed to `main.py`."""

    args: List[str] = [
        "--dataset",
        determine_dataset(config.backbone),
        "--model",
        config.strategy,
        "--backbone",
        config.backbone,
        "--seed",
        str(config.seed),
        "--n_epochs",
        str(resolve_epochs(config)),
        "--batch_size",
        str(resolve_batch_size(config)),
        "--lr",
        str(resolve_learning_rate(config)),
        "--num_workers",
        str(config.num_workers if config.num_workers is not None else 4),
        "--results_path",
        str(results_path),
        "--savecheck",
        "last",
        "--code_optimization",
        str(config.code_optimization),
    ]

    # Evaluation-only flag
    if evaluation_only:
        args.extend(["--inference_only", "1"])

    # Checkpoint loading
    if checkpoint_path is not None:
        args.extend(["--loadcheck", str(checkpoint_path)])

    if not config.enable_cache:
        args.extend(["--einstellung_enable_cache", "0"])

    if config.debug:
        args.extend(["--debug_mode", "1"])

    # Strategy-specific overrides (mirrors legacy behaviour)
    if config.strategy == "derpp":
        args.extend(["--buffer_size", "500", "--alpha", "0.1", "--beta", "0.5"])
    elif config.strategy == "ewc_on":
        args.extend(["--e_lambda", "1000", "--gamma", "1.0"])
    elif config.strategy == "gpm":
        args.extend([
            "--gpm-threshold-base",
            "0.97",
            "--gpm-threshold-increment",
            "0.003",
            "--gpm-activation-samples",
            "512",
        ])
    elif config.strategy == "dgr":
        args.extend([
            "--dgr-z-dim",
            "100",
            "--dgr-vae-lr",
            "0.001",
            "--dgr-replay-ratio",
            "0.5",
            "--dgr-temperature",
            "2.0",
        ])

    # Arbitrary extra args from config
    for key, value in config.extra_args.items():
        args.extend([key, str(value)])

    return args
