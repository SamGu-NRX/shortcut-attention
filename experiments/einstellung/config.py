"""Configuration objects for Einstellung experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional


class ExecutionMode(str, Enum):
    """Enumerates the high-level execution modes supported by the runner."""

    TRAIN = "train"  # Train even if checkpoints exist (default behaviour)
    AUTO_CHECKPOINT = "auto_checkpoint"  # Reuse checkpoint automatically when present
    SKIP_TRAINING = "skip_training"  # Evaluate only, requires checkpoint
    FORCE_RETRAIN = "force_retrain"  # Ignore checkpoints and train from scratch

    @classmethod
    def from_flags(
        cls,
        *,
        skip_training: bool = False,
        force_retrain: bool = False,
        auto_checkpoint: bool = False,
    ) -> "ExecutionMode":
        if force_retrain:
            return cls.FORCE_RETRAIN
        if skip_training:
            return cls.SKIP_TRAINING
        if auto_checkpoint:
            return cls.AUTO_CHECKPOINT
        return cls.TRAIN


@dataclass
class ExperimentConfig:
    """User-facing configuration for a single Einstellung run."""

    strategy: str
    backbone: str = "resnet18"
    seed: int = 42

    # Training overrides
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    num_workers: Optional[int] = None

    # Execution semantics
    execution_mode: ExecutionMode = ExecutionMode.TRAIN
    checkpoint_path: Optional[Path] = None
    debug: bool = False
    enable_cache: bool = True
    code_optimization: int = 1

    # Output + reporting
    results_root: Path = Path("einstellung_results")
    session_dir: Optional[Path] = None
    output_prefix: Optional[str] = None
    reference_top1: float = 0.35
    reference_top5: float = 0.65

    # Misc options
    extra_args: Dict[str, str] = field(default_factory=dict)

    def resolve_results_dir(self) -> Path:
        """Compute the directory where this run stores artefacts."""
        return self.results_root / f"{self.strategy}_{self.backbone}_seed{self.seed}"

    def __post_init__(self) -> None:
        """Apply strategy-specific defaults without mutating shared state."""
        self.extra_args = dict(self.extra_args)

        if self.strategy == "scratch_t2":
            if "--start_from" not in self.extra_args:
                self.extra_args["--start_from"] = "1"

            if "--stop_after" not in self.extra_args:
                start_token = self.extra_args.get("--start_from", "1")
                try:
                    start_idx = int(start_token)
                except (TypeError, ValueError):
                    start_idx = 1
                self.extra_args["--stop_after"] = str(start_idx + 1)

    def with_epoch_override(self, epochs: Optional[int]) -> "ExperimentConfig":
        """Return a new config overriding the number of epochs."""
        clone = dataclass_replace(self)
        clone.epochs = epochs
        return clone


def dataclass_replace(config: ExperimentConfig) -> ExperimentConfig:
    """Internal helper to create a shallow copy of the config."""
    return ExperimentConfig(
        strategy=config.strategy,
        backbone=config.backbone,
        seed=config.seed,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_workers=config.num_workers,
        execution_mode=config.execution_mode,
        checkpoint_path=config.checkpoint_path,
        debug=config.debug,
        enable_cache=config.enable_cache,
        code_optimization=config.code_optimization,
        results_root=config.results_root,
        session_dir=config.session_dir,
        output_prefix=config.output_prefix,
        reference_top1=config.reference_top1,
        reference_top5=config.reference_top5,
        extra_args=dict(config.extra_args),
    )
