"""Execution engine for Einstellung experiments."""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .args_builder import build_mammoth_args, determine_dataset
from .config import ExperimentConfig, ExecutionMode
from .storage import describe_checkpoint, find_existing_checkpoints

LOGGER = logging.getLogger("einstellung.runner")


@dataclass
class RunArtifacts:
    timeline: pd.DataFrame
    summary: pd.DataFrame
    final_metrics: Dict[str, Any]
    summary_path: Path
    timeline_path: Path


class EinstellungRunner:
    """High-level orchestrator for single Einstellung experiments."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.python = Path(sys.executable)

    def run(self, config: ExperimentConfig) -> Dict[str, Any]:
        results_dir = config.session_dir or config.resolve_results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = determine_dataset(config.backbone)

        checkpoints = find_existing_checkpoints(
            config.strategy,
            config.backbone,
            config.seed,
            dataset=dataset_name,
        )

        checkpoint_to_use: Optional[Path] = None
        evaluation_only = False
        used_checkpoint = False

        if config.execution_mode == ExecutionMode.FORCE_RETRAIN:
            used_checkpoint = False
        elif config.execution_mode == ExecutionMode.SKIP_TRAINING:
            if not checkpoints:
                raise RuntimeError("skip_training requested but no checkpoint available")
            checkpoint_to_use = checkpoints[0]
            used_checkpoint = True
            evaluation_only = True
        elif config.execution_mode == ExecutionMode.AUTO_CHECKPOINT and checkpoints:
            checkpoint_to_use = checkpoints[0]
            used_checkpoint = True
            evaluation_only = True
        else:
            used_checkpoint = False

        args = build_mammoth_args(
            config,
            results_path=results_dir,
            evaluation_only=evaluation_only,
            checkpoint_path=checkpoint_to_use,
        )

        command = [str(self.python), "main.py", *args]

        try:
            log_path = results_dir / "command_history.log"
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(" ".join(command) + "\n")
        except OSError:
            LOGGER.debug("Could not write command history to %s", results_dir)

        process = subprocess.run(
            command,
            cwd=self.project_root,
        )

        artifacts: Optional[RunArtifacts] = None
        success = process.returncode == 0

        if success:
            try:
                prefix = config.output_prefix or config.strategy
                artifacts = self._load_artifacts(results_dir, prefix)
            except FileNotFoundError as exc:  # pragma: no cover - defensive
                success = False
                LOGGER.error("Missing expected output files: %s", exc)

        result: Dict[str, Any] = {
            "strategy": config.strategy,
            "backbone": config.backbone,
            "seed": config.seed,
            "success": success,
            "used_checkpoint": used_checkpoint,
            "evaluation_only": evaluation_only,
            "checkpoint_path": str(checkpoint_to_use) if checkpoint_to_use else None,
            "checkpoints_available": [str(p) for p in checkpoints],
            "command": command,
            "results_dir": str(results_dir),
            "reference_top1": config.reference_top1,
            "reference_top5": config.reference_top5,
            "return_code": process.returncode,
        }

        if checkpoint_to_use is not None:
            result["checkpoint_info"] = describe_checkpoint(checkpoint_to_use)

        if artifacts is not None:
            result.update(
                {
                    "timeline_path": str(artifacts.timeline_path),
                    "summary_path": str(artifacts.summary_path),
                    "final_results_path": str(artifacts.summary_path),
                    "final_metrics": artifacts.final_metrics,
                    "summary_records": artifacts.summary.to_dict(orient="records"),
                }
            )

            t2_summary = artifacts.summary[artifacts.summary["subset"] == "T2_shortcut_normal"]
            if not t2_summary.empty:
                row = t2_summary.iloc[0]
                result.update(
                    {
                        "final_top1": float(row["top1"]),
                        "final_top5": float(row["top5"]) if pd.notna(row["top5"]) else None,
                        "top1_delta": float(row["top1_delta"]),
                        "top5_delta": float(row["top5_delta"]) if pd.notna(row["top5_delta"]) else None,
                        "performance_deficit": row["performance_deficit"],
                        "shortcut_feature_reliance": row["shortcut_feature_reliance"],
                        "adaptation_delay": row["adaptation_delay"],
                    }
                )
                result["final_accuracy"] = float(row["top1"]) * 100

        result.setdefault('output_dir', result.get('results_dir'))
        return result

    def _load_artifacts(self, results_dir: Path, prefix: str) -> RunArtifacts:
        timeline_path = results_dir / f"timeline_{prefix}.csv"
        summary_path = results_dir / f"summary_{prefix}.csv"

        missing = [p.name for p in (timeline_path, summary_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing outputs in {results_dir}: {', '.join(missing)}"
            )

        timeline = pd.read_csv(timeline_path)
        summary = pd.read_csv(summary_path)
        final_metrics = {}
        if "subset" in summary.columns:
            topline = summary[summary["subset"] == "T2_shortcut_normal"]
            if not topline.empty:
                row = topline.iloc[0]
                final_metrics = {
                    "adaptation_delay": row.get("adaptation_delay"),
                    "performance_deficit": row.get("performance_deficit"),
                    "shortcut_feature_reliance": row.get("shortcut_feature_reliance"),
                    "top1": row.get("top1"),
                    "top5": row.get("top5"),
                }

        return RunArtifacts(
            timeline=timeline,
            summary=summary,
            final_metrics=final_metrics,
            summary_path=summary_path,
            timeline_path=timeline_path,
        )
