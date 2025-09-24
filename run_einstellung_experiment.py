#!/usr/bin/env python3
"""Modernised Einstellung experiment CLI."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from experiments.einstellung import (
    ComparativeExperimentPlan,
    ExperimentConfig,
    ExecutionMode,
    EinstellungRunner,
    run_comparative_suite,
)
from experiments.einstellung.args_builder import build_mammoth_args, determine_dataset
from experiments.einstellung.reporting import write_single_run_report

LOGGER = logging.getLogger("einstellung.cli")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Einstellung Effect experiments")

    parser.add_argument("--comparative", action="store_true", help="Run comparative suite")
    parser.add_argument("--model", default="derpp", choices=["sgd", "derpp", "ewc_on", "gpm", "dgr", "scratch_t2", "interleaved"], help="Strategy to run")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "vit"], help="Backbone architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, help="Override training epochs")

    parser.add_argument("--skip_training", action="store_true", help="Skip training and evaluate existing checkpoint")
    parser.add_argument("--force_retrain", action="store_true", help="Always retrain even if checkpoints exist")
    parser.add_argument("--auto_checkpoint", action="store_true", help="Use checkpoint automatically when available")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode (short runs)")
    parser.add_argument("--code_optimization", type=int, default=1, choices=[0, 1, 2, 3], help="CUDA optimisation level")
    parser.add_argument("--disable_cache", action="store_true", help="Disable Einstellung dataset caching")
    parser.add_argument("--enable_cache", action="store_true", default=True, help="Enable Einstellung dataset caching")
    parser.add_argument("--results_root", default="einstellung_results", help="Root directory for outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args(argv)


def execution_mode_from_args(args: argparse.Namespace) -> ExecutionMode:
    return ExecutionMode.from_flags(
        skip_training=args.skip_training,
        force_retrain=args.force_retrain,
        auto_checkpoint=args.auto_checkpoint,
    )


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        strategy=args.model,
        backbone=args.backbone,
        seed=args.seed,
        epochs=args.epochs,
        debug=args.debug,
        enable_cache=args.enable_cache and not args.disable_cache,
        code_optimization=args.code_optimization,
        execution_mode=execution_mode_from_args(args),
        results_root=Path(args.results_root),
    )


def run_single(args: argparse.Namespace) -> Dict[str, any]:
    config = build_config(args)
    runner = EinstellungRunner(project_root=Path.cwd())
    result = runner.run(config)

    if result.get("success"):
        summary_df = pd.read_csv(result["summary_path"]) if result.get("summary_path") else pd.DataFrame()
        report_dir = Path(result["results_dir"]) / "reports"
        write_single_run_report(result=result, summary_df=summary_df, output_dir=report_dir)

    return result


def run_comparative(args: argparse.Namespace) -> List[Dict[str, any]]:
    config = build_config(args)
    runner = EinstellungRunner(project_root=Path.cwd())

    plan = ComparativeExperimentPlan(
        baselines=["scratch_t2", "interleaved"],
        continual_methods=["sgd", "derpp", "ewc_on", "gpm", "dgr"],
        backbone=args.backbone,
        seed=args.seed,
        epochs=args.epochs,
    )

    output_root = Path("comparative_results")
    results, report_path = run_comparative_suite(runner, config, plan, output_root)

    LOGGER.info("Comparative report: %s", report_path)
    return results


def run_einstellung_experiment(
    strategy: str = "derpp",
    backbone: str = "resnet18",
    seed: int = 42,
    skip_training: bool = False,
    force_retrain: bool = False,
    auto_checkpoint: bool = True,
    debug: bool = False,
    enable_cache: bool = True,
    code_optimization: int = 1,
    epochs: Optional[int] = None,
) -> Dict[str, any]:
    args = argparse.Namespace(
        comparative=False,
        model=strategy,
        backbone=backbone,
        seed=seed,
        epochs=epochs,
        skip_training=skip_training,
        force_retrain=force_retrain,
        auto_checkpoint=auto_checkpoint,
        debug=debug,
        enable_cache=enable_cache,
        disable_cache=not enable_cache,
        code_optimization=code_optimization,
        results_root="einstellung_results",
        verbose=False,
    )
    return run_single(args)


def run_comparative_experiment(
    skip_training: bool = False,
    force_retrain: bool = False,
    auto_checkpoint: bool = True,
    debug: bool = False,
    enable_cache: bool = True,
    code_optimization: int = 1,
    epochs: Optional[int] = None,
) -> List[Dict[str, any]]:
    args = argparse.Namespace(
        comparative=True,
        model="derpp",
        backbone="resnet18",
        seed=42,
        epochs=epochs,
        skip_training=skip_training,
        force_retrain=force_retrain,
        auto_checkpoint=auto_checkpoint,
        debug=debug,
        enable_cache=enable_cache,
        disable_cache=not enable_cache,
        code_optimization=code_optimization,
        results_root="einstellung_results",
        verbose=False,
    )
    return run_comparative(args)


def create_einstellung_args(strategy: str, backbone: str, seed: int, debug: bool = False, epochs: Optional[int] = None) -> List[str]:
    """Legacy helper retained for tests – returns CLI args for main.py."""
    config = ExperimentConfig(
        strategy=strategy,
        backbone=backbone,
        seed=seed,
        debug=debug,
        epochs=epochs,
    )
    return build_mammoth_args(config, results_path=Path("/tmp"), evaluation_only=False, checkpoint_path=None)


def extract_accuracy_from_output(output: str) -> Optional[float]:
    pattern = r"Accuracy for \d+ task\(s\):\s*\[Class-IL\]:\s*([\d.]+)\s*%"
    match = re.search(pattern, output or "")
    return float(match.group(1)) if match else None


def find_csv_file(output_dir: str) -> Optional[str]:
    root = Path(output_dir)
    if not root.exists():
        return None
    for candidate in [root / "timeline.csv", root / "eri_sc_metrics.csv", root / "summary.csv"]:
        if candidate.exists():
            return str(candidate)
    csv_files = sorted(root.glob("**/*.csv"))
    return str(csv_files[0]) if csv_files else None


def aggregate_comparative_results(results_list: List[Dict[str, any]], output_dir: str) -> str:
    frames: List[pd.DataFrame] = []

    for result in results_list:
        if not result.get("success", True):
            continue

        csv_path = None
        if "summary_path" in result:
            csv_path = result["timeline_path"] if "timeline_path" in result else result["summary_path"]
        elif "output_dir" in result:
            csv_path = find_csv_file(result["output_dir"])

        if not csv_path or not validate_csv_file(csv_path):
            continue

        df = pd.read_csv(csv_path)
        frames.append(df)

    if not frames:
        raise ValueError("No valid CSV files found in experiment results")

    merged = pd.concat(frames, ignore_index=True)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    aggregated_path = output_path / "comparative_eri_metrics.csv"
    merged.to_csv(aggregated_path, index=False)
    return str(aggregated_path)


def validate_csv_file(csv_path: str) -> bool:
    try:
        if not Path(csv_path).exists() or Path(csv_path).stat().st_size == 0:
            return False
        df = pd.read_csv(csv_path)
    except Exception:
        return False

    required_cols = {"method", "seed", "epoch_eff", "split", "acc"}
    if not required_cols.issubset(df.columns):
        return False

    if df.empty:
        return False

    if df["acc"].min() < 0 or df["acc"].max() > 1:
        return False

    return True


def get_significance_indicator(method: str, statistical_results: Dict[str, any]) -> str:
    """Placeholder significance indicator retained for backwards compatibility."""
    if not statistical_results:
        return ""
    comparisons = statistical_results.get("pairwise", {})
    p_value = comparisons.get(method)
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def generate_eri_visualizations(output_dir: str, context: Dict[str, Any]) -> bool:
    """Forward visualisation requests to the ERI visualisation package when available."""
    try:
        from eri_vis.runner import generate_eri_visualizations as impl  # type: ignore

        return bool(impl(output_dir, context))
    except ImportError:
        LOGGER.warning("eri_vis package not installed; skipping visualisation")
        return False
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Visualisation generation failed: %s", exc)
        return False


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.comparative:
        results = run_comparative(args)
        successes = sum(1 for r in results if r.get("success"))
        LOGGER.info("Comparative run finished – %d/%d successful", successes, len(results))
        return 0 if successes else 1

    result = run_single(args)
    if result.get("success"):
        LOGGER.info("Run complete: top-1=%.2f%%", (result.get("final_top1") or 0) * 100)
        return 0

    LOGGER.error("Run failed: %s", result.get("stderr"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
