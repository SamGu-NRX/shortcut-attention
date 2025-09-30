#!/usr/bin/env python3
"""Modernised Einstellung experiment CLI."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from experiments.einstellung import (
    ComparativeExperimentPlan,
    ExperimentConfig,
    ExecutionMode,
    EinstellungRunner,
    run_comparative_suite,
)
from experiments.einstellung.analysis import combine_timelines
from experiments.einstellung.visualization import generate_comparative_plots
from experiments.einstellung.args_builder import build_mammoth_args, determine_dataset
from experiments.einstellung.reporting import write_single_run_report

LOGGER = logging.getLogger("einstellung.cli")


def create_session_dir(root: Path, seeds: Sequence[int]) -> Path:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = ""
    if seeds:
        unique_seeds = sorted({int(s) for s in seeds})
        if len(unique_seeds) == 1:
            suffix = f"_seed{unique_seeds[0]}"
        else:
            seed_fragment = "-".join(str(s) for s in unique_seeds)
            suffix = f"_seeds{seed_fragment}"
    session_dir = root / f"session_{timestamp}{suffix}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def log_command(session_dir: Path, command: List[str]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    history_file = session_dir / "command_history.log"
    with history_file.open("a", encoding="utf-8") as fh:
        fh.write(" ".join(command) + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Einstellung Effect experiments")

    parser.add_argument("--comparative", action="store_true", help="Run comparative suite")
    parser.add_argument("--model", default="derpp", help="Strategy or comma-separated list of strategies to run (e.g., sgd,dgr,gpm)")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "vit"], help="Backbone architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--seeds",
        nargs="+",
        help="List of seeds to run (supports space or comma separated values)",
    )
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


def build_config(
    args: argparse.Namespace,
    *,
    strategy: Optional[str] = None,
    seed: Optional[int] = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        strategy=strategy if strategy is not None else args.model,
        backbone=args.backbone,
        seed=seed if seed is not None else args.seed,
        epochs=args.epochs,
        debug=args.debug,
        enable_cache=args.enable_cache and not args.disable_cache,
        code_optimization=args.code_optimization,
        execution_mode=execution_mode_from_args(args),
        results_root=Path(args.results_root),
    )


def parse_seed_values(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        tokens: List[str] = []
        for item in args.seeds:
            tokens.extend(re.split(r"[\s,]+", item.strip()))

        seeds: List[int] = []
        for token in tokens:
            if not token:
                continue
            try:
                seeds.append(int(token))
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid seed value '{token}'") from exc

        if not seeds:
            raise ValueError("No valid seeds provided")

        return seeds

    return [int(args.seed)]


def resolve_run_directory(
    session_root: Path,
    model: str,
    seed: int,
    *,
    use_nested_layout: bool,
) -> Path:
    if not use_nested_layout:
        return session_root
    return session_root / model / f"seed{seed}"


def generate_session_plots(session_dir: Path, results: Iterable[Dict[str, Any]]) -> Dict[str, Path]:
    timeline_df = combine_timelines(results)
    if timeline_df.empty:
        return {}

    plots_dir = session_dir / "plots"
    try:
        return generate_comparative_plots(timeline_df, plots_dir)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Session plot generation failed: %s", exc)
        return {}


def run_models(
    args: argparse.Namespace,
    models: List[str],
    seeds: List[int],
    *,
    session_dir: Path,
    runner: EinstellungRunner,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    use_nested_layout = len(models) > 1 or len(seeds) > 1

    for model in models:
        for seed in seeds:
            config = build_config(args, strategy=model, seed=seed)
            config.output_prefix = model

            run_dir = resolve_run_directory(
                session_dir,
                model,
                seed,
                use_nested_layout=use_nested_layout,
            )
            config.session_dir = run_dir

            result = runner.run(config)
            results.append(result)

            if result.get("success"):
                summary_path = result.get("summary_path")
                if summary_path:
                    summary_df = pd.read_csv(summary_path)
                else:
                    summary_df = pd.DataFrame()
                report_dir = Path(result["results_dir"]) / "reports"
                write_single_run_report(result=result, summary_df=summary_df, output_dir=report_dir)

                final_top1 = result.get("final_top1")
                if final_top1 is not None:
                    LOGGER.info(
                        "Run complete – model='%s', seed=%s, top-1=%.2f%%",
                        model,
                        seed,
                        final_top1 * 100,
                    )
                else:
                    LOGGER.info("Run complete – model='%s', seed=%s", model, seed)
            else:
                LOGGER.error(
                    "Run failed – model='%s', seed=%s, return_code=%s",
                    model,
                    seed,
                    result.get("return_code"),
                )

    generated = generate_session_plots(session_dir, results)
    if generated:
        LOGGER.info(
            "Generated session plots: %s",
            ", ".join(f"{name}:{path}" for name, path in generated.items()),
        )

    return results


def run_comparative(
    args: argparse.Namespace,
    models: List[str],
    seeds: List[int],
    *,
    session_dir: Path,
    runner: EinstellungRunner,
) -> List[Dict[str, any]]:
    base_strategy = models[0] if models else "scratch_t2"
    base_seed = seeds[0] if seeds else args.seed
    base_config = build_config(args, strategy=base_strategy, seed=base_seed)
    base_config.session_dir = session_dir

    plan = ComparativeExperimentPlan(
        baselines=["scratch_t2"],
        continual_methods=models,
        backbone=args.backbone,
        seeds=tuple(seeds),
        epochs=args.epochs,
    )

    output_root = Path("comparative_results")
    results, report_path = run_comparative_suite(runner, base_config, plan, output_root)

    if report_path:
        LOGGER.info("Comparative report: %s", report_path)

    aggregate_plots = generate_session_plots(session_dir, results)
    if aggregate_plots:
        LOGGER.info(
            "Generated comparative session plots: %s",
            ", ".join(f"{name}:{path}" for name, path in aggregate_plots.items()),
        )

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
        seeds=None,
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

    runner = EinstellungRunner(project_root=Path.cwd())
    seeds_list = [seed]
    session_dir = create_session_dir(Path(args.results_root), seeds_list)
    log_command(
        session_dir,
        [
            sys.executable,
            __file__,
            "--model",
            strategy,
            "--seed",
            str(seed),
        ],
    )

    results = run_models(args, [strategy], seeds_list, session_dir=session_dir, runner=runner)
    return results[0] if results else {}


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
        seeds=None,
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
    default_models = ["sgd", "derpp", "ewc_on", "gpm", "dgr"]
    runner = EinstellungRunner(project_root=Path.cwd())
    seeds_list = [args.seed]
    session_dir = create_session_dir(Path(args.results_root), seeds_list)
    log_command(
        session_dir,
        [
            sys.executable,
            __file__,
            "--comparative",
            "--seed",
            str(args.seed),
        ],
    )
    return run_comparative(
        args,
        default_models,
        seeds_list,
        session_dir=session_dir,
        runner=runner,
    )


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
    candidates = sorted(root.glob("**/timeline*.csv")) + sorted(root.glob("**/summary*.csv"))
    for candidate in candidates:
        if validate_csv_file(str(candidate)):
            return str(candidate)
    return None


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

    required_cols = {"method", "seed", "epoch_eff", "split", "acc", "top5", "loss"}
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
    for name in [
        'matplotlib',
        'matplotlib.font_manager',
        'matplotlib.pyplot',
        'matplotlib.backends'
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)

    models = [model.strip() for model in args.model.split(',') if model.strip()]
    if not models:
        LOGGER.error("Must specify at least one model to run.")
        return 1

    try:
        seeds = parse_seed_values(args)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1

    runner = EinstellungRunner(project_root=Path.cwd())
    session_dir = create_session_dir(Path(args.results_root), seeds)

    invocation = [sys.executable, Path(__file__).name]
    invocation.extend(argv if argv is not None else sys.argv[1:])
    log_command(session_dir, invocation)

    if args.comparative:
        continual_methods = [m for m in models if m != "scratch_t2"]
        results = run_comparative(
            args,
            continual_methods,
            seeds,
            session_dir=session_dir,
            runner=runner,
        )
        successes = sum(1 for r in results if r.get("success"))
        LOGGER.info(
            "Comparative run finished – %d/%d successful",
            successes,
            len(results),
        )
        return 0 if successes == len(results) else 1

    results = run_models(
        args,
        models,
        seeds,
        session_dir=session_dir,
        runner=runner,
    )

    successes = sum(1 for r in results if r.get("success"))
    total_runs = len(results)

    if successes == total_runs:
        if total_runs > 1:
            LOGGER.info("All %d runs completed successfully.", total_runs)
        return 0

    LOGGER.error("%d/%d runs failed.", total_runs - successes, total_runs)
    return 1


if __name__ == "__main__":
    sys.exit(main())
