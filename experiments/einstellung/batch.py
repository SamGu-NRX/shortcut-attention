"""Comparative Einstellung experiment orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import shutil

from .analysis import (
    build_comparative_table,
    combine_summaries,
    combine_timelines,
    write_aggregated_outputs,
)
from .config import ExperimentConfig
from .reporting import write_comparative_report
from .runner import EinstellungRunner
from .visualization import generate_comparative_plots


@dataclass
class ComparativeExperimentPlan:
    baselines: List[str]
    continual_methods: List[str]
    backbone: str = "resnet18"
    seeds: Tuple[int, ...] = (42,)
    epochs: int | None = None

    def strategies(self) -> Iterable[str]:
        return [*self.baselines, *self.continual_methods]


def _resolve_run_directory(
    root: Optional[Path],
    strategy: str,
    seed: int,
    nested: bool,
) -> Optional[Path]:
    if root is None:
        return None
    if not nested:
        return root
    return root / strategy / f"seed{seed}"


def run_comparative_suite(
    runner: EinstellungRunner,
    base_config: ExperimentConfig,
    plan: ComparativeExperimentPlan,
    output_root: Path,
) -> Tuple[List[dict], Path]:
    """Run the comparative suite and return individual results + report path."""
    results: List[dict] = []

    strategies = list(plan.strategies())
    seeds = plan.seeds if plan.seeds else (base_config.seed,)
    nested = len(strategies) > 1 or len(seeds) > 1

    for seed in seeds:
        for strategy in strategies:
            cfg = replace(
                base_config,
                strategy=strategy,
                backbone=plan.backbone,
                seed=seed,
                epochs=plan.epochs if plan.epochs is not None else base_config.epochs,
                session_dir=_resolve_run_directory(
                    base_config.session_dir,
                    strategy,
                    seed,
                    nested,
                ),
            )
            cfg.output_prefix = strategy

            result = runner.run(cfg)
            result.setdefault('output_dir', result.get('results_dir'))
            results.append(result)

    summary_df = combine_summaries(results)
    aggregated_paths = write_aggregated_outputs(summary_df, output_root)
    table = build_comparative_table(summary_df)
    report_path = write_comparative_report(
        comparative_table=table,
        summary_df=summary_df,
        output_dir=output_root,
    )

    timeline_df = combine_timelines(results)
    plot_paths = {}
    if not timeline_df.empty:
        session_dir = base_config.session_dir or output_root
        plots_dir = session_dir / "plots"
        try:
            plot_paths = generate_comparative_plots(timeline_df, plots_dir)
        except Exception as exc:  # pragma: no cover - defensive
            logging.getLogger("einstellung.batch").warning(
                "Comparative plot generation failed: %s", exc
            )

    combined_outputs = {k: str(v) for k, v in aggregated_paths.items()}
    combined_outputs.update({k: str(v) for k, v in plot_paths.items()})

    session_dir = base_config.session_dir
    if session_dir:
        session_dir.mkdir(parents=True, exist_ok=True)

        for path in aggregated_paths.values():
            try:
                target = session_dir / Path(path).name
                shutil.copy2(path, target)
            except (OSError, IOError):  # pragma: no cover
                logging.getLogger("einstellung.batch").debug("Unable to copy %s", path)

        try:
            if report_path:
                shutil.copy2(report_path, session_dir / Path(report_path).name)
        except (OSError, IOError):
            logging.getLogger("einstellung.batch").debug("Unable to copy report %s", report_path)

        for path in plot_paths.values():
            try:
                target = session_dir / Path(path).name
                shutil.copy2(path, target)
            except (OSError, IOError):
                logging.getLogger("einstellung.batch").debug("Unable to copy plot %s", path)

    for result in results:
        if combined_outputs:
            result.setdefault("comparative_outputs", {}).update(combined_outputs)

    return results, report_path
