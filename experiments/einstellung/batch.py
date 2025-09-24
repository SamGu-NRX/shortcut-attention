"""Comparative Einstellung experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Tuple

from .analysis import build_comparative_table, combine_summaries, write_aggregated_outputs
from .config import ExperimentConfig
from .reporting import write_comparative_report
from .runner import EinstellungRunner


@dataclass
class ComparativeExperimentPlan:
    baselines: List[str]
    continual_methods: List[str]
    backbone: str = "resnet18"
    seed: int = 42
    epochs: int | None = None

    def strategies(self) -> Iterable[str]:
        return [*self.baselines, *self.continual_methods]


def run_comparative_suite(
    runner: EinstellungRunner,
    base_config: ExperimentConfig,
    plan: ComparativeExperimentPlan,
    output_root: Path,
) -> Tuple[List[dict], Path]:
    """Run the comparative suite and return individual results + report path."""
    results: List[dict] = []

    for strategy in plan.strategies():
        cfg = replace(
            base_config,
            strategy=strategy,
            backbone=plan.backbone,
            seed=plan.seed,
            epochs=plan.epochs if plan.epochs is not None else base_config.epochs,
        )

        result = runner.run(cfg)
        results.append(result)

    summary_df = combine_summaries(results)
    aggregated_paths = write_aggregated_outputs(summary_df, output_root)
    table = build_comparative_table(summary_df)
    report_path = write_comparative_report(
        comparative_table=table,
        summary_df=summary_df,
        output_dir=output_root,
    )

    for key, path in aggregated_paths.items():
        for result in results:
            result.setdefault("comparative_outputs", {})[key] = str(path)

    return results, report_path
