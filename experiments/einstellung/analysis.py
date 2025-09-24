"""Post-processing helpers for Einstellung experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


def combine_summaries(results: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Concatenate per-run summary CSVs into a single DataFrame."""
    frames: List[pd.DataFrame] = []
    for result in results:
        if not result.get("success"):
            continue
        summary_path = result.get("summary_path")
        if not summary_path:
            continue
        summary_df = pd.read_csv(summary_path)
        summary_df["strategy"] = result.get("strategy")
        summary_df["backbone"] = result.get("backbone")
        summary_df["seed"] = result.get("seed")
        frames.append(summary_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def write_aggregated_outputs(summary_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Persist aggregated summary CSVs for comparative analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "comparative_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    topline_path = output_dir / "comparative_topline.csv"
    topline = summary_df[summary_df["subset"] == "T2_shortcut_normal"].copy()
    topline.to_csv(topline_path, index=False)

    return {"summary": summary_path, "topline": topline_path}


def build_comparative_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate key metrics per strategy/backbone for reporting tables."""
    if summary_df.empty:
        return summary_df

    topline = summary_df[summary_df["subset"] == "T2_shortcut_normal"].copy()
    grouped = (
        topline.groupby(["strategy", "backbone"], as_index=False)
        .agg(
            {
                "top1": "mean",
                "top5": "mean",
                "top1_delta": "mean",
                "top5_delta": "mean",
                "performance_deficit": "mean",
                "shortcut_feature_reliance": "mean",
                "adaptation_delay": "mean",
            }
        )
    )

    grouped = grouped.rename(
        columns={
            "top1": "top1_final",
            "top5": "top5_final",
        }
    )

    return grouped
