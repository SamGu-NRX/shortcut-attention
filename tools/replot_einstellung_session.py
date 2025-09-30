#!/usr/bin/env python3
"""Regenerate ERI dynamics plots from an existing Einstellung session."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

# Ensure project root (one level up from tools/) is importable when script is run directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.einstellung.visualization import generate_comparative_plots

LOGGER = logging.getLogger("einstellung.replot")
REQUIRED_COLUMNS = {"method", "seed", "epoch_eff", "split", "acc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate ERI plots from an existing Einstellung session directory."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path.cwd(),
        help="Path to a session directory containing timeline_*.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output directory (defaults to <input>/plots_rebuilt)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.6,
        help="Default threshold for Adaptation Delay when --metric is not provided",
    )
    parser.add_argument(
        "--tau-grid",
        type=float,
        nargs="+",
        help="Optional list of tau values for sensitivity heatmap",
    )
    parser.add_argument(
        "--metric",
        action="append",
        help=(
            "Metric column(s) to plot. Use form <column>[:tau]. "
            "Supported columns include 'acc' (top-1) and 'top5'. "
            "Repeat flag to generate multiple metric-specific outputs."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional list of methods to include (case-sensitive)",
    )
    parser.add_argument(
        "--separate-panels",
        action="store_true",
        help="Also generate standalone PD and SFR plots",
    )
    parser.add_argument(
        "--baseline",
        default="sgd",
        help="Method name to use as baseline/reference (default: sgd)",
    )
    parser.add_argument(
        "--show-baseline-curve",
        action="store_true",
        help="Render the baseline accuracy curve (default: reference guides only)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=3,
        help="Smoothing window size for AD/PD/SFR computation (default: 3)",
    )
    parser.add_argument(
        "--raw-suffix",
        default="_raw",
        help="Filename suffix for unsmoothed plots (default: _raw)",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip generating the unsmoothed plot variants",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def discover_timelines(root: Path) -> List[Path]:
    return sorted(p for p in root.glob("**/timeline_*.csv") if p.is_file())


def load_timelines(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Skipping %s (could not read: %s)", path, exc)
            continue

        if not REQUIRED_COLUMNS.issubset(df.columns):
            LOGGER.warning("Skipping %s (missing expected columns)", path)
            continue

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No valid timeline_*.csv files found")

    return pd.concat(frames, ignore_index=True)


def parse_metric_specs(
    specs: Optional[List[str]],
    default_tau: float,
) -> List[Tuple[str, float]]:
    if not specs:
        return [("acc", default_tau)]

    parsed: List[Tuple[str, float]] = []
    for spec in specs:
        if not spec:
            continue
        parts = spec.split(":", 1)
        column = parts[0].strip()
        if not column:
            raise ValueError("Metric specification requires a column name")

        tau_val = default_tau
        if len(parts) == 2 and parts[1].strip():
            try:
                tau_val = float(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid tau value in metric specification '{spec}'") from exc

        parsed.append((column, tau_val))

    if not parsed:
        raise ValueError("No valid metric specifications provided")

    return parsed


def prepare_metric_dataframe(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column == "acc":
        metric_df = df.copy()
    else:
        if column not in df.columns:
            raise KeyError(f"Timeline data does not contain column '{column}'")
        metric_df = df.copy()
        metric_df["acc"] = metric_df[column]

    metric_df = metric_df.dropna(subset=["acc"])
    metric_df = metric_df.drop_duplicates(
        subset=["method", "seed", "epoch_eff", "split"], keep="last"
    )
    return metric_df


def filter_methods(df: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    methods_set = {m.strip() for m in methods if m.strip()}
    if not methods_set:
        return df

    missing = methods_set - set(df["method"].unique())
    if missing:
        LOGGER.warning(
            "Requested methods not present in data: %s",
            ", ".join(sorted(missing)),
        )

    filtered = df[df["method"].isin(methods_set)]
    if filtered.empty:
        raise ValueError("No data remains after method filtering")
    return filtered


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    LOGGER.debug("Arguments: %s", args)

    timeline_paths = discover_timelines(args.input)
    if not timeline_paths:
        LOGGER.error("No timeline_*.csv files found under %s", args.input)
        return 1

    LOGGER.info("Found %d timeline files", len(timeline_paths))

    try:
        timeline_df = load_timelines(timeline_paths)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    if args.smoothing_window < 1:
        LOGGER.error("smoothing_window must be >= 1")
        return 1

    if args.methods:
        try:
            timeline_df = filter_methods(timeline_df, args.methods)
        except ValueError as exc:
            LOGGER.error("%s", exc)
            return 1

    try:
        metric_specs = parse_metric_specs(args.metric, args.tau)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1

    output_root = args.output or args.input
    output_root.mkdir(parents=True, exist_ok=True)

    all_plots: List[Tuple[str, Path]] = []

    for idx, (column, tau) in enumerate(metric_specs):
        try:
            metric_df = prepare_metric_dataframe(timeline_df, column)
        except (KeyError, ValueError) as exc:
            LOGGER.error("Skipping metric '%s': %s", column, exc)
            continue

        if metric_df.empty:
            LOGGER.warning("Skipping metric '%s' (no data after filtering)", column)
            continue

        if idx == 0 and column == metric_specs[0][0] and column == "acc":
            output_dir = output_root
        else:
            output_dir = output_root / column
            output_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Generating plots for metric '%s' with tau=%.3f", column, tau)
        plots = generate_comparative_plots(
            metric_df,
            output_dir,
            tau=tau,
            tau_grid=args.tau_grid,
            separate_panels=args.separate_panels,
            baseline_method=args.baseline,
            show_baseline_curve=args.show_baseline_curve,
            smoothing_window=args.smoothing_window,
            use_smoothing=True,
            include_raw_variant=not args.no_raw,
            raw_suffix=args.raw_suffix,
        )

        if not plots:
            LOGGER.warning("No plots generated for metric '%s'", column)
            continue

        all_plots.extend((f"{column}:{name}", path) for name, path in plots.items())

    if not all_plots:
        LOGGER.error("No plots were generated. Check input data and metric selections.")
        return 1

    LOGGER.info(
        "Generated plots:\n%s",
        "\n".join(f"- {label}: {path}" for label, path in all_plots),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
