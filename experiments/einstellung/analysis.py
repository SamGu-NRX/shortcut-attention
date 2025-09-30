"""Post-processing helpers for Einstellung experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from eri_vis.dataset import ERITimelineDataset
from eri_vis.processing import ERITimelineProcessor


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


def combine_timelines(results: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Concatenate per-run timeline CSVs into a single DataFrame."""
    frames: List[pd.DataFrame] = []

    for result in results:
        if not result.get("success"):
            continue

        timeline_path = result.get("timeline_path")
        if not timeline_path:
            continue

        try:
            df = pd.read_csv(timeline_path)
        except FileNotFoundError:
            continue

        df = df.copy()
        df["strategy"] = result.get("strategy")
        df["backbone"] = result.get("backbone")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def _normalize_method_name(method: str) -> str:
    mapping = {
        'scratch_t2': 'Scratch_T2',
        'Scratch_t2': 'Scratch_T2',
        'interleaved': 'Interleaved',
    }
    return mapping.get(method, method)


def _build_dataset(timeline_df: pd.DataFrame) -> ERITimelineDataset:
    data = timeline_df.copy()
    methods = sorted(data['method'].unique())
    splits = sorted(data['split'].unique())
    seeds = sorted(data['seed'].unique())
    epoch_range = (
        float(data['epoch_eff'].min()),
        float(data['epoch_eff'].max()),
    )

    metadata = {
        'source': 'comparative_analysis',
        'n_methods': len(methods),
        'n_splits': len(splits),
        'n_seeds': len(seeds),
        'n_rows': len(data),
        'epoch_range': epoch_range,
    }

    return ERITimelineDataset(
        data=data,
        metadata=metadata,
        methods=methods,
        splits=splits,
        seeds=seeds,
        epoch_range=epoch_range,
    )


def build_publication_metrics(
    summary_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    *,
    baseline_method: str = 'Scratch_T2',
    smoothing_window: int = 3,
    tau: float = 0.6,
    use_smoothing: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute publication-ready seed-level and aggregated ERI metrics."""

    if summary_df.empty or timeline_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    normalized_baseline = _normalize_method_name(baseline_method)

    summary = summary_df.copy()
    summary['method'] = summary['method'].apply(_normalize_method_name)

    timeline = timeline_df.copy()
    timeline['method'] = timeline['method'].apply(_normalize_method_name)

    if normalized_baseline not in summary['method'].unique():
        return pd.DataFrame(), pd.DataFrame()

    dataset = _build_dataset(timeline)

    processor = ERITimelineProcessor(
        smoothing_window=smoothing_window,
        tau=tau,
        baseline_method=normalized_baseline,
        use_smoothing=use_smoothing,
    )
    curves = processor.compute_accuracy_curves(dataset)
    ad_values = processor.compute_adaptation_delays(curves)
    ad_seedwise = processor.get_seedwise_ad()

    subset_alias = {
        'T2_shortcut_normal': 'A_SC_patch',
        'T2_shortcut_masked': 'A_SC_mask',
        'T1_all': 'A_T1',
        'T2_nonshortcut_normal': 'A_NSC_patch',
    }

    accuracy_wide = (
        summary[summary['subset'].isin(subset_alias.keys())]
        .pivot_table(index=['method', 'seed'], columns='subset', values='top1', aggfunc='mean')
        .rename(columns=subset_alias)
        .reset_index()
    )

    if accuracy_wide.empty:
        return pd.DataFrame(), pd.DataFrame()

    baseline_df = accuracy_wide[accuracy_wide['method'] == normalized_baseline].set_index('seed')

    per_seed_rows: List[Dict[str, Any]] = []

    for method in sorted(accuracy_wide['method'].unique()):
        method_rows = accuracy_wide[accuracy_wide['method'] == method].set_index('seed')

        ad_info = ad_seedwise.get(method, {})
        ad_seeds = ad_info.get('seed_ids')
        ad_values_array = ad_info.get('values')
        ad_map = {}
        if isinstance(ad_seeds, np.ndarray) and isinstance(ad_values_array, np.ndarray):
            ad_map = dict(zip(ad_seeds.tolist(), ad_values_array.tolist()))
        elif isinstance(ad_values_array, list) and isinstance(ad_seeds, list):
            ad_map = dict(zip(ad_seeds, ad_values_array))

        censored = ad_info.get('censored')
        censored_set = set(censored.tolist() if isinstance(censored, np.ndarray) else (censored or []))

        for seed, values in method_rows.iterrows():
            baseline_values = baseline_df.loc[seed] if seed in baseline_df.index else None

            a_sc_patch = float(values.get('A_SC_patch')) if not pd.isna(values.get('A_SC_patch')) else np.nan
            a_sc_mask = float(values.get('A_SC_mask')) if not pd.isna(values.get('A_SC_mask')) else np.nan
            a_t1 = float(values.get('A_T1')) if not pd.isna(values.get('A_T1')) else np.nan
            a_nsc_patch = float(values.get('A_NSC_patch')) if not pd.isna(values.get('A_NSC_patch')) else np.nan

            if method == normalized_baseline:
                ad_val = 0.0
            else:
                ad_val = ad_map.get(seed, np.nan)
                if seed in censored_set:
                    ad_val = np.nan
            if method == normalized_baseline and np.isnan(ad_values.get(method, np.nan)):
                ad_val = 0.0

            if baseline_values is not None:
                baseline_patch = baseline_values.get('A_SC_patch')
                baseline_mask = baseline_values.get('A_SC_mask')
            else:
                baseline_patch = np.nan
                baseline_mask = np.nan

            pd_val = (
                float(baseline_patch) - a_sc_patch
                if not pd.isna(baseline_patch) and not np.isnan(a_sc_patch)
                else np.nan
            )

            baseline_delta = (
                float(baseline_patch) - float(baseline_mask)
                if not pd.isna(baseline_patch) and not pd.isna(baseline_mask)
                else np.nan
            )
            method_delta = (
                a_sc_patch - a_sc_mask
                if not np.isnan(a_sc_patch) and not np.isnan(a_sc_mask)
                else np.nan
            )

            sfr_rel = (
                method_delta - baseline_delta
                if not np.isnan(method_delta) and not np.isnan(baseline_delta)
                else np.nan
            )

            per_seed_rows.append(
                {
                    'method': method,
                    'seed': int(seed),
                    'AD': ad_val,
                    'PD': pd_val,
                    'SFR_rel': sfr_rel,
                    'A_SC_patch': a_sc_patch,
                    'A_SC_mask': a_sc_mask,
                    'A_T1': a_t1,
                    'A_NSC_patch': a_nsc_patch,
                    'AD_censored': seed in censored_set,
                }
            )

    if not per_seed_rows:
        return pd.DataFrame(), pd.DataFrame()

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df = per_seed_df.sort_values(['method', 'seed']).reset_index(drop=True)

    numeric_cols = ['AD', 'PD', 'SFR_rel', 'A_SC_patch', 'A_SC_mask', 'A_T1', 'A_NSC_patch']
    grouped = per_seed_df.groupby('method')

    summary_parts = []
    for method, group in grouped:
        stats: Dict[str, Any] = {
            'method': method,
            'n_seeds': int(group['seed'].nunique()),
        }
        for col in numeric_cols:
            col_values = group[col].dropna()
            stats[f'{col}_mean'] = float(col_values.mean()) if not col_values.empty else np.nan
            stats[f'{col}_std'] = float(col_values.std(ddof=1)) if len(col_values) > 1 else 0.0 if len(col_values) == 1 else np.nan
        summary_parts.append(stats)

    summary_df_out = pd.DataFrame(summary_parts).sort_values('method').reset_index(drop=True)

    return per_seed_df, summary_df_out


def write_aggregated_outputs(
    summary_df: pd.DataFrame,
    output_dir: Path,
    timeline_df: Optional[pd.DataFrame] = None,
    *,
    baseline_method: str = 'Scratch_T2',
) -> Dict[str, Path]:
    """Persist aggregated summary CSVs for comparative analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "comparative_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    topline_path = output_dir / "comparative_topline.csv"
    topline = summary_df[summary_df["subset"] == "T2_shortcut_normal"].copy()
    topline.to_csv(topline_path, index=False)

    outputs: Dict[str, Path] = {
        "summary": summary_path,
        "topline": topline_path,
    }

    if timeline_df is not None and not timeline_df.empty:
        per_seed_df, summary_metrics_df = build_publication_metrics(
            summary_df=summary_df,
            timeline_df=timeline_df,
            baseline_method=baseline_method,
        )

        if not per_seed_df.empty:
            per_seed_path = output_dir / "publication_metrics_per_seed.csv"
            per_seed_df.to_csv(per_seed_path, index=False)
            outputs["publication_per_seed"] = per_seed_path

        if not summary_metrics_df.empty:
            summary_metrics_path = output_dir / "publication_metrics_summary.csv"
            summary_metrics_df.to_csv(summary_metrics_path, index=False)
            outputs["publication_summary"] = summary_metrics_path

    return outputs


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
