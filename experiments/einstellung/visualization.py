"""Visualization helpers for comparative Einstellung experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eri_vis.dataset import ERITimelineDataset
from eri_vis.processing import ERITimelineProcessor
from eri_vis.plot_dynamics import ERIDynamicsPlotter
from eri_vis.plot_heatmap import ERIHeatmapPlotter
from eri_vis.styles import PlotStyleConfig


def _normalize_method_name(method: str) -> str:
    mapping = {
        'scratch_t2': 'Scratch_T2',
        'Scratch_t2': 'Scratch_T2',
        'interleaved': 'Interleaved',
    }
    return mapping.get(method, method)


def should_generate_plots(timeline_df: pd.DataFrame) -> bool:
    if timeline_df.empty:
        return False

    # Require at least two distinct epochs overall
    if timeline_df['epoch_eff'].nunique() <= 1:
        return False

    # Require at least one method with more than one epoch
    epoch_counts = timeline_df.groupby('method')['epoch_eff'].nunique()
    return (epoch_counts > 1).any()


def _build_dataset(timeline_df: pd.DataFrame) -> ERITimelineDataset:
    data = timeline_df.copy()
    methods = sorted(data['method'].unique())
    splits = sorted(data['split'].unique())
    seeds = sorted(data['seed'].unique())
    epoch_range = (
        float(data['epoch_eff'].min()),
        float(data['epoch_eff'].max())
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


def generate_comparative_plots(
    timeline_df: pd.DataFrame,
    output_dir: Path,
    tau: float = 0.6,
    tau_grid: Optional[Iterable[float]] = None,
) -> Dict[str, Path]:
    """Generate comparative dynamics and heatmap plots."""
    plots: Dict[str, Path] = {}

    if not should_generate_plots(timeline_df):
        return plots

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = timeline_df.copy()
    plot_df['method'] = plot_df['method'].apply(_normalize_method_name)

    dataset = _build_dataset(plot_df)
    processor = ERITimelineProcessor(smoothing_window=3, tau=tau)

    curves = processor.compute_accuracy_curves(dataset)
    patched_curves = {k: v for k, v in curves.items() if k.endswith('_T2_shortcut_normal')}
    masked_curves = {k: v for k, v in curves.items() if k.endswith('_T2_shortcut_masked')}

    if not patched_curves or not masked_curves:
        return plots

    ad_values = processor.compute_adaptation_delays(patched_curves)
    pd_series = processor.compute_performance_deficits(curves)
    sfr_series = processor.compute_sfr_relative(curves)

    style = PlotStyleConfig()
    dynamics_plotter = ERIDynamicsPlotter(style)

    dynamics_path = output_dir / 'eri_dynamics.pdf'
    fig = dynamics_plotter.create_dynamics_figure(
        patched_curves=patched_curves,
        masked_curves=masked_curves,
        pd_series=pd_series,
        sfr_series=sfr_series,
        ad_values=ad_values,
        tau=tau,
        title='Comparative Dynamics'
    )
    fig.savefig(dynamics_path, **style.save_kwargs)
    plt.close(fig)
    plots['dynamics'] = dynamics_path

    accuracy_path = output_dir / 'eri_accuracy.pdf'
    fig_acc = dynamics_plotter.create_accuracy_only_figure(
        patched_curves=patched_curves,
        masked_curves=masked_curves,
        ad_values=ad_values,
        tau=tau,
        title='Accuracy Trajectories on Shortcut Task'
    )
    fig_acc.savefig(accuracy_path, **style.save_kwargs)
    plt.close(fig_acc)
    plots['accuracy'] = accuracy_path

    if tau_grid is None:
        tau_grid = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    tau_grid = list(tau_grid)
    if len(tau_grid) >= 2:
        baseline_method = 'Scratch_T2' if 'Scratch_T2' in plot_df['method'].unique() else plot_df['method'].iat[0]
        heatmap_plotter = ERIHeatmapPlotter(style)
        try:
            sensitivity = heatmap_plotter.compute_tau_sensitivity(
                curves, tau_grid, baseline_method=baseline_method
            )
        except ValueError:
            sensitivity = None

        if sensitivity is not None and len(sensitivity.methods) > 0:
            heatmap_path = output_dir / 'comparative_heatmap.pdf'
            fig = heatmap_plotter.create_tau_sensitivity_heatmap(sensitivity)
            fig.savefig(heatmap_path, **style.save_kwargs)
            plt.close(fig)
            plots['heatmap'] = heatmap_path

    return plots
