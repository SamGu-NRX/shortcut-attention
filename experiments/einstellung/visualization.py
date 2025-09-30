"""Visualization helpers for comparative Einstellung experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eri_vis.dataset import ERITimelineDataset
from eri_vis.processing import ERITimelineProcessor, AccuracyCurve
from eri_vis.plot_dynamics import ERIDynamicsPlotter
from eri_vis.plot_heatmap import ERIHeatmapPlotter
from eri_vis.styles import PlotStyleConfig


def _with_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


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
    separate_panels: bool = False,
    *,
    baseline_method: str = "Scratch_T2",
    show_baseline_curve: bool = True,
    smoothing_window: int = 3,
    use_smoothing: bool = True,
    include_raw_variant: bool = False,
    raw_suffix: str = "_raw",
) -> Dict[str, Path]:
    """Generate comparative dynamics and heatmap plots."""
    plots: Dict[str, Path] = {}

    if not should_generate_plots(timeline_df):
        return plots

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = timeline_df.copy()
    plot_df['method'] = plot_df['method'].apply(_normalize_method_name)

    dataset = _build_dataset(plot_df)
    multi_seed = len(dataset.seeds) > 1
    style = PlotStyleConfig()
    dynamics_plotter = ERIDynamicsPlotter(style)

    normalized_baseline = _normalize_method_name(baseline_method)

    def prepare_bundle(smoothing: int, smoothing_enabled: bool):
        processor = ERITimelineProcessor(
            smoothing_window=smoothing,
            tau=tau,
            baseline_method=normalized_baseline,
            use_smoothing=smoothing_enabled,
        )
        curves = processor.compute_accuracy_curves(dataset)
        patched_curves = {k: v for k, v in curves.items() if k.endswith('_T2_shortcut_normal')}
        masked_curves = {k: v for k, v in curves.items() if k.endswith('_T2_shortcut_masked')}
        if not patched_curves or not masked_curves:
            return None
        ad_values = processor.compute_adaptation_delays(patched_curves)
        pd_series = processor.compute_performance_deficits(curves)
        sfr_series = processor.compute_sfr_relative(curves)
        return {
            'curves': curves,
            'patched': patched_curves,
            'masked': masked_curves,
            'ad': ad_values,
            'pd': pd_series,
            'sfr': sfr_series,
        }

    def render_variant(
        bundle: Dict[str, Any],
        *,
        suffix_key: str,
        key_prefix: str,
        shading_mode: str,
        shading_scale: float,
    ) -> Dict[str, Path]:
        variant_plots: Dict[str, Path] = {}

        # Combined multi-panel figure (retain for completeness)
        dynamics_path = _with_suffix(output_dir / 'eri_dynamics.pdf', suffix_key)
        fig = dynamics_plotter.create_dynamics_figure(
            patched_curves=bundle['patched'],
            masked_curves=bundle['masked'],
            pd_series=bundle['pd'],
            sfr_series=bundle['sfr'],
            ad_values=bundle['ad'],
            tau=tau,
            title='Comparative Dynamics',
            baseline_method=normalized_baseline,
            show_baseline_curve=show_baseline_curve,
            show_masked=True,
            show_confidence=multi_seed,
            shading_mode=shading_mode,
            shading_scale=shading_scale,
        )
        fig.savefig(dynamics_path, **style.save_kwargs)
        plt.close(fig)
        variant_plots[f'{key_prefix}dynamics'] = dynamics_path

        # Panel A: overall accuracy (patched/masked average)
        overall_curves = {}
        for patched_curve in bundle['patched'].values():
            method = patched_curve.method
            masked_curve = next((c for c in bundle['masked'].values() if c.method == method), None)
            if masked_curve is None:
                continue

            epochs = patched_curve.epochs
            masked_epochs = masked_curve.epochs
            if len(epochs) != len(masked_epochs) or not np.allclose(epochs, masked_epochs):
                common_epochs = np.union1d(epochs, masked_epochs)
                patched_interp = np.interp(common_epochs, epochs, patched_curve.mean_accuracy)
                masked_interp = np.interp(common_epochs, masked_epochs, masked_curve.mean_accuracy)
                patched_ci = np.interp(
                    common_epochs,
                    epochs,
                    patched_curve.conf_interval if len(patched_curve.conf_interval) == len(epochs) else np.zeros_like(epochs),
                )
                masked_ci = np.interp(
                    common_epochs,
                    masked_epochs,
                    masked_curve.conf_interval if len(masked_curve.conf_interval) == len(masked_epochs) else np.zeros_like(masked_epochs),
                )
                mean_acc = 0.5 * (patched_interp + masked_interp)
                ci = 0.5 * np.sqrt(np.square(patched_ci) + np.square(masked_ci))
                overall_curves[method] = AccuracyCurve(
                    epochs=common_epochs,
                    mean_accuracy=mean_acc,
                    conf_interval=ci,
                    method=method,
                    split='T2_shortcut_overall',
                    n_seeds=patched_curve.n_seeds,
                )
            else:
                mean_acc = 0.5 * (patched_curve.mean_accuracy + masked_curve.mean_accuracy)
                patched_ci = patched_curve.conf_interval
                masked_ci = masked_curve.conf_interval
                if len(patched_ci) != len(epochs):
                    patched_ci = np.zeros_like(mean_acc)
                if len(masked_ci) != len(masked_epochs):
                    masked_ci = np.zeros_like(mean_acc)
                ci = 0.5 * np.sqrt(np.square(patched_ci) + np.square(masked_ci))
                overall_curves[method] = AccuracyCurve(
                    epochs=epochs,
                    mean_accuracy=mean_acc,
                    conf_interval=ci,
                    method=method,
                    split='T2_shortcut_overall',
                    n_seeds=patched_curve.n_seeds,
                )

        panel_a_path = _with_suffix(output_dir / 'eri_panel_A.pdf', suffix_key)
        fig_overall = dynamics_plotter.create_overall_accuracy_figure(
            overall_curves,
            patched_reference=bundle['patched'],
            ad_values=bundle['ad'],
            tau=tau,
            baseline_method=normalized_baseline,
            show_baseline_curve=False,
            show_confidence=multi_seed,
            shading_mode=shading_mode,
            shading_scale=shading_scale,
        )
        fig_overall.savefig(panel_a_path, **style.save_kwargs)
        plt.close(fig_overall)
        variant_plots[f'{key_prefix}panel_A'] = panel_a_path
        variant_plots[f'{key_prefix}accuracy'] = panel_a_path

        # Panel B: Performance Deficit
        if bundle['pd']:
            panel_b_path = _with_suffix(output_dir / 'eri_panel_B.pdf', suffix_key)
            fig_pd = dynamics_plotter.create_pd_only_figure(
                bundle['pd'],
                title='Performance Deficit (A_S - A_CL)',
            )
            fig_pd.savefig(panel_b_path, **style.save_kwargs)
            plt.close(fig_pd)
            variant_plots[f'{key_prefix}panel_B'] = panel_b_path
            variant_plots[f'{key_prefix}pd'] = panel_b_path

        # Panel C: Relative Shortcut Forgetting
        if bundle['sfr']:
            panel_c_path = _with_suffix(output_dir / 'eri_panel_C.pdf', suffix_key)
            fig_sfr = dynamics_plotter.create_sfr_only_figure(
                bundle['sfr'],
                title='Relative Shortcut Forgetting (Δ_CL - Δ_S)',
            )
            fig_sfr.savefig(panel_c_path, **style.save_kwargs)
            plt.close(fig_sfr)
            variant_plots[f'{key_prefix}panel_C'] = panel_c_path
            variant_plots[f'{key_prefix}sfr'] = panel_c_path

        # Supplementary: patched accuracy (no confidence shading)
        patched_path = _with_suffix(output_dir / 'eri_patched_accuracy.pdf', suffix_key)
        fig_patched = dynamics_plotter.create_split_accuracy_figure(
            bundle['patched'],
            split_label='Patched Shortcut',
            baseline_method=normalized_baseline,
            show_baseline_curve=show_baseline_curve,
            include_ad=True,
            ad_values=bundle['ad'],
            tau=tau,
            title='Patched Shortcut Accuracy',
            show_confidence=multi_seed,
            shading_mode=shading_mode,
            shading_scale=shading_scale,
        )
        fig_patched.savefig(patched_path, **style.save_kwargs)
        plt.close(fig_patched)
        variant_plots[f'{key_prefix}patched_accuracy'] = patched_path

        # Supplementary: masked accuracy (no confidence shading)
        masked_path = _with_suffix(output_dir / 'eri_masked_accuracy.pdf', suffix_key)
        fig_masked = dynamics_plotter.create_split_accuracy_figure(
            bundle['masked'],
            split_label='Masked Shortcut',
            baseline_method=normalized_baseline,
            show_baseline_curve=show_baseline_curve,
            include_ad=False,
            ad_values=None,
            tau=tau,
            title='Masked Shortcut Accuracy',
            show_confidence=multi_seed,
            shading_mode=shading_mode,
            shading_scale=shading_scale,
        )
        fig_masked.savefig(masked_path, **style.save_kwargs)
        plt.close(fig_masked)
        variant_plots[f'{key_prefix}masked_accuracy'] = masked_path

        # Composite ERI score bar chart (primary variant only)
        if not suffix_key:
            eri_bar_path = output_dir / 'eri_scores.pdf'
            fig_bar = dynamics_plotter.create_eri_bar_chart(
                baseline_method=normalized_baseline,
                ad_values=bundle['ad'],
                pd_series=bundle['pd'],
                sfr_series=bundle['sfr'],
            )
            fig_bar.savefig(eri_bar_path, **style.save_kwargs)
            plt.close(fig_bar)
            variant_plots[f'{key_prefix}eri_scores'] = eri_bar_path

        return variant_plots

    primary_bundle = prepare_bundle(smoothing_window, use_smoothing)
    if primary_bundle is None:
        return plots

    plots.update(
        render_variant(
            primary_bundle,
            suffix_key="",
            key_prefix="",
            shading_mode='std' if multi_seed else 'ci',
            shading_scale=1.0,
        )
    )

    if include_raw_variant:
        raw_bundle = prepare_bundle(1, False)
        if raw_bundle is not None:
            plots.update(
                render_variant(
                    raw_bundle,
                    suffix_key=raw_suffix,
                    key_prefix="raw_",
                    shading_mode='ci',
                    shading_scale=0.5,
                )
            )

    # Optional half-shading supplementary output for smoother visuals
    plots.update(
        render_variant(
            primary_bundle,
            suffix_key="_half",
            key_prefix="half_",
            shading_mode='std' if multi_seed else 'ci',
            shading_scale=0.5,
        )
    )

    if tau_grid is None:
        tau_grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    tau_grid = list(tau_grid)
    if len(tau_grid) >= 2:
        baseline_for_heatmap = (
            normalized_baseline
            if normalized_baseline in plot_df['method'].unique()
            else plot_df['method'].iat[0]
        )
        heatmap_plotter = ERIHeatmapPlotter(style)
        try:
            sensitivity = heatmap_plotter.compute_tau_sensitivity(
                primary_bundle['curves'],
                tau_grid,
                baseline_method=baseline_for_heatmap,
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
