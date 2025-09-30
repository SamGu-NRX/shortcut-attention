"""
ERIDynamicsPlotter - Main visualization generator for ERI dynamics.

This module provides the ERIDynamicsPlotter class for generating publication-ready
3-panel figures showing ERI dynamics including accuracy trajectories, performance
deficits, and shortcut forgetting rates.

Integrates with the existing Mammoth Einstellung experiment infrastructure:
- Uses data structures from eri_vis.processing (AccuracyCurve, TimeSeries)
- Compatible with eri_.styles (PlotStyleConfig)
- Generates publication-ready PDF outputs under 5MB
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .processing import AccuracyCurve, TimeSeries
from .styles import PlotStyleConfig, DEFAULT_STYLE


class ERIDynamicsPlotter:
    """
    Main visualization generator for ERI dynamics.

    This class creates publication-ready 3-panel figures showing:
    - Panel A: Accuracy trajectories with confidence intervals and AD markers
    - Panel B: Performance Deficit (PD_t) time series with zero reference line
    - Panel C: Shortcut Forgetting Rate (SFR_rel) time series with zero reference line
    """

    def __init__(self, style: Optional[PlotStyleConfig] = None):
        """
        Initialize the plotter.

        Args:
            style: PlotStyleConfig instance (uses DEFAULT_STYLE if None)
        """
        self.style = style if style is not None else DEFAULT_STYLE
        self.logger = logging.getLogger(__name__)

    def _finalize_layout(
        self,
        fig: Figure,
        *,
        top: float = 0.94,
        bottom: float = 0.12,
        left: float = 0.12,
        right: float = 0.98,
        rect: Optional[Tuple[float, float, float, float]] = None,
        apply_tight: bool = True,
    ) -> None:
        """Apply consistent margins and optional tight layout before saving."""

        if apply_tight and self.style.tight_layout:
            if rect is not None:
                fig.tight_layout(rect=rect)
            else:
                fig.tight_layout()

        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    def create_dynamics_figure(
        self,
        patched_curves: Dict[str, AccuracyCurve],
        masked_curves: Dict[str, AccuracyCurve],
        pd_series: Dict[str, TimeSeries],
        sfr_series: Dict[str, TimeSeries],
        ad_values: Dict[str, float],
        tau: float,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        *,
        baseline_method: str = "Scratch_T2",
        show_baseline_curve: bool = True,
        show_masked: bool = True,
        show_confidence: bool = True,
        shading_mode: str = "ci",
        shading_scale: float = 1.0,
    ) -> Figure:
        """
        Create the main 3-panel ERI dynamics figure.

        Args:
            patched_curves: Dictionary of AccuracyCurve objects for shortcut_normal split
            masked_curves: Dictionary of AccuracyCurve objects for shortcut_masked split
            pd_series: Dictionary of TimeSeries objects for Performance Deficit
            sfr_series: Dictionary of TimeSeries objects for Shortcut Forgetting Rate
            ad_values: Dictionary of Adaptation Delay values per method
            tau: Threshold value used for AD computation
            title: Optional figure title
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object

        Raises:
            ValueError: If required data is missing or invalid
        """
        # Validate inputs
        self._validate_inputs(patched_curves, masked_curves, pd_series, sfr_series, ad_values)

        # Apply style configuration
        self.style.apply_style()

        # Create figure with 3 panels
        fig, (ax_a, ax_b, ax_c) = plt.subplots(
            3, 1,
            figsize=self.style.figure_size,
            dpi=self.style.dpi
        )

        # Panel A: Accuracy trajectories with CI and AD markers
        self._create_panel_a(
            ax_a,
            patched_curves,
            masked_curves,
            ad_values,
            tau,
            baseline_method=baseline_method,
            show_baseline_curve=show_baseline_curve,
            show_masked=show_masked,
            include_ad=True,
            panel_label='A',
            ad_reference_curves=None,
            show_confidence=show_confidence,
            shading_mode=shading_mode,
            shading_scale=shading_scale,
        )
        ax_a.set_title('Accuracy Trajectories on Shortcut Task', fontsize=self.style.font_sizes['title'])

        # Panel B: Performance Deficit (PD_t) time series
        self._create_panel_b(ax_b, pd_series, panel_label='B')

        # Panel C: Shortcut Forgetting Rate (SFR_rel) time series
        self._create_panel_c(ax_c, sfr_series, panel_label='C')

        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=self.style.font_sizes['title'], y=0.97)

        layout_top = 0.9 if title else 0.94
        self._finalize_layout(fig, top=layout_top, bottom=0.12)

        # Save figure if path provided
        if save_path:
            self.save_figure(fig, save_path)

        return fig

    def create_accuracy_only_figure(
        self,
        patched_curves: Dict[str, AccuracyCurve],
        masked_curves: Dict[str, AccuracyCurve],
        ad_values: Dict[str, float],
        tau: float,
        title: Optional[str] = None,
        *,
        baseline_method: str = "Scratch_T2",
        show_baseline_curve: bool = True,
        show_masked: bool = True,
        include_ad: bool = True,
        show_confidence: bool = True,
        shading_mode: str = "ci",
        shading_scale: float = 1.0,
    ) -> plt.Figure:
        """Generate a single-panel accuracy trajectory figure."""
        self._validate_inputs(patched_curves, masked_curves, {}, {}, ad_values)
        self.style.apply_style()

        fig, ax = plt.subplots(1, 1, figsize=self.style.figure_size, dpi=self.style.dpi)
        self._create_panel_a(
            ax,
            patched_curves,
            masked_curves,
            ad_values,
            tau,
            baseline_method=baseline_method,
            show_baseline_curve=show_baseline_curve,
            show_masked=show_masked,
            include_ad=include_ad,
            panel_label=None,
            ad_reference_curves=None,
            show_confidence=show_confidence,
            shading_mode=shading_mode,
            shading_scale=shading_scale,
        )

        if title:
            ax.set_title(title, fontsize=self.style.font_sizes['title'])
        else:
            ax.set_title('Accuracy Trajectories on Shortcut Task', fontsize=self.style.font_sizes['title'])

        self._finalize_layout(fig)
        return fig

    def create_split_accuracy_figure(
        self,
        curves: Dict[str, AccuracyCurve],
        *,
        split_label: str,
        baseline_method: str = "Scratch_T2",
        show_baseline_curve: bool = True,
        include_ad: bool = False,
        ad_values: Optional[Dict[str, float]] = None,
        tau: float = 0.6,
        title: Optional[str] = None,
        show_confidence: bool = True,
        shading_mode: str = "ci",
        shading_scale: float = 1.0,
    ) -> plt.Figure:
        """Generate a single-panel accuracy figure for a specific split."""
        self.style.apply_style()
        fig, ax = plt.subplots(1, 1, figsize=self.style.figure_size, dpi=self.style.dpi)

        patched_curves = curves
        masked_curves: Dict[str, AccuracyCurve] = {}
        ad_map = ad_values if include_ad and ad_values is not None else {}

        self._create_panel_a(
            ax,
            patched_curves,
            masked_curves,
            ad_map,
            tau,
            baseline_method=baseline_method,
            show_baseline_curve=show_baseline_curve,
            show_masked=False,
            include_ad=include_ad,
            panel_label=None,
            ad_reference_curves=None,
            show_confidence=show_confidence,
            shading_mode=shading_mode,
            shading_scale=shading_scale,
        )

        if title is None:
            title = f"{split_label} Accuracy"

        ax.set_title(title, fontsize=self.style.font_sizes['title'])
        self._finalize_layout(fig)
        return fig

    def create_overall_accuracy_figure(
        self,
        overall_curves: Dict[str, AccuracyCurve],
        *,
        patched_reference: Dict[str, AccuracyCurve],
        ad_values: Dict[str, float],
        tau: float,
        baseline_method: str = "Scratch_T2",
        show_baseline_curve: bool = False,
        title: Optional[str] = None,
        show_confidence: bool = False,
        shading_mode: str = "ci",
        shading_scale: float = 1.0,
    ) -> plt.Figure:
        """Generate a single-panel overall shortcut accuracy figure."""
        self.style.apply_style()
        fig, ax = plt.subplots(1, 1, figsize=self.style.figure_size, dpi=self.style.dpi)

        self._create_panel_a(
            ax,
            overall_curves,
            {},
            ad_values,
            tau,
            baseline_method=baseline_method,
            show_baseline_curve=show_baseline_curve,
            show_masked=False,
            include_ad=True,
            panel_label=None,
            ad_reference_curves=patched_reference,
            show_confidence=show_confidence,
            shading_mode=shading_mode,
            shading_scale=shading_scale,
        )

        title = title or 'Shortcut Accuracy (Patched/Masked Average)'
        ax.set_title(title, fontsize=self.style.font_sizes['title'])
        self._finalize_layout(fig)
        return fig

    def create_pd_only_figure(
        self,
        pd_series: Dict[str, TimeSeries],
        *,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Generate a single-panel Performance Deficit figure."""
        self.style.apply_style()
        fig, ax = plt.subplots(1, 1, figsize=self.style.figure_size, dpi=self.style.dpi)
        self._create_panel_b(ax, pd_series, panel_label=None)
        if title:
            ax.set_title(title, fontsize=self.style.font_sizes['title'])
        self._finalize_layout(fig)
        return fig

    def create_sfr_only_figure(
        self,
        sfr_series: Dict[str, TimeSeries],
        *,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Generate a single-panel Shortcut Forgetting Rate figure."""
        self.style.apply_style()
        fig, ax = plt.subplots(1, 1, figsize=self.style.figure_size, dpi=self.style.dpi)
        self._create_panel_c(ax, sfr_series, panel_label=None)
        if title:
            ax.set_title(title, fontsize=self.style.font_sizes['title'])
        self._finalize_layout(fig)
        return fig

    def _validate_inputs(
        self,
        patched_curves: Dict[str, AccuracyCurve],
        masked_curves: Dict[str, AccuracyCurve],
        pd_series: Dict[str, TimeSeries],
        sfr_series: Dict[str, TimeSeries],
        ad_values: Dict[str, float]
    ) -> None:
        """Validate input data for figure creation."""
        if not patched_curves:
            raise ValueError("patched_curves cannot be empty")

        if masked_curves is None:
            masked_curves = {}

        # Check that we have at least some data for panels B and C
        if not pd_series and not sfr_series:
            self.logger.warning("No PD_t or SFR_rel data provided - panels B and C will be empty")

        # Validate that curves have consistent structure
        for key, curve in patched_curves.items():
            if len(curve.epochs) != len(curve.mean_accuracy):
                raise ValueError(f"Inconsistent curve data for {key}")

    def _create_panel_a(
        self,
        ax: Axes,
        patched_curves: Dict[str, AccuracyCurve],
        masked_curves: Dict[str, AccuracyCurve],
        ad_values: Dict[str, float],
        tau: float,
        *,
        baseline_method: str,
        show_baseline_curve: bool,
        show_masked: bool,
        include_ad: bool,
        panel_label: Optional[str] = 'A',
        ad_reference_curves: Optional[Dict[str, AccuracyCurve]] = None,
        show_confidence: bool = True,
        shading_mode: str = "ci",
        shading_scale: float = 1.0,
    ) -> None:
        """
        Create Panel A: Accuracy trajectories with confidence intervals and AD markers.

        Args:
            ax: Matplotlib axes object
            patched_curves: AccuracyCurve objects for shortcut_normal split
            masked_curves: AccuracyCurve objects for shortcut_masked split
            ad_values: Adaptation Delay values per method
            tau: Threshold value for AD computation
            baseline_method: Reference method used for AD/PD/SFR computations
            show_baseline_curve: Whether to render the baseline curve explicitly
            show_masked: Whether to overlay masked trajectories
            include_ad: Whether to draw AD annotations/markers
            panel_label: Optional panel label (e.g., 'A')
        """
        if panel_label:
            ax.text(
                0.02, 0.98, panel_label,
                transform=ax.transAxes,
                fontsize=self.style.font_sizes['panel_label'],
                fontweight='bold',
                verticalalignment='top'
            )

        baseline_lower = baseline_method.lower()
        reference_curves = ad_reference_curves or patched_curves

        baseline_curve: Optional[AccuracyCurve] = None
        for curve in reference_curves.values():
            if curve.method.lower() == baseline_lower and 'shortcut_normal' in curve.split.lower():
                baseline_curve = curve
                break

        methods_plotted = set()
        for key, curve in patched_curves.items():
            method = curve.method
            is_baseline = method.lower() == baseline_lower
            if is_baseline:
                baseline_curve = curve

            color = self.style.get_method_color(method)
            line_style = '-' if not is_baseline else ':'
            line_width = self.style.line_width if not is_baseline else max(self.style.line_width * 0.9, 1.0)
            alpha = 1.0 if show_baseline_curve or not is_baseline else 0.85

            if not show_baseline_curve and not is_baseline:
                pass

            ax.plot(
                curve.epochs, curve.mean_accuracy,
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                alpha=alpha,
                label=f"{method} (patched)"
            )

            if show_confidence:
                shading = self._get_shading_values(curve, shading_mode)
                if shading is not None and len(shading) == len(curve.epochs):
                    shade = shading_scale * shading
                    lower = np.clip(curve.mean_accuracy - shade, 0.0, 1.0)
                    upper = np.clip(curve.mean_accuracy + shade, 0.0, 1.0)
                    ax.fill_between(
                        curve.epochs,
                        lower,
                        upper,
                        color=color,
                        alpha=self.style.confidence_alpha,
                    )

            methods_plotted.add(method)

        # Plot masked curves (dashed lines) if requested
        if show_masked:
            for key, curve in masked_curves.items():
                method = curve.method
                if method not in methods_plotted:
                    continue

                color = self.style.get_method_color(method)
                line_style = '--'
                if method.lower() == baseline_lower:
                    line_style = (0, (2, 2))

                ax.plot(
                    curve.epochs, curve.mean_accuracy,
                    color=color, linewidth=self.style.line_width,
                    label=f"{method} (masked)", linestyle=line_style
                )

                if show_confidence:
                    shading = self._get_shading_values(curve, shading_mode)
                    if shading is not None and len(shading) == len(curve.epochs):
                        shade = shading_scale * shading
                        lower = np.clip(curve.mean_accuracy - shade, 0.0, 1.0)
                        upper = np.clip(curve.mean_accuracy + shade, 0.0, 1.0)
                        ax.fill_between(
                            curve.epochs,
                            lower,
                            upper,
                            color=color,
                            alpha=self.style.confidence_alpha,
                        )

        # Add threshold line
        ax.axhline(y=tau, color='gray', linestyle=':', alpha=0.7, label=f'τ = {tau}')

        # Baseline reference indicators when baseline curve is hidden
        if baseline_curve is not None and not show_baseline_curve:
            baseline_color = self.style.get_method_color(baseline_method)
            final_accuracy = baseline_curve.mean_accuracy[-1]
            ax.axhline(
                y=final_accuracy,
                color=baseline_color,
                linestyle='-.',
                alpha=0.5,
                linewidth=1,
                label='_nolegend_'
            )

        # AD markers removed per revised specification

        ax.set_xlabel('Effective Epoch', fontsize=self.style.font_sizes['axis_label'])
        ax.set_ylabel('Accuracy', fontsize=self.style.font_sizes['axis_label'])
        ax.grid(True, alpha=self.style.grid_alpha)
        ax.legend(fontsize=self.style.font_sizes['legend'], loc='lower right')

        # Set y-axis limits to show full range
        ax.set_ylim(0, 1)

    def _add_ad_markers(
        self,
        ax: Axes,
        patched_curves: Dict[str, AccuracyCurve],
        ad_values: Dict[str, float],
        tau: float,
        *,
        baseline_curve: Optional[AccuracyCurve],
        baseline_method: str,
        show_baseline_curve: bool,
    ) -> None:
        """
        Add AD markers and annotations to Panel A.

        Args:
            ax: Matplotlib axes object
            patched_curves: AccuracyCurve objects for shortcut_normal split
            ad_values: Adaptation Delay values per method
            tau: Threshold value
        """
        if baseline_curve is None:
            return

        baseline_crossing_epoch = self._find_threshold_crossing(baseline_curve, tau)
        if np.isnan(baseline_crossing_epoch):
            return

        baseline_color = self.style.get_method_color(baseline_method)
        ax.axvline(
            x=baseline_crossing_epoch,
            color=baseline_color,
            linestyle='--' if show_baseline_curve else ':',
            alpha=0.6,
            linewidth=1,
            label='_nolegend_'
        )

        ax.plot(
            baseline_crossing_epoch,
            tau,
            marker='o',
            color=baseline_color,
            markersize=4,
            alpha=0.9,
        )
        baseline_vertical_offset = 0.045
        ax.text(
            baseline_crossing_epoch,
            min(tau + baseline_vertical_offset, 0.96),
            f"{baseline_method} E(τ)={baseline_crossing_epoch:.1f}",
            color=baseline_color,
            fontsize=self.style.font_sizes['annotation'],
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=baseline_color, alpha=0.5),
        )

        label_offsets = {}

        def _get_label_position(x_value: float) -> float:
            if not label_offsets:
                label_offsets[x_value] = 0
                return 0
            existing = sorted(label_offsets.keys())
            min_sep = 0.6
            closest = min(existing, key=lambda xv: abs(xv - x_value))
            if abs(closest - x_value) < min_sep:
                delta = min_sep - abs(closest - x_value)
                direction = 1 if x_value >= closest else -1
                adjusted = x_value + direction * delta
                label_offsets[adjusted] = 0
                return adjusted - x_value
            label_offsets[x_value] = 0
            return 0

        for method, ad_value in ad_values.items():
            method_curve = next((curve for curve in patched_curves.values() if curve.method == method), None)
            if method_curve is None:
                continue

            method_crossing_epoch = self._find_threshold_crossing(method_curve, tau)
            if np.isnan(method_crossing_epoch):
                # Annotate absence of threshold crossing for clarity
                x_pos = method_curve.epochs[-1]
                no_cross_offset = 0.04
                y_pos = min(tau + no_cross_offset, 0.95)
                ax.text(
                    x_pos,
                    y_pos,
                    f"{method}: no τ crossing",
                    color=self.style.get_method_color(method),
                    fontsize=self.style.font_sizes['annotation'],
                    ha='right',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=self.style.get_method_color(method), alpha=0.5),
                )
                continue

            if np.isnan(ad_value):
                # We have a crossing but AD not computed; annotate crossing epoch only
                ax.plot(
                    method_crossing_epoch,
                    tau,
                    marker='o',
                    color=self.style.get_method_color(method),
                    markersize=4,
                    alpha=0.9,
                )
                ax.text(
                    method_crossing_epoch,
                    min(tau + vertical_offset, 0.95),
                    f"{method} E(τ)={method_crossing_epoch:.1f}",
                    color=self.style.get_method_color(method),
                    fontsize=self.style.font_sizes['annotation'],
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=self.style.get_method_color(method), alpha=0.5),
                )
                continue

            color = self.style.get_method_color(method)

            ax.axvline(
                x=method_crossing_epoch,
                color=color,
                linestyle=':',
                alpha=0.6,
                linewidth=1
            )

            ax.plot(
                method_crossing_epoch,
                tau,
                marker='o',
                color=color,
                markersize=4,
                alpha=0.9,
            )

            offset = _get_label_position(method_crossing_epoch)
            label_x = method_crossing_epoch + offset
            label_y = min(tau + 0.04, 0.95)
            ax.text(
                label_x,
                label_y,
                f"{method}: ΔE={ad_value:+.1f} (E={method_crossing_epoch:.1f})",
                color=color,
                fontsize=self.style.font_sizes['annotation'],
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=color, alpha=0.55)
            )

    def _get_shading_values(self, curve: AccuracyCurve, mode: str) -> Optional[np.ndarray]:
        if mode == 'std' and curve.std_dev is not None:
            return curve.std_dev
        if mode == 'ci' and curve.conf_interval is not None:
            return curve.conf_interval
        if mode == 'half_ci' and curve.conf_interval is not None:
            return curve.conf_interval / 2.0
        # Fallbacks
        if curve.conf_interval is not None and len(curve.conf_interval) == len(curve.epochs):
            return curve.conf_interval
        return None

    def _find_threshold_crossing(self, curve: AccuracyCurve, threshold: float) -> float:
        """Return first epoch where mean accuracy crosses the threshold."""
        if curve is None or len(curve.mean_accuracy) == 0:
            return np.nan

        accuracies = np.asarray(curve.mean_accuracy)
        epochs = np.asarray(curve.epochs)

        indices = np.where(accuracies >= threshold)[0]
        if len(indices) == 0:
            return np.nan

        return float(epochs[indices[0]])

    def _create_panel_b(
        self,
        ax: Axes,
        pd_series: Dict[str, TimeSeries],
        *,
        panel_label: Optional[str] = 'B',
    ) -> None:
        """
        Create Panel B: Performance Deficit (PD_t) time series.

        Args:
            ax: Matplotlib axes object
            pd_series: Dictionary of TimeSeries objects for Performance Deficit
        """
        # Set panel label
        if panel_label:
            ax.text(
                0.02, 0.98, panel_label,
                transform=ax.transAxes,
                fontsize=self.style.font_sizes['panel_label'],
                fontweight='bold',
                verticalalignment='top'
            )

        if not pd_series:
            # Empty panel with message
            ax.text(
                0.5, 0.5, 'No Performance Deficit data available',
                transform=ax.transAxes,
                fontsize=self.style.font_sizes['axis_label'],
                ha='center', va='center',
                style='italic', alpha=0.7
            )
        else:
            # Plot PD_t curves
            for method, series in pd_series.items():
                color = self.style.get_method_color(method)

                ax.plot(
                    series.epochs, series.values,
                    color=color, linewidth=self.style.line_width,
                    label=method
                )

                # Add confidence interval if available
                if len(series.conf_interval) > 0 and np.any(series.conf_interval > 0):
                    ax.fill_between(
                        series.epochs,
                        series.values - series.conf_interval,
                        series.values + series.conf_interval,
                        color=color, alpha=self.style.confidence_alpha
                    )

            # Add zero reference line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

            # Legend
            ax.legend(fontsize=self.style.font_sizes['legend'], loc='best')

        # Formatting
        ax.set_xlabel('Effective Epoch', fontsize=self.style.font_sizes['axis_label'])
        ax.set_ylabel('Performance Deficit (PD_t)', fontsize=self.style.font_sizes['axis_label'])
        ax.set_title('Performance Deficit: A_S(e) - A_CL(e)', fontsize=self.style.font_sizes['title'])
        ax.grid(True, alpha=self.style.grid_alpha)

    def _create_panel_c(
        self,
        ax: Axes,
        sfr_series: Dict[str, TimeSeries],
        *,
        panel_label: Optional[str] = 'C',
    ) -> None:
        """
        Create Panel C: Shortcut Forgetting Rate (SFR_rel) time series.

        Args:
            ax: Matplotlib axes object
            sfr_series: Dictionary of TimeSeries objects for Shortcut Forgetting Rate
        """
        # Set panel label
        if panel_label:
            ax.text(
                0.02, 0.98, panel_label,
                transform=ax.transAxes,
                fontsize=self.style.font_sizes['panel_label'],
                fontweight='bold',
                verticalalignment='top'
            )

        if not sfr_series:
            # Empty panel with message
            ax.text(
                0.5, 0.5, 'No Shortcut Forgetting Rate data available',
                transform=ax.transAxes,
                fontsize=self.style.font_sizes['axis_label'],
                ha='center', va='center',
                style='italic', alpha=0.7
            )
        else:
            # Plot SFR_rel curves
            for method, series in sfr_series.items():
                color = self.style.get_method_color(method)

                ax.plot(
                    series.epochs, series.values,
                    color=color, linewidth=self.style.line_width,
                    label=method
                )

                # Add confidence interval if available
                if len(series.conf_interval) > 0 and np.any(series.conf_interval > 0):
                    ax.fill_between(
                        series.epochs,
                        series.values - series.conf_interval,
                        series.values + series.conf_interval,
                        color=color, alpha=self.style.confidence_alpha
                    )

            # Add zero reference line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

            # Legend
            ax.legend(fontsize=self.style.font_sizes['legend'], loc='best')

        # Formatting
        ax.set_xlabel('Effective Epoch', fontsize=self.style.font_sizes['axis_label'])
        ax.set_ylabel('Shortcut Forgetting Rate (SFR_rel)', fontsize=self.style.font_sizes['axis_label'])
        ax.set_title('Relative Shortcut Forgetting: Δ_CL(e) - Δ_S(e)', fontsize=self.style.font_sizes['title'])
        ax.grid(True, alpha=self.style.grid_alpha)

    def save_figure(self, fig: Figure, filepath: str, **kwargs) -> None:
        """
        Save figure with configured settings and file size optimization.

        Args:
            fig: Matplotlib figure to save
            filepath: Output file path
            **kwargs: Additional keyword arguments (override defaults)

        Raises:
            ValueError: If file size exceeds 5MB limit
        """
        # Use style's save method
        self.style.save_figure(fig, filepath, **kwargs)

        # Check file size
        file_path = Path(filepath)
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 5.0:
                self.logger.warning(
                    f"Generated file size ({file_size_mb:.1f} MB) exceeds 5MB limit. "
                    f"Consider reducing DPI or figure complexity."
                )
            else:
                self.logger.info(f"Figure saved successfully: {filepath} ({file_size_mb:.1f} MB)")

    def create_eri_bar_chart(
        self,
        *,
        baseline_method: str,
        ad_values: Dict[str, float],
        pd_series: Dict[str, TimeSeries],
        sfr_series: Dict[str, TimeSeries],
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        title: Optional[str] = None,
    ) -> Figure:
        """Create a bar chart of composite ERI scores per method."""

        self.style.apply_style()

        pd_final: Dict[str, float] = {}
        for method, series in pd_series.items():
            if len(series.values) > 0:
                pd_final[method] = float(series.values[-1])

        sfr_final: Dict[str, float] = {}
        for method, series in sfr_series.items():
            if len(series.values) > 0:
                sfr_final[method] = float(series.values[-1])

        baseline_lower = baseline_method.lower()

        methods = sorted({*ad_values.keys(), *pd_final.keys(), *sfr_final.keys()})
        methods = [m for m in methods if m.lower() != baseline_lower]

        preferred_order = ['derpp', 'ewc_on', 'gpm', 'dgr', 'sgd']
        order_lookup = {name.lower(): idx for idx, name in enumerate(preferred_order)}
        methods.sort(key=lambda m: (order_lookup.get(m.lower(), len(order_lookup)), m))

        footnote_symbols = {'AD_norm': '†', 'PD': '‡', 'SFR_rel': '§'}
        footnote_notes = {key: [] for key in footnote_symbols}

        def _is_missing(value: Optional[float]) -> bool:
            if value is None:
                return True
            try:
                return bool(np.isnan(value))
            except TypeError:
                return False

        records: List[Dict[str, Union[str, Tuple[float, float, float], float, List[str]]]] = []
        for method in methods:
            metrics = {}
            missing_components: List[str] = []

            ad_raw = ad_values.get(method)
            if _is_missing(ad_raw):
                metrics['AD_norm'] = 0.0
                missing_components.append('AD_norm')
                self.logger.warning("Setting AD contribution to 0 for %s (missing data)", method)
            else:
                metrics['AD_norm'] = float(np.clip(float(ad_raw) / 50.0, -1.0, 1.0))

            pd_raw = pd_final.get(method)
            if _is_missing(pd_raw):
                metrics['PD'] = 0.0
                missing_components.append('PD')
                self.logger.warning("Setting PD contribution to 0 for %s (missing data)", method)
            else:
                metrics['PD'] = float(pd_raw)

            sfr_raw = sfr_final.get(method)
            if _is_missing(sfr_raw):
                metrics['SFR_rel'] = 0.0
                missing_components.append('SFR_rel')
                self.logger.warning("Setting SFR_rel contribution to 0 for %s (missing data)", method)
            else:
                metrics['SFR_rel'] = float(sfr_raw)

            if len(missing_components) == len(footnote_symbols):
                self.logger.warning("Skipping %s – all ERI components missing", method)
                continue

            contributions = (
                weights[0] * metrics['AD_norm'],
                weights[1] * metrics['PD'],
                weights[2] * metrics['SFR_rel'],
            )
            eri_score = sum(contributions)

            for component in missing_components:
                if component in footnote_notes:
                    footnote_notes[component].append(method)

            annotation_suffix = ''.join(footnote_symbols[comp] for comp in missing_components if comp in footnote_symbols)

            records.append({
                'method': method,
                'contributions': contributions,
                'eri_score': eri_score,
                'missing': missing_components,
                'annotation_suffix': annotation_suffix,
            })

        if not records:
            fig, ax = plt.subplots(1, 1, figsize=self.style.figure_size, dpi=self.style.dpi)
            ax.text(
                0.5,
                0.5,
                'Insufficient data to compute ERI scores',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=self.style.font_sizes['axis_label'],
                style='italic',
                alpha=0.7,
            )
            ax.axis('off')
            return fig

        methods_ordered = [record['method'] for record in records]
        eri_totals = [record['eri_score'] for record in records]
        contributions = np.array([record['contributions'] for record in records])

        fig, ax = plt.subplots(1, 1, figsize=self.style.figure_size, dpi=self.style.dpi)

        indices = np.arange(len(methods_ordered))
        width = 0.6

        component_labels = [r'$0.4\cdot AD_{norm}$', r'$0.4\cdot PD$', r'$0.2\cdot SFR_{rel}$']
        component_colors = ['#5b8def', '#d85f5f', '#6ac36a']

        pos_bottom = np.zeros(len(methods_ordered))
        neg_bottom = np.zeros(len(methods_ordered))
        drawn_components = set()

        for idx in range(contributions.shape[1]):
            values = contributions[:, idx]
            color = component_colors[idx]
            label = component_labels[idx]

            positive_mask = values > 0
            negative_mask = values < 0

            if np.any(positive_mask):
                label_to_use = label if label not in drawn_components else '_nolegend_'
                ax.bar(
                    indices[positive_mask],
                    values[positive_mask],
                    width=width,
                    bottom=pos_bottom[positive_mask],
                    color=color,
                    alpha=0.85,
                    label=label_to_use
                )
                pos_bottom[positive_mask] += values[positive_mask]
                if label_to_use != '_nolegend_':
                    drawn_components.add(label)

            if np.any(negative_mask):
                label_to_use = label if label not in drawn_components else '_nolegend_'
                ax.bar(
                    indices[negative_mask],
                    values[negative_mask],
                    width=width,
                    bottom=neg_bottom[negative_mask],
                    color=color,
                    alpha=0.85,
                    label=label_to_use
                )
                neg_bottom[negative_mask] += values[negative_mask]
                if label_to_use != '_nolegend_':
                    drawn_components.add(label)

        ax.axhline(0, color='black', linewidth=1, alpha=0.6)
        ax.set_xticks(indices)
        ax.set_xticklabels(methods_ordered, rotation=15, ha='right')
        ax.set_xlim(-0.5, len(methods_ordered) - 0.5)
        ax.set_ylabel('ERI Score', fontsize=self.style.font_sizes['axis_label'])
        ax.legend(fontsize=self.style.font_sizes['legend'], loc='best')

        for idx, value in enumerate(eri_totals):
            suffix = records[idx]['annotation_suffix']
            label = f'{value:.3f}{suffix}' if suffix else f'{value:.3f}'
            ax.text(
                indices[idx],
                value + (0.01 if value >= 0 else -0.01),
                label,
                ha='center',
                va='bottom' if value >= 0 else 'top',
                fontsize=self.style.font_sizes['annotation']
            )

        if title:
            ax.set_title(title, fontsize=self.style.font_sizes['title'])
        else:
            ax.set_title('Composite Einstellung Rigidity Index', fontsize=self.style.font_sizes['title'])

        footnote_texts = []
        footnote_descriptions = {
            'AD_norm': 'AD contribution unavailable (no τ crossing or missing curve)',
            'PD': 'PD contribution unavailable (insufficient baseline overlap)',
            'SFR_rel': 'SFR_rel contribution unavailable (missing patched/masked alignment)',
        }

        if any(footnote_notes.values()):
            for component, methods_missing in footnote_notes.items():
                if not methods_missing:
                    continue
                symbol = footnote_symbols[component]
                sorted_methods = sorted(
                    {m: methods_ordered.index(m) for m in methods_missing if m in methods_ordered}.items(),
                    key=lambda item: item[1]
                )
                method_labels = ', '.join(item[0] for item in sorted_methods) or ', '.join(methods_missing)
                footnote_texts.append(
                    f"{symbol} {footnote_descriptions[component]}: {method_labels}"
                )

        formula = r'$ERI = 0.4\cdot AD_{norm} + 0.4\cdot PD + 0.2\cdot SFR_{rel}$'
        formula_y = -0.18 if footnote_texts else -0.12
        ax.text(
            0.5,
            formula_y,
            formula,
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=self.style.font_sizes['axis_label'],
        )

        if footnote_texts:
            ax.text(
                0.5,
                -0.28,
                '\n'.join(footnote_texts),
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=max(self.style.font_sizes['annotation'] - 1, 8),
                alpha=0.8,
            )

        bottom_margin = 0.28 if footnote_texts else 0.16
        rect_height = 0.8 if footnote_texts else 0.86
        self._finalize_layout(fig, top=0.92, bottom=bottom_margin, rect=(0, 0, 1, rect_height))
        return fig

    def create_quick_dynamics_plot(
        self,
        patched_curves: Dict[str, AccuracyCurve],
        tau: float = 0.6,
        methods: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create a simplified dynamics plot with just accuracy trajectories.

        This is a convenience method for quick visualization when full
        PD_t and SFR_rel data is not available.

        Args:
            patched_curves: Dictionary of AccuracyCurve objects for shortcut_normal split
            tau: Threshold value for visualization
            methods: Optional list of methods to include (all if None)
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure object
        """
        # Filter methods if specified
        if methods:
            filtered_curves = {
                key: curve for key, curve in patched_curves.items()
                if curve.method in methods
            }
        else:
            filtered_curves = patched_curves

        # Apply style configuration
        self.style.apply_style()

        # Create single panel figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=self.style.dpi)

        # Plot curves
        for key, curve in filtered_curves.items():
            method = curve.method
            color = self.style.get_method_color(method)

            # Plot mean accuracy with confidence interval
            ax.plot(
                curve.epochs, curve.mean_accuracy,
                color=color, linewidth=self.style.line_width,
                label=method
            )

            # Add confidence interval
            if len(curve.conf_interval) > 0 and np.any(curve.conf_interval > 0):
                ax.fill_between(
                    curve.epochs,
                    curve.mean_accuracy - curve.conf_interval,
                    curve.mean_accuracy + curve.conf_interval,
                    color=color, alpha=self.style.confidence_alpha
                )

        # Add threshold line
        ax.axhline(y=tau, color='gray', linestyle=':', alpha=0.7, label=f'τ = {tau}')

        # Formatting
        ax.set_xlabel('Effective Epoch', fontsize=self.style.font_sizes['axis_label'])
        ax.set_ylabel('Accuracy', fontsize=self.style.font_sizes['axis_label'])
        ax.set_title('Accuracy Trajectories on Shortcut Task', fontsize=self.style.font_sizes['title'])
        ax.grid(True, alpha=self.style.grid_alpha)
        ax.legend(fontsize=self.style.font_sizes['legend'], loc='best')
        ax.set_ylim(0, 1)

        self._finalize_layout(fig)

        # Save figure if path provided
        if save_path:
            self.save_figure(fig, save_path)

        return fig

    def get_figure_info(self, fig: Figure) -> Dict[str, any]:
        """
        Get information about a generated figure.

        Args:
            fig: Matplotlib figure object

        Returns:
            Dictionary containing figure information
        """
        return {
            'figure_size_inches': fig.get_size_inches(),
            'dpi': fig.dpi,
            'n_axes': len(fig.axes),
            'n_artists': sum(len(ax.get_children()) for ax in fig.axes),
            'style_config': {
                'color_palette_size': len(self.style.color_palette),
                'confidence_alpha': self.style.confidence_alpha,
                'line_width': self.style.line_width,
            }
        }
