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

    def create_dynamics_figure(
        self,
        patched_curves: Dict[str, AccuracyCurve],
        masked_curves: Dict[str, AccuracyCurve],
        pd_series: Dict[str, TimeSeries],
        sfr_series: Dict[str, TimeSeries],
        ad_values: Dict[str, float],
        tau: float,
        title: Optional[str] = None,
        save_path: Optional[str] = None
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
        self._create_panel_a(ax_a, patched_curves, masked_curves, ad_values, tau)

        # Panel B: Performance Deficit (PD_t) time series
        self._create_panel_b(ax_b, pd_series)

        # Panel C: Shortcut Forgetting Rate (SFR_rel) time series
        self._create_panel_c(ax_c, sfr_series)

        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=self.style.font_sizes['title'], y=0.98)

        # Adjust layout
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.95)

        # Save figure if path provided
        if save_path:
            self.save_figure(fig, save_path)

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

        if not masked_curves:
            raise ValueError("masked_curves cannot be empty")

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
        tau: float
    ) -> None:
        """
        Create Panel A: Accuracy trajectories with confidence intervals and AD markers.

        Args:
            ax: Matplotlib axes object
            patched_curves: AccuracyCurve objects for shortcut_normal split
            masked_curves: AccuracyCurve objects for shortcut_masked split
            ad_values: Adaptation Delay values per method
            tau: Threshold value for AD computation
        """
        # Set panel label
        ax.text(
            0.02, 0.98, 'A',
            transform=ax.transAxes,
            fontsize=self.style.font_sizes['panel_label'],
            fontweight='bold',
            verticalalignment='top'
        )

        # Plot patched curves (solid lines)
        methods_plotted = set()
        for key, curve in patched_curves.items():
            method = curve.method
            color = self.style.get_method_color(method)

            # Plot mean accuracy with confidence interval
            ax.plot(
                curve.epochs, curve.mean_accuracy,
                color=color, linewidth=self.style.line_width,
                label=f"{method} (patched)", linestyle='-'
            )

            # Add confidence interval
            if len(curve.conf_interval) > 0 and np.any(curve.conf_interval > 0):
                ax.fill_between(
                    curve.epochs,
                    curve.mean_accuracy - curve.conf_interval,
                    curve.mean_accuracy + curve.conf_interval,
                    color=color, alpha=self.style.confidence_alpha
                )

            methods_plotted.add(method)

        # Plot masked curves (dashed lines)
        for key, curve in masked_curves.items():
            method = curve.method
            if method not in methods_plotted:
                continue  # Only plot if we have corresponding patched curve

            color = self.style.get_method_color(method)

            # Plot mean accuracy with confidence interval
            ax.plot(
                curve.epochs, curve.mean_accuracy,
                color=color, linewidth=self.style.line_width,
                label=f"{method} (masked)", linestyle='--'
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

        # Add AD markers and annotations
        self._add_ad_markers(ax, patched_curves, ad_values, tau)

        # Formatting
        ax.set_xlabel('Effective Epoch', fontsize=self.style.font_sizes['axis_label'])
        ax.set_ylabel('Accuracy', fontsize=self.style.font_sizes['axis_label'])
        ax.set_title('Accuracy Trajectories on Shortcut Task', fontsize=self.style.font_sizes['title'])
        ax.grid(True, alpha=self.style.grid_alpha)
        ax.legend(fontsize=self.style.font_sizes['legend'], loc='best')

        # Set y-axis limits to show full range
        ax.set_ylim(0, 1)

    def _add_ad_markers(
        self,
        ax: Axes,
        patched_curves: Dict[str, AccuracyCurve],
        ad_values: Dict[str, float],
        tau: float
    ) -> None:
        """
        Add AD markers and annotations to Panel A.

        Args:
            ax: Matplotlib axes object
            patched_curves: AccuracyCurve objects for shortcut_normal split
            ad_values: Adaptation Delay values per method
            tau: Threshold value
        """
        # Find Scratch_T2 crossing epoch for reference
        scratch_crossing_epoch = None
        for key, curve in patched_curves.items():
            if curve.method == 'Scratch_T2':
                crossing_indices = np.where(curve.mean_accuracy >= tau)[0]
                if len(crossing_indices) > 0:
                    scratch_crossing_epoch = curve.epochs[crossing_indices[0]]
                break

        # Add vertical lines for threshold crossings and AD annotations
        annotation_y_positions = np.linspace(0.1, 0.3, len(ad_values))

        for i, (method, ad_value) in enumerate(ad_values.items()):
            if np.isnan(ad_value):
                continue  # Skip censored runs

            color = self.style.get_method_color(method)

            # Find method crossing epoch
            method_curve = None
            for curve in patched_curves.values():
                if curve.method == method:
                    method_curve = curve
                    break

            if method_curve is None:
                continue

            crossing_indices = np.where(method_curve.mean_accuracy >= tau)[0]
            if len(crossing_indices) > 0:
                method_crossing_epoch = method_curve.epochs[crossing_indices[0]]

                # Add vertical dashed line at crossing
                ax.axvline(
                    x=method_crossing_epoch,
                    color=color,
                    linestyle='--',
                    alpha=0.7,
                    linewidth=1
                )

                # Add AD annotation
                if scratch_crossing_epoch is not None:
                    # Position annotation
                    y_pos = annotation_y_positions[i % len(annotation_y_positions)]

                    ax.annotate(
                        f'AD = {ad_value:.1f}',
                        xy=(method_crossing_epoch, tau),
                        xytext=(method_crossing_epoch + 1, y_pos),
                        fontsize=self.style.font_sizes['annotation'],
                        color=color,
                        arrowprops=dict(
                            arrowstyle='->',
                            color=color,
                            alpha=0.7,
                            lw=1
                        )
                    )

        # Add Scratch_T2 reference line if found
        if scratch_crossing_epoch is not None:
            ax.axvline(
                x=scratch_crossing_epoch,
                color=self.style.get_method_color('Scratch_T2'),
                linestyle='-',
                alpha=0.7,
                linewidth=1,
                label='E_S(τ)'
            )

    def _create_panel_b(self, ax: Axes, pd_series: Dict[str, TimeSeries]) -> None:
        """
        Create Panel B: Performance Deficit (PD_t) time series.

        Args:
            ax: Matplotlib axes object
            pd_series: Dictionary of TimeSeries objects for Performance Deficit
        """
        # Set panel label
        ax.text(
            0.02, 0.98, 'B',
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

    def _create_panel_c(self, ax: Axes, sfr_series: Dict[str, TimeSeries]) -> None:
        """
        Create Panel C: Shortcut Forgetting Rate (SFR_rel) time series.

        Args:
            ax: Matplotlib axes object
            sfr_series: Dictionary of TimeSeries objects for Shortcut Forgetting Rate
        """
        # Set panel label
        ax.text(
            0.02, 0.98, 'C',
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

        plt.tight_layout()

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
