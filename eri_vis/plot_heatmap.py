"""
ERIHeatmapPlotter - Robustness Analysis Visualization for ERI metrics.

This module provides the ERIHeatmapPlotter class for creating AD(τ) sensitivity
heatmaps that demonstrate the robustness of the Adaptation Delay metric across
different threshold values.

Integrates with the existing Mammoth Einstellung experiment infrastructure:
- Uses AccuracyCurve objects from ERITimelineProcessor
- Compatible with all existing Mammoth continual learning strategies
- Handles censored runs (no threshold crossing) appropriately
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

from .processing import AccuracyCurve, ERITimelineProcessor
from .styles import PlotStyleConfig, DEFAULT_STYLE


@dataclass
class TauSensitivityResult:
    """
    Results from tau sensitivity analysis for AD(τ) computation.

    Attributes:
        methods: List of method names (excluding baseline)
        taus: Array of tau threshold values
        ad_matrix: Matrix of AD values, shape (len(methods), len(taus))
                  NaN values indicate censored runs (no threshold crossing)
        baseline_method: Name of baseline method used for comparison
        n_censored: Number of censored entries per method
    """
    methods: List[str]
    taus: np.ndarray
    ad_matrix: np.ndarray  # shape: (n_methods, n_taus)
    baseline_method: str = "Scratch_T2"
    n_censored: Optional[Dict[str, int]] = None


class ERIHeatmapPlotter:
    """
    Plotter for creating AD(τ) sensitivity heatmaps.

    This class creates publication-ready heatmaps showing how Adaptation Delay
    varies across different threshold values (τ) for differtinual learning
    methods compared to a baseline (typically Scratch_T2).
    """

    def __init__(self, style: Optional[PlotStyleConfig] = None):
        """
        Initialize the heatmap plotter.

        Args:
            style: PlotStyleConfig instance (uses DEFAULT_STYLE if None)
        """
        self.style = style or DEFAULT_STYLE
        self.logger = logging.getLogger(__name__)

    def compute_tau_sensitivity(
        self,
        curves: Dict[str, AccuracyCurve],
        taus: List[float],
        baseline_method: str = "Scratch_T2",
        *,
        smoothing_window: int = 3,
        use_smoothing: bool = True,
    ) -> TauSensitivityResult:
        """
        Compute AD(τ) sensitivity matrix across different tau values.

        Args:
            curves: Dictionary of AccuracyCurve objects
            taus: List of tau threshold values to analyze
            baseline_method: Name of baseline method for comparison

        Returns:
            TauSensitivityResult containing the sensitivity matrix

        Raises:
            ValueError: If baseline method not found or no valid curves
        """
        if not curves:
            raise ValueError("No curves provided for sensitivity analysis")

        if not taus:
            raise ValueError("No tau values provided")

        taus = np.array(sorted(taus))

        # Find baseline curve for shortcut_normal split
        baseline_curve = None
        for key, curve in curves.items():
            if (curve.method == baseline_method and
                'shortcut_normal' in key):
                baseline_curve = curve
                break

        if baseline_curve is None:
            raise ValueError(f"Baseline method '{baseline_method}' not found with shortcut_normal split")

        # Get all continual learning methods (excluding baseline)
        cl_methods = []
        cl_curves = {}

        for key, curve in curves.items():
            if (curve.method != baseline_method and
                'shortcut_normal' in key):
                if curve.method not in cl_methods:
                    cl_methods.append(curve.method)
                    cl_curves[curve.method] = curve

        if not cl_methods:
            raise ValueError("No continual learning methods found")

        cl_methods = sorted(cl_methods)

        # Initialize AD matrix
        ad_matrix = np.full((len(cl_methods), len(taus)), np.nan)
        n_censored = {method: 0 for method in cl_methods}

        # Compute AD for each tau using the timeline processor to ensure
        # consistency with bar chart calculations (seedwise relative to baseline).
        for j, tau in enumerate(taus):
            tau_processor = ERITimelineProcessor(
                smoothing_window=smoothing_window,
                tau=float(tau),
                baseline_method=baseline_method,
                use_smoothing=use_smoothing,
            )

            try:
                ad_results = tau_processor.compute_adaptation_delays(curves)
                seedwise_info = tau_processor.get_seedwise_ad()
            except Exception as exc:
                self.logger.warning(
                    "Failed to compute AD values for τ=%.3f: %s",
                    tau,
                    exc,
                )
                continue

            for i, method in enumerate(cl_methods):
                ad_value = ad_results.get(method, np.nan)
                ad_matrix[i, j] = ad_value

                info = seedwise_info.get(method)
                if info is None:
                    continue

                censored = info.get('censored')
                if isinstance(censored, np.ndarray):
                    n_censored[method] += int(censored.size)
                elif isinstance(censored, list):
                    n_censored[method] += len(censored)

        return TauSensitivityResult(
            methods=cl_methods,
            taus=taus,
            ad_matrix=ad_matrix,
            baseline_method=baseline_method,
            n_censored=n_censored
        )

    def _find_threshold_crossing(self, curve: AccuracyCurve, threshold: float) -> float:
        """
        Find the first epoch where the curve crosses the threshold.

        Args:
            curve: AccuracyCurve object
            threshold: Threshold value to find crossing for

        Returns:
            Epoch of first crossing, or NaN if no crossing found
        """
        if len(curve.mean_accuracy) == 0:
            return np.nan

        # Find first index where accuracy >= threshold
        crossing_indices = np.where(curve.mean_accuracy >= threshold)[0]

        if len(crossing_indices) == 0:
            return np.nan  # No crossing found (censored)

        crossing_idx = crossing_indices[0]
        return curve.epochs[crossing_idx]

    @staticmethod
    def _find_threshold_crossing_series(
        epochs: np.ndarray,
        values: np.ndarray,
        threshold: float,
    ) -> float:
        """Find threshold crossing for a single seed trajectory."""
        if values is None or epochs is None:
            return np.nan

        if len(values) == 0 or len(epochs) == 0:
            return np.nan

        if len(values) != len(epochs):
            return np.nan

        indices = np.where(values >= threshold)[0]
        if len(indices) == 0:
            return np.nan

        return float(epochs[indices[0]])

    def create_tau_sensitivity_heatmap(
        self,
        sensitivity_result: TauSensitivityResult,
        title: str = "Adaptation Delay Sensitivity Analysis",
        figsize: Optional[Tuple[float, float]] = None
    ) -> plt.Figure:
        """
        Create AD(τ) sensitivity heatmap.

        Args:
            sensitivity_result: TauSensitivityResult from compute_tau_sensitivity
            title: Figure title
            figsize: Figure size (uses style default if None)

        Returns:
            Matplotlib Figure object

        Raises:
            ValueError: If sensitivity result is invalid
        """
        if sensitivity_result.ad_matrix.size == 0:
            raise ValueError("Empty sensitivity matrix")

        # Set up figure
        if figsize is None:
            figsize = self.style.figure_size

        fig, ax = plt.subplots(figsize=figsize, dpi=self.style.dpi)

        # Get data
        methods = sensitivity_result.methods
        taus = sensitivity_result.taus
        ad_matrix = sensitivity_result.ad_matrix

        # Create diverging colormap centered at 0
        # Determine color range based on data (excluding NaN)
        valid_data = ad_matrix[~np.isnan(ad_matrix)]
        if len(valid_data) == 0:
            self.logger.warning("All AD values are NaN - creating empty heatmap")
            vmin, vmax = -1, 1
        else:
            abs_max = np.max(np.abs(valid_data))
            vmin, vmax = -abs_max, abs_max

        # Use diverging colormap centered at 0
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = plt.cm.RdBu_r  # Red for positive (worse), Blue for negative (better)

        # Create heatmap
        im = ax.imshow(
            ad_matrix,
            cmap=cmap,
            norm=norm,
            aspect='auto',
            interpolation='nearest'
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(
            f'AD(τ) vs {sensitivity_result.baseline_method} (epochs)',
            fontsize=self.style.font_sizes['axis_label']
        )
        cbar.ax.tick_params(labelsize=self.style.font_sizes['tick_label'])

        # Set ticks and labels
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f'{tau:.2f}' for tau in taus])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)

        # Set labels and title
        ax.set_xlabel('Threshold τ', fontsize=self.style.font_sizes['axis_label'])
        ax.set_ylabel('Method', fontsize=self.style.font_sizes['axis_label'])
        ax.set_title(title, fontsize=self.style.font_sizes['title'], pad=20)

        # Add cell annotations
        self._add_cell_annotations(ax, ad_matrix, methods, taus)

        # Add hatching for NaN values
        self._add_nan_hatching(ax, ad_matrix)

        # Add legend for NaN handling
        if np.any(np.isnan(ad_matrix)):
            self._add_nan_legend(ax, sensitivity_result)

        # Style the plot
        ax.tick_params(labelsize=self.style.font_sizes['tick_label'])

        # Remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add grid
        ax.set_xticks(np.arange(len(taus)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(methods)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

        plt.tight_layout()

        return fig

    def _add_cell_annotations(
        self,
        ax: plt.Axes,
        ad_matrix: np.ndarray,
        methods: List[str],
        taus: np.ndarray
    ) -> None:
        """Add text annotations to heatmap cells."""
        for i in range(len(methods)):
            for j in range(len(taus)):
                value = ad_matrix[i, j]
                if not np.isnan(value):
                    # Choose text color based on background
                    text_color = 'white' if abs(value) > 0.5 * np.nanmax(np.abs(ad_matrix)) else 'black'

                    ax.text(
                        j, i, f'{value:.1f}',
                        ha='center', va='center',
                        color=text_color,
                        fontsize=self.style.font_sizes['annotation'],
                        fontweight='bold'
                    )

    def _add_nan_hatching(self, ax: plt.Axes, ad_matrix: np.ndarray) -> None:
        """Add hatching pattern to NaN cells."""
        for i in range(ad_matrix.shape[0]):
            for j in range(ad_matrix.shape[1]):
                if np.isnan(ad_matrix[i, j]):
                    # Add hatching pattern
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        linewidth=0,
                        edgecolor='none',
                        facecolor='lightgray',
                        alpha=0.3,
                        hatch='///'
                    )
                    ax.add_patch(rect)

    def _add_nan_legend(
        self,
        ax: plt.Axes,
        sensitivity_result: TauSensitivityResult
    ) -> None:
        """Add legend explaining NaN handling."""
        # Count total censored entries
        if sensitivity_result.n_censored is not None:
            total_censored = sum(sensitivity_result.n_censored.values())
        else:
            # Count NaN entries directly from matrix
            total_censored = np.sum(np.isnan(sensitivity_result.ad_matrix))

        total_entries = len(sensitivity_result.methods) * len(sensitivity_result.taus)

        legend_text = f"Hatched cells: No threshold crossing ({total_censored}/{total_entries} entries)"

        # Add text box in upper right corner
        ax.text(
            0.98, 0.98, legend_text,
            transform=ax.transAxes,
            fontsize=self.style.font_sizes['annotation'],
            ha='right', va='top',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='gray',
                alpha=0.8
            )
        )

    def save_heatmap(
        self,
        fig: plt.Figure,
        filepath: str,
        **kwargs
    ) -> None:
        """
        Save heatmap figure with configured settings.

        Args:
            fig: Matplotlib figure to save
            filepath: Output file path
            **kwargs: Additional keyword arguments for saving
        """
        self.style.save_figure(fig, filepath, **kwargs)

    def create_method_comparison_heatmap(
        self,
        curves: Dict[str, AccuracyCurve],
        tau_range: Tuple[float, float] = (0.5, 0.8),
        tau_step: float = 0.05,
        baseline_method: str = "Scratch_T2",
        title: str = "AD(τ) Robustness Analysis"
    ) -> plt.Figure:
        """
        Create a complete AD(τ) sensitivity heatmap from curves.

        This is a convenience method that combines sensitivity computation
        and heatmap creation in one call.

        Args:
            curves: Dictionary of AccuracyCurve objects
            tau_range: Tuple of (min_tau, max_tau) for analysis
            tau_step: Step size for tau values
            baseline_method: Name of baseline method
            title: Figure title

        Returns:
            Matplotlib Figure object

        Raises:
            ValueError: If invalid parameters or no valid data
        """
        # Generate tau values
        taus = np.arange(tau_range[0], tau_range[1] + tau_step/2, tau_step)

        # Compute sensitivity
        sensitivity_result = self.compute_tau_sensitivity(
            curves, taus.tolist(), baseline_method
        )

        # Create heatmap
        return self.create_tau_sensitivity_heatmap(sensitivity_result, title)
