#!/usr/bin/env python3
"""
Fix Comparative ERI Visualization

This script addresses the incompatibility between the current eri_vis system
and the comparative experiment results. The main issues are:

1. eri_vis expects individual method datasets, but comparative results have all methods in one CSV
2. Processing logic assumes one method per dataset
3. AD and SFR calculations are broken due to data structure assumptions
4. Duplicate entries per epoch need to be handled
5. Need to use top1 accuracy for tau threshold (τ = 0.6)

This script creates a working visualization system for comparative results.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComparativeERIProcessor:
    """
    Processor for comparative ERI data that handles multiple methods in one dataset.
    """

    def __init__(self, tau: float = 0.6, smoothing_window: int = 3):
        self.tau = tau
        self.smoothing_window = smoothing_window

    def load_session_data(self, session_dir: str) -> pd.DataFrame:
        """Load and combine all timeline data from session directory."""
        session_path = Path(session_dir)

        # Find all timeline CSV files
        timeline_files = list(session_path.glob("timeline_*.csv"))

        if not timeline_files:
            raise ValueError(f"No timeline files found in {session_dir}")

        # Load and combine all timeline data
        all_data = []
        for file_path in timeline_files:
            df = pd.read_csv(file_path)
            all_data.append(df)
            logger.info(f"Loaded {len(df)} rows from {file_path.name}")

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Handle duplicate entries per epoch by taking the mean
        # This addresses the issue where there are multiple evaluations per epoch
        combined_df = combined_df.groupby(['method', 'seed', 'epoch_eff', 'split']).agg({
            'acc': 'mean',
            'top5': 'mean',
            'loss': 'mean'
        }).reset_index()

        logger.info(f"Combined data: {len(combined_df)} rows with methods: {sorted(combined_df['method'].unique())}")
        logger.info(f"Splits: {sorted(combined_df['split'].unique())}")
        logger.info(f"Seeds: {sorted(combined_df['seed'].unique())}")
        logger.info(f"Epoch range: {combined_df['epoch_eff'].min():.1f} - {combined_df['epoch_eff'].max():.1f}")

        return combined_df

    def smooth_curve(self, values: np.ndarray) -> np.ndarray:
        """Apply smoothing to a curve."""
        if self.smoothing_window <= 1 or len(values) <= 1:
            return values.copy()

        # Simple moving average with edge handling
        smoothed = np.zeros_like(values)
        half_window = self.smoothing_window // 2

        for i in range(len(values)):
            start = max(0, i - half_window)
            end = min(len(values), i + half_window + 1)
            smoothed[i] = np.mean(values[start:end])

        return smoothed

    def compute_method_curves(self, df: pd.DataFrame) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Compute accuracy curves for each method and split.

        Returns:
            Dict[method][split] = (epochs, mean_acc, conf_interval)
        """
        curves      for j, epoch in enumerate(epochs):

        for method in df['method'].unique():
            curves[method] = {}
            method_data = df[df['method'] == method]

            for split in method_data['split'].unique():
                split_data = method_data[method_data['split'] == split]

                # Get unique epochs and seeds
                epochs = np.sort(split_data['epoch_eff'].unique())
                seeds = np.sort(split_data['seed'].unique())

                if len(epochs) == 0:
                    continue

                # Create accuracy matrix: seeds x epochs
                acc_matrix = np.full((len(seeds), len(epochs)), np.nan)

                for i, seed in enumerate(seeds):
                    seed_data = split_data[split_data['seed'] == seed]
                    for j, epoch in enumerate(epochs):
                        epoch_data = seed_data[seed_data['epoch_eff'] == epoch]
                        if len(epoch_data) > 0:
                            # Use top1 accuracy (acc column)
                            acc_matrix[i, j] = epoch_data['acc'].mean()

                # Handle missing data with forward fill
                for i in range(acc_matrix.shape[0]):
                    row = acc_matrix[i, :]
                    mask = ~np.isnan(row)
                    if np.any(mask):
                        # Forward fill
                        last_valid = None
                        for j in range(len(row)):
                            if mask[j]:
                                last_valid = row[j]
                            elif last_valid is not None:
                                row[j] = last_valid
                        acc_matrix[i, :] = row

                # Remove seeds that are still all NaN
                valid_seeds_mask = ~np.all(np.isnan(acc_matrix), axis=1)
                if not np.any(valid_seeds_mask):
                    logger.warning(f"No valid data for {method} {split}")
                    continue

                acc_matrix = acc_matrix[valid_seeds_mask, :]

                # Apply smoothing
                smoothed_matrix = np.zeros_like(acc_matrix)
                for i in range(acc_matrix.shape[0]):
                    smoothed_matrix[i, :] = self.smooth_curve(acc_matrix[i, :])

                # Compute mean and confidence intervals
                mean_acc = np.mean(smoothed_matrix, axis=0)
                if smoothed_matrix.shape[0] > 1:
                    std_acc = np.std(smoothed_matrix, axis=0, ddof=1)
                    n_seeds = smoothed_matrix.shape[0]
                    # 95% confidence interval using t-distribution approximation
                    t_critical = 1.96  # Approximate for large samples
                    if n_seeds < 30:
                        from scipy import stats
                        t_critical = stats.t.ppf(0.975, df=n_seeds-1)
                    conf_interval = t_critical * std_acc / np.sqrt(n_seeds)
                else:
                    conf_interval = np.zeros_like(mean_acc)

                curves[method][split] = (epochs, mean_acc, conf_interval)
                logger.info(f"Computed curve for {method} {split}: {len(epochs)} epochs, {np.sum(valid_seeds_mask)} seeds")

        return curves

    def compute_adaptation_delays(self, curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> Dict[str, float]:
        """Compute adaptation delays relative to Scratch_T2."""
        ad_values = {}

        # Find Scratch_T2 crossing epoch
        scratch_crossing = None
        if 'scratch_t2' in curves and 'T2_shortcut_normal' in curves['scratch_t2']:
            epochs, mean_acc, _ = curves['scratch_t2']['T2_shortcut_normal']
            crossing_indices = np.where(mean_acc >= self.tau)[0]
            if len(crossing_indices) > 0:
                scratch_crossing = epochs[crossing_indices[0]]
                logger.info(f"Scratch_T2 crosses tau={self.tau} at epoch {scratch_crossing}")

        if scratch_crossing is None:
            logger.warning("No Scratch_T2 crossing found for AD computation")
            return ad_values

        # Compute AD for each method
        for method in curves.keys():
            if method == 'scratch_t2':
                continue

            if 'T2_shortcut_normal' not in curves[method]:
                logger.warning(f"No T2_shortcut_normal data for {method}")
                continue

            epochs, mean_acc, _ = curves[method]['T2_shortcut_normal']
            crossing_indices = np.where(mean_acc >= self.tau)[0]

            if len(crossing_indices) > 0:
                method_crossing = epochs[crossing_indices[0]]
                ad_values[method] = method_crossing - scratch_crossing
                logger.info(f"{method} AD = {ad_values[method]:.2f} (crosses at epoch {method_crossing})")
            else:
                ad_values[method] = np.nan
                logger.warning(f"{method} never crosses tau={self.tau} (censored)")

        return ad_values

    def compute_performance_deficits(self, curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute Performance Deficit (PD_t) time series."""
        pd_series = {}

        # Get Scratch_T2 baseline
        if 'scratch_t2' not in curves or 'T2_shortcut_normal' not in curves['scratch_t2']:
            logger.warning("No Scratch_T2 baseline for PD_t computation")
            return pd_series

        scratch_epochs, scratch_acc, _ = curves['scratch_t2']['T2_shortcut_normal']

        # Compute PD_t for each method
        for method in curves.keys():
            if method == 'scratch_t2':
                continue

            if 'T2_shortcut_normal' not in curves[method]:
                continue

            method_epochs, method_acc, _ = curves[method]['T2_shortcut_normal']

            # Align epochs using interpolation
            common_epochs = np.intersect1d(scratch_epochs, method_epochs)
            if len(common_epochs) == 0:
                # Use interpolation to align
                min_epoch = max(scratch_epochs.min(), method_epochs.min())
                max_epoch = min(scratch_epochs.max(), method_epochs.max())
                if min_epoch < max_epoch:
                    common_epochs = np.linspace(min_epoch, max_epoch, 50)
                else:
                    continue

            # Interpolate both curves to common epochs
            scratch_interp = np.interp(common_epochs, scratch_epochs, scratch_acc)
            method_interp = np.interp(common_epochs, method_epochs, method_acc)

            # Compute PD_t = A_S - A_CL
            pd_values = scratch_interp - method_interp

            pd_series[method] = (common_epochs, pd_values)
            logger.info(f"Computed PD_t for {method}: {len(common_epochs)} points")

        return pd_series

    def compute_sfr_relative(self, curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute Shortcut Forgetting Rate relative (SFR_rel) time series."""
        sfr_series = {}

        # Get Scratch_T2 baseline curves
        if ('scratch_t2' not in curves or
            'T2_shortcut_normal' not in curves['scratch_t2'] or
            'T2_shortcut_masked' not in curves['scratch_t2']):
            logger.warning("Missing Scratch_T2 curves for SFR_rel computation")
            return sfr_series

        scratch_normal_epochs, scratch_normal_acc, _ = curves['scratch_t2']['T2_shortcut_normal']
        scratch_masked_epochs, scratch_masked_acc, _ = curves['scratch_t2']['T2_shortcut_masked']

        # Align scratch curves
        common_scratch_epochs = np.intersect1d(scratch_normal_epochs, scratch_masked_epochs)
        if len(common_scratch_epochs) == 0:
            logger.warning("Cannot align Scratch_T2 curves for SFR_rel")
            return sfr_series

        scratch_normal_interp = np.interp(common_scratch_epochs, scratch_normal_epochs, scratch_normal_acc)
        scratch_masked_interp = np.interp(common_scratch_epochs, scratch_masked_epochs, scratch_masked_acc)
        delta_scratch = scratch_normal_interp - scratch_masked_interp

        # Compute SFR_rel for each method
        for method in curves.keys():
            if method == 'scratch_t2':
                continue

            if ('T2_shortcut_normal' not in curves[method] or
                'T2_shortcut_masked' not in curves[method]):
                continue

            method_normal_epochs, method_normal_acc, _ = curves[method]['T2_shortcut_normal']
            method_masked_epochs, method_masked_acc, _ = curves[method]['T2_shortcut_masked']

            # Align method curves
            common_method_epochs = np.intersect1d(method_normal_epochs, method_masked_epochs)
            if len(common_method_epochs) == 0:
                continue

            method_normal_interp = np.interp(common_method_epochs, method_normal_epochs, method_normal_acc)
            method_masked_interp = np.interp(common_method_epochs, method_masked_epochs, method_masked_acc)
            delta_method = method_normal_interp - method_masked_interp

            # Align with scratch delta
            final_epochs = np.intersect1d(common_scratch_epochs, common_method_epochs)
            if len(final_epochs) == 0:
                continue

            delta_scratch_final = np.interp(final_epochs, common_scratch_epochs, delta_scratch)
            delta_method_final = np.interp(final_epochs, common_method_epochs, delta_method)

            # SFR_rel = Δ_CL - Δ_S
            sfr_rel_values = delta_method_final - delta_scratch_final

            sfr_series[method] = (final_epochs, sfr_rel_values)
            logger.info(f"Computed SFR_rel for {method}: {len(final_epochs)} points")

        return sfr_series


class ComparativeERIPlotter:
    """
    Plotter for comparative ERI visualizations.
    """

    def __init__(self, figsize=(15, 12), dpi=300):
        self.figsize = figsize
        self.dpi = dpi

        # Color palette for methods
        self.colors = {
            'scratch_t2': '#333333',      # Dark gray for baseline
            'sgd': '#1f77b4',             # Blue for naive CL
            'derpp': '#d62728',           # Red for DER++
            'ewc_on': '#2ca02c',          # Green for EWC
            'gpm': '#8c564b',             # Brown for GPM
            'dgr': '#9467bd',             # Purple for DGR
            'interleaved': '#e377c2'      # Pink for oracle
        }

    def get_method_color(self, method: str) -> str:
        """Get color for method."""
        return self.colors.get(method.lower(), '#17becf')  # default cyan

    def create_comparative_dynamics_figure(self,
                                         curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
                                         ad_values: Dict[str, float],
                                         pd_series: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                         sfr_series: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                         tau: float = 0.6,
                                         save_path: Optional[str] = None) -> Figure:
        """Create comprehensive comparative dynamics figure."""

        fig, axes = plt.subplots(3, 1, figsize=self.figsize, dpi=self.dpi)

        # Panel A: Accuracy trajectories
        ax_a = axes[0]
        self._plot_accuracy_trajectories(ax_a, curves, ad_values, tau)

        # Panel B: Performance Deficit
        ax_b = axes[1]
        self._plot_performance_deficits(ax_b, pd_series)

        # Panel C: Shortcut Forgetting Rate
        ax_c = axes[2]
        self._plot_sfr_relative(ax_c, sfr_series)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved comparative dynamics figure to {save_path}")

        return fig

    def _plot_accuracy_trajectories(self, ax: Axes,
                                  curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
                                  ad_values: Dict[str, float],
                                  tau: float):
        """Plot accuracy trajectories with AD markers."""

        ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

        # Plot shortcut_normal curves (solid lines)
        for method in curves.keys():
            if 'T2_shortcut_normal' not in curves[method]:
                continue

            epochs, mean_acc, conf_interval = curves[method]['T2_shortcut_normal']
            color = self.get_method_color(method)

            # Plot mean accuracy
            ax.plot(epochs, mean_acc, color=color, linewidth=2,
                   label=f"{method} (normal)", linestyle='-')

            # Add confidence interval
            if len(conf_interval) > 0 and np.any(conf_interval > 0):
                ax.fill_between(epochs, mean_acc - conf_interval, mean_acc + conf_interval,
                               color=color, alpha=0.2)

        # Plot shortcut_masked curves (dashed lines)
        for method in curves.keys():
            if 'T2_shortcut_masked' not in curves[method]:
                continue

            epochs, mean_acc, conf_interval = curves[method]['T2_shortcut_masked']
            color = self.get_method_color(method)

            # Plot mean accuracy
            ax.plot(epochs, mean_acc, color=color, linewidth=2,
                   label=f"{method} (masked)", linestyle='--')

            # Add confidence interval
            if len(conf_interval) > 0 and np.any(conf_interval > 0):
                ax.fill_between(epochs, mean_acc - conf_interval, mean_acc + conf_interval,
                               color=color, alpha=0.2)

        # Add threshold line
        ax.axhline(y=tau, color='gray', linestyle=':', alpha=0.7, label=f'τ = {tau}')

        # Add AD markers
        self._add_ad_markers(ax, curves, ad_values, tau)

        ax.set_xlabel('Effective Epoch')
        ax.set_ylabel('Top-1 Accuracy')
        ax.set_title('Accuracy Trajectories on Shortcut Task')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1)

    def _add_ad_markers(self, ax: Axes,
                       curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
                       ad_values: Dict[str, float],
                       tau: float):
        """Add AD markers and annotations."""

        # Find Scratch_T2 crossing epoch
        scratch_crossing = None
        if 'scratch_t2' in curves and 'T2_shortcut_normal' in curves['scratch_t2']:
            epochs, mean_acc, _ = curves['scratch_t2']['T2_shortcut_normal']
            crossing_indices = np.where(mean_acc >= tau)[0]
            if len(crossing_indices) > 0:
                scratch_crossing = epochs[crossing_indices[0]]

        if scratch_crossing is None:
            return

        # Add Scratch_T2 reference line
        ax.axvline(x=scratch_crossing, color=self.get_method_color('scratch_t2'),
                  linestyle='-', alpha=0.7, linewidth=1, label='E_S(τ)')

        # Add method crossing lines and annotations
        y_positions = np.linspace(0.1, 0.4, len(ad_values))

        for i, (method, ad_value) in enumerate(ad_values.items()):
            if np.isnan(ad_value):
                continue

            color = self.get_method_color(method)

            # Find method crossing epoch
            if method in curves and 'T2_shortcut_normal' in curves[method]:
                epochs, mean_acc, _ = curves[method]['T2_shortcut_normal']
                crossing_indices = np.where(mean_acc >= tau)[0]
                if len(crossing_indices) > 0:
                    method_crossing = epochs[crossing_indices[0]]

                    # Add vertical line
                    ax.axvline(x=method_crossing, color=color, linestyle='--', alpha=0.7, linewidth=1)

                    # Add annotation
                    y_pos = y_positions[i % len(y_positions)]
                    ax.annotate(f'AD = {ad_value:.1f}',
                               xy=(method_crossing, tau),
                               xytext=(method_crossing + 2, y_pos),
                               fontsize=10, color=color,
                               arrowprops=dict(arrowstyle='->', color=color, alpha=0.7, lw=1))

    def _plot_performance_deficits(self, ax: Axes, pd_series: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """Plot Performance Deficit time series."""

        ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

        if not pd_series:
            ax.text(0.5, 0.5, 'No Performance Deficit data available',
                   transform=ax.transAxes, fontsize=12, ha='center', va='center',
                   style='italic', alpha=0.7)
        else:
            for method, (epochs, pd_values) in pd_series.items():
                color = self.get_method_color(method)
                ax.plot(epochs, pd_values, color=color, linewidth=2, label=method)

            # Add zero reference line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax.legend()

        ax.set_xlabel('Effective Epoch')
        ax.set_ylabel('Performance Deficit (PD_t)')
        ax.set_title('Performance Deficit: A_S(e) - A_CL(e)')
        ax.grid(True, alpha=0.3)

    def _plot_sfr_relative(self, ax: Axes, sfr_series: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """Plot Shortcut Forgetting Rate relative time series."""

        ax.text(0.02, 0.98, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

        if not sfr_series:
            ax.text(0.5, 0.5, 'No Shortcut Forgetting Rate data available',
                   transform=ax.transAxes, fontsize=12, ha='center', va='center',
                   style='italic', alpha=0.7)
        else:
            for method, (epochs, sfr_values) in sfr_series.items():
                color = self.get_method_color(method)
                ax.plot(epochs, sfr_values, color=color, linewidth=2, label=method)

            # Add zero reference line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            ax.legend()

        ax.set_xlabel('Effective Epoch')
        ax.set_ylabel('Shortcut Forgetting Rate (SFR_rel)')
        ax.set_title('Relative Shortcut Forgetting: Δ_CL(e) - Δ_S(e)')
        ax.grid(True, alpha=0.3)


def main():
    """Main function to generate comparative ERI visualizations."""

    # Input and output paths
    session_dir = "einstellung_results/session_20250923-012304_seed42"
    output_dir = Path(session_dir) / "fixed_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize processor and plotter
        processor = ComparativeERIProcessor(tau=0.6, smoothing_window=3)
        plotter = ComparativeERIPlotter()

        # Load data from session directory
        logger.info(f"Loading session data from {session_dir}")
        df = processor.load_session_data(session_dir)

        # Compute curves
        logger.info("Computing accuracy curves...")
        curves = processor.compute_method_curves(df)

        # Compute derived metrics
        logger.info("Computing adaptation delays...")
        ad_values = processor.compute_adaptation_delays(curves)

        logger.info("Computing performance deficits...")
        pd_series = processor.compute_performance_deficits(curves)

        logger.info("Computing shortcut forgetting rates...")
        sfr_series = processor.compute_sfr_relative(curves)

        # Generate visualization
        logger.info("Generating comparative dynamics figure...")
        fig_path = output_dir / "comparative_eri_dynamics_fixed.pdf"
        fig = plotter.create_comparative_dynamics_figure(
            curves=curves,
            ad_values=ad_values,
            pd_series=pd_series,
            sfr_series=sfr_series,
            tau=processor.tau,
            save_path=str(fig_path)
        )

        # Also save as PNG for easy viewing
        png_path = output_dir / "comparative_eri_dynamics_fixed.png"
        fig.savefig(png_path, dpi=150, bbox_inches='tight')

        # Print summary
        logger.info("\n=== SUMMARY ===")
        logger.info(f"Methods analyzed: {list(curves.keys())}")
        logger.info(f"Adaptation Delays (τ={processor.tau}):")
        for method, ad in ad_values.items():
            if np.isnan(ad):
                logger.info(f"  {method}: CENSORED (never crossed threshold)")
            else:
                logger.info(f"  {method}: {ad:.2f} epochs")

        # Print final accuracies
        logger.info(f"\nFinal Top-1 Accuracies on T2_shortcut_normal:")
        for method in curves.keys():
            if 'T2_shortcut_normal' in curves[method]:
                epochs, mean_acc, _ = curves[method]['T2_shortcut_normal']
                final_acc = mean_acc[-1]
                logger.info(f"  {method}: {final_acc:.3f}")

        logger.info(f"\nVisualizations saved to:")
        logger.info(f"  PDF: {fig_path}")
        logger.info(f"  PNG: {png_path}")

        # Show the plot
        plt.show()

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
