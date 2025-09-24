"""
Overall ERI Metric Visualization - Composite score bar chart.

This module provides visualization for the overall ERI metric that combines
AD, PD, and SFR_rel into a single interpretable score.

Integrates with the corrected ERI metrics calculator to provide accurate
composite scores with proper error bars and color coding.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from .styles import PlotStyleConfig
from .metrics_calculator import ERIMetrics


class OverallERIPlotter:
    """
    Plotter for overall ERI metric visualization.

    Creates publication-ready bar charts showing composite ERI scores
    with confidence intervals and intuitive color coding.
    """

    def __init__(self, style_config: Optional[PlotStyleConfig] = None):
        """
        Initialize the plotter.

        Args:
            style_config: Plot styling configuration
        """
        self.style_config = style_config or PlotStyleConfig()
        self.logger = logging.getLogger(__name__)

        # Create color map for rigidity levels (green = low, red = high)
        self.rigidity_cmap = LinearSegmentedColormap.from_list(
            'rigidity', ['#2ca02c', '#ffff00', '#d62728'], N=256
        )

    def plot_overall_eri(self,
                        method_stats: Dict[str, Dict[str, float]],
                        output_path: Path,
                        ad_weight: float = 0.4,
                        pd_weight: float = 0.4,
                        sfr_weight: float = 0.2,
                        ad_max: float = 50.0,
                        figsize: Tuple[float, float] = (10, 6),
                        dpi: int = 300) -> None:
        """
        Generate overall ERI metric bar chart.

        Args:
            method_stats: Dictionary with method-level statistics
            output_path: Path to save the figure
            ad_weight: Weight for Adaptation Delay component
            pd_weight: Weight for Performance Deficit component
            sfr_weight: Weight for Shortcut Feature Reliance component
            ad_max: Maximum AD value for normalization
            figsize: Figure size (width, height)
            dpi: Figure DPI for high-quality output
        """
        # Compute overall ERI scores
        eri_scores = self._compute_overall_eri_scores(
            method_stats, ad_weight, pd_weight, sfr_weight, ad_max
        )

        if not eri_scores:
            raise ValueError("No valid ERI scores computed")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Sort methods by ERI score (ascending - lower is better)
        sorted_methods = sorted(eri_scores.keys(), key=lambda m: eri_scores[m]['mean'])

        # Prepare data for plotting
        methods = sorted_methods
        means = [eri_scores[m]['mean'] for m in methods]
        errors = [eri_scores[m]['ci'] for m in methods]
        colors = [self._get_rigidity_color(score) for score in means]

        # Create bar chart
        bars = ax.bar(range(len(methods)), means, yerr=errors,
                     capsize=5, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=0.5)

        # Add scratch baseline reference line if available
        if 'scratch_t2' in method_stats:
            scratch_eri = self._compute_single_eri_score(
                method_stats['scratch_t2'], ad_weight, pd_weight, sfr_weight, ad_max
            )
            if scratch_eri is not None:
                ax.axhline(y=scratch_eri['mean'], color='gray', linestyle='--',
                          alpha=0.7, label='Scratch_T2 baseline')

        # Customize plot
        ax.set_xlabel('Continual Learning Method', fontsize=self.style_config.label_fontsize)
        ax.set_ylabel('Overall ERI Score', fontsize=self.style_config.label_fontsize)
        ax.set_title('Overall Einstellung Rigidity Index (ERI)\nLower = Less Rigid = Better',
                    fontsize=self.style_config.title_fontsize, pad=20)

        # Set x-axis labels
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([self._format_method_name(m) for m in methods],
                          rotation=45, ha='right')

        # Add value annotations on bars
        for i, (bar, mean, error) in enumerate(zip(bars, means, errors)):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height + error),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)

        # Add component breakdown for best and worst methods
        if len(methods) >= 2:
            self._add_component_annotations(ax, method_stats, methods[0], methods[-1],
                                          ad_weight, pd_weight, sfr_weight)

        # Add color legend
        self._add_rigidity_legend(ax)

        # Add interpretation text
        ax.text(0.02, 0.98, 'Higher ERI = More Rigid = Worse Performance',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Grid and styling
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # Adjust layout and save
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Overall ERI plot saved to {output_path}")

    def _compute_overall_eri_scores(self,
                                  method_stats: Dict[str, Dict[str, float]],
                                  ad_weight: float,
                                  pd_weight: float,
                                  sfr_weight: float,
                                  ad_max: float) -> Dict[str, Dict[str, float]]:
        """Compute overall ERI scores for all methods."""
        eri_scores = {}

        for method, stats in method_stats.items():
            eri_score = self._compute_single_eri_score(
                stats, ad_weight, pd_weight, sfr_weight, ad_max
            )
            if eri_score is not None:
                eri_scores[method] = eri_score

        return eri_scores

    def _compute_single_eri_score(self,
                                stats: Dict[str, float],
                                ad_weight: float,
                                pd_weight: float,
                                sfr_weight: float,
                                ad_max: float) -> Optional[Dict[str, float]]:
        """Compute overall ERI score for a single method."""
        # Extract component means
        ad_mean = stats.get('AD_mean')
        pd_mean = stats.get('PD_mean')
        sfr_mean = stats.get('SFR_rel_mean')

        if None in [ad_mean, pd_mean, sfr_mean]:
            return None

        # Normalize AD to [0,1] range
        ad_normalized = min(abs(ad_mean) / ad_max, 1.0)

        # Ensure PD and SFR_rel are non-negative
        pd_normalized = max(0.0, pd_mean)
        sfr_normalized = max(0.0, sfr_mean)

        # Compute overall ERI
        eri_mean = (ad_weight * ad_normalized +
                   pd_weight * pd_normalized +
                   sfr_weight * sfr_normalized)

        # Propagate uncertainty (simplified - assumes independence)
        ad_ci = stats.get('AD_ci', 0.0)
        pd_ci = stats.get('PD_ci', 0.0)
        sfr_ci = stats.get('SFR_rel_ci', 0.0)

        # Normalize CI components
        ad_ci_norm = ad_ci / ad_max if ad_ci is not None else 0.0

        # Combine uncertainties (simplified)
        eri_ci = np.sqrt((ad_weight * ad_ci_norm) ** 2 +
                        (pd_weight * pd_ci) ** 2 +
                        (sfr_weight * sfr_ci) ** 2)

        return {
            'mean': eri_mean,
            'ci': eri_ci,
            'components': {
                'AD_norm': ad_normalized,
                'PD': pd_normalized,
                'SFR_rel': sfr_normalized
            }
        }

    def _get_rigidity_color(self, eri_score: float) -> str:
        """Get color for rigidity level."""
        # Normalize score to [0,1] for color mapping
        normalized_score = min(max(eri_score, 0.0), 1.0)
        rgba = self.rigidity_cmap(normalized_score)
        return rgba

    def _format_method_name(self, method: str) -> str:
        """Format method name for display."""
        # Convert to title case and handle common abbreviations
        formatted = method.replace('_', ' ').title()
        formatted = formatted.replace('Derpp', 'DER++')
        formatted = formatted.replace('Ewc', 'EWC')
        formatted = formatted.replace('Sgd', 'SGD')
        formatted = formatted.replace('Gpm', 'GPM')
        return formatted

    def _add_component_annotations(self,
                                 ax,
                                 method_stats: Dict[str, Dict[str, float]],
                                 best_method: str,
                                 worst_method: str,
                                 ad_weight: float,
                                 pd_weight: float,
                                 sfr_weight: float) -> None:
        """Add component breakdown annotations for best and worst methods."""
        # Get component values
        best_stats = method_stats[best_method]
        worst_stats = method_stats[worst_method]

        # Format component breakdown
        best_text = (f"Best ({self._format_method_name(best_method)}):\n"
                    f"AD: {best_stats.get('AD_mean', 0):.2f}, "
                    f"PD: {best_stats.get('PD_mean', 0):.3f}, "
                    f"SFR: {best_stats.get('SFR_rel_mean', 0):.3f}")

        worst_text = (f"Worst ({self._format_method_name(worst_method)}):\n"
                     f"AD: {worst_stats.get('AD_mean', 0):.2f}, "
                     f"PD: {worst_stats.get('PD_mean', 0):.3f}, "
                     f"SFR: {worst_stats.get('SFR_rel_mean', 0):.3f}")

        # Add annotations
        ax.text(0.02, 0.85, best_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        ax.text(0.02, 0.65, worst_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    def _add_rigidity_legend(self, ax) -> None:
        """Add color legend for rigidity levels."""
        # Create legend patches
        low_patch = mpatches.Patch(color='#2ca02c', label='Low Rigidity (Good)')
        med_patch = mpatches.Patch(color='#ffff00', label='Medium Rigidity')
        high_patch = mpatches.Patch(color='#d62728', label='High Rigidity (Bad)')

        # Add legend
        legend = ax.legend(handles=[low_patch, med_patch, high_patch],
                          loc='upper right', fontsize=9,
                          title='Rigidity Level')
        legend.get_title().set_fontsize(9)

    def create_eri_comparison_table(self,
                                  method_stats: Dict[str, Dict[str, float]],
                                  output_path: Path,
                                  ad_weight: float = 0.4,
                                  pd_weight: float = 0.4,
                                  sfr_weight: float = 0.2,
                                  ad_max: float = 50.0) -> None:
        """
        Create a detailed comparison table of ERI metrics.

        Args:
            method_stats: Dictionary with method-level statistics
            output_path: Path to save the table (CSV format)
            ad_weight: Weight for Adaptation Delay component
            pd_weight: Weight for Performance Deficit component
            sfr_weight: Weight for Shortcut Feature Reliance component
            ad_max: Maximum AD value for normalization
        """
        import pandas as pd

        # Compute overall ERI scores
        eri_scores = self._compute_overall_eri_scores(
            method_stats, ad_weight, pd_weight, sfr_weight, ad_max
        )

        # Create comparison table
        table_data = []

        for method in sorted(eri_scores.keys(), key=lambda m: eri_scores[m]['mean']):
            stats = method_stats[method]
            eri = eri_scores[method]

            row = {
                'Method': self._format_method_name(method),
                'Overall_ERI': f"{eri['mean']:.4f} ± {eri['ci']:.4f}",
                'AD_mean': f"{stats.get('AD_mean', np.nan):.3f}",
                'AD_ci': f"±{stats.get('AD_ci', np.nan):.3f}",
                'PD_mean': f"{stats.get('PD_mean', np.nan):.4f}",
                'PD_ci': f"±{stats.get('PD_ci', np.nan):.4f}",
                'SFR_rel_mean': f"{stats.get('SFR_rel_mean', np.nan):.4f}",
                'SFR_rel_ci': f"±{stats.get('SFR_rel_ci', np.nan):.4f}",
                'N_seeds': stats.get('AD_n', 0),
                'Censored_runs': stats.get('censored_runs', 0)
            }
            table_data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        self.logger.info(f"ERI comparison table saved to {output_path}")
