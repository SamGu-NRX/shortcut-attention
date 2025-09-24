"""
ERI Composite Score Calculator - Overall ERI metric computation.

This module provides functionality for computing composite ERI scores
that combine AD, PD, and SFR_rel into a single interpretable metric.

The composite score uses weighted combination with normalization:
ERI_overall = w_AD × AD_norm + w_PD × PD + w_SFR × SFR_rel
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .metrics_calculator import ERIMetrics


@dataclass
class CompositeERIConfig:
    """Configuration for composite ERI score computation."""
    ad_weight: float = 0.4      # Weight for Adaptation Delay
    pd_weight: float = 0.4      # Weight for Performance Deficit
    sfr_weight: float = 0.2     # Weight for Shortcut Feature Reliance
    ad_max: float = 50.0        # Maximum AD for normalization

    def __post_init__(self):
        """Validate configuration."""
        if not np.isclose(self.ad_weight + self.pd_weight + self.sfr_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
        if self.ad_max <= 0:
            raise ValueError("ad_max must be positive")


class CompositeERICalculator:
    """
    Calculator for composite ERI scores.

    Combines individual ERI facets (AD, PD, SFR_rel) into a single
    interpretable metric with proper normalization and uncertainty propagation.
    """

    def __init__(self, config: Optional[CompositeERIConfig] = None):
        """
        Initialize the calculator.

        Args:
            config: Configuration for composite score computation
        """
        self.config = config or CompositeERIConfig()
        self.logger = logging.getLogger(__name__)

    def compute_composite_score(self, metrics: ERIMetrics) -> Optional[float]:
        """
        Compute composite ERI score for a single metrics object.

        Args:
            metrics: ERIMetrics object with computed facets

        Returns:
            Composite ERI score, or None if incomplete data
        """
        if None in [metrics.adaptation_delay, metrics.performance_deficit,
                   metrics.shortcut_feature_reliance]:
            return None

        # Normalize AD to [0,1] range
        ad_normalized = min(abs(metrics.adaptation_delay) / self.config.ad_max, 1.0)

        # Ensure PD and SFR_rel are non-negative
        pd_normalized = max(0.0, metrics.performance_deficit)
        sfr_normalized = max(0.0, metrics.shortcut_feature_reliance)

        # Compute weighted combination
        composite_score = (self.config.ad_weight * ad_normalized +
                          self.config.pd_weight * pd_normalized +
                          self.config.sfr_weight * sfr_normalized)

        return composite_score

    def compute_method_composite_stats(self,
                                     metrics_list: List[ERIMetrics]) -> Dict[str, float]:
        """
        Compute composite score statistics for a list of metrics (e.g., across seeds).

        Args:
            metrics_list: List of ERIMetrics objects for the same method

        Returns:
            Dictionary with mean, std, ci, and component statistics
        """
        if not metrics_list:
            return {}

        # Compute composite scores
        composite_scores = []
        ad_values = []
        pd_values = []
        sfr_values = []

        for metrics in metrics_list:
            composite = self.compute_composite_score(metrics)
            if composite is not None:
                composite_scores.append(composite)

                # Store components for analysis
                ad_norm = min(abs(metrics.adaptation_delay) / self.config.ad_max, 1.0)
                pd_norm = max(0.0, metrics.performance_deficit)
                sfr_norm = max(0.0, metrics.shortcut_feature_reliance)

                ad_values.append(ad_norm)
                pd_values.append(pd_norm)
                sfr_values.append(sfr_norm)

        if not composite_scores:
            return {}

        # Compute statistics
        stats = {
            'composite_mean': np.mean(composite_scores),
            'composite_std': np.std(composite_scores, ddof=1) if len(composite_scores) > 1 else 0.0,
            'composite_n': len(composite_scores),
            'ad_component_mean': np.mean(ad_values),
            'pd_component_mean': np.mean(pd_values),
            'sfr_component_mean': np.mean(sfr_values),
        }

        # Compute confidence interval
        if len(composite_scores) > 1:
            from scipy import stats as scipy_stats
            t_val = scipy_stats.t.ppf(0.975, df=len(composite_scores)-1)
            sem = stats['composite_std'] / np.sqrt(len(composite_scores))
            stats['composite_ci'] = t_val * sem
        else:
            stats['composite_ci'] = 0.0

        return stats

    def rank_methods_by_composite(self,
                                method_metrics: Dict[str, List[ERIMetrics]]) -> List[Tuple[str, float]]:
        """
        Rank methods by their composite ERI scores.

        Args:
            method_metrics: Dictionary mapping method names to lists of ERIMetrics

        Returns:
            List of (method_name, mean_composite_score) tuples, sorted by score (ascending)
        """
        method_scores = []

        for method, metrics_list in method_metrics.items():
            stats = self.compute_method_composite_stats(metrics_list)
            if stats and 'composite_mean' in stats:
                method_scores.append((method, stats['composite_mean']))

        # Sort by score (ascending - lower is better)
        method_scores.sort(key=lambda x: x[1])

        return method_scores

    def analyze_component_contributions(self,
                                      method_metrics: Dict[str, List[ERIMetrics]]) -> Dict[str, Dict[str, float]]:
        """
        Analyze how each component contributes to the overall ERI score.

        Args:
            method_metrics: Dictionary mapping method names to lists of ERIMetrics

        Returns:
            Dictionary with component contribution analysis
        """
        analysis = {}

        for method, metrics_list in method_metrics.items():
            stats = self.compute_method_composite_stats(metrics_list)

            if stats:
                # Calculate weighted contributions
                ad_contribution = self.config.ad_weight * stats['ad_component_mean']
                pd_contribution = self.config.pd_weight * stats['pd_component_mean']
                sfr_contribution = self.config.sfr_weight * stats['sfr_component_mean']

                total = ad_contribution + pd_contribution + sfr_contribution

                analysis[method] = {
                    'ad_contribution': ad_contribution,
                    'pd_contribution': pd_contribution,
                    'sfr_contribution': sfr_contribution,
                    'ad_percentage': (ad_contribution / total * 100) if total > 0 else 0,
                    'pd_percentage': (pd_contribution / total * 100) if total > 0 else 0,
                    'sfr_percentage': (sfr_contribution / total * 100) if total > 0 else 0,
                    'total_score': total
                }

        return analysis

    def generate_interpretation_text(self,
                                   method_scores: List[Tuple[str, float]],
                                   component_analysis: Dict[str, Dict[str, float]]) -> str:
        """
        Generate human-readable interpretation of ERI results.

        Args:
            method_scores: Ranked method scores from rank_methods_by_composite
            component_analysis: Component analysis from analyze_component_contributions

        Returns:
            Formatted interpretation text
        """
        if not method_scores:
            return "No valid ERI scores computed."

        best_method, best_score = method_scores[0]
        worst_method, worst_score = method_scores[-1]

        interpretation = f"""
ERI Analysis Summary:

Best Method: {best_method} (ERI = {best_score:.4f})
Worst Method: {worst_method} (ERI = {worst_score:.4f})

Lower ERI scores indicate less rigidity and better continual learning performance.

Component Analysis for Best Method ({best_method}):
"""

        if best_method in component_analysis:
            best_analysis = component_analysis[best_method]
            interpretation += f"""
- Adaptation Delay contributes {best_analysis['ad_percentage']:.1f}% ({best_analysis['ad_contribution']:.4f})
- Performance Deficit contributes {best_analysis['pd_percentage']:.1f}% ({best_analysis['pd_contribution']:.4f})
- Shortcut Feature Reliance contributes {best_analysis['sfr_percentage']:.1f}% ({best_analysis['sfr_contribution']:.4f})
"""

        interpretation += f"""
Component Analysis for Worst Method ({worst_method}):
"""

        if worst_method in component_analysis:
            worst_analysis = component_analysis[worst_method]
            interpretation += f"""
- Adaptation Delay contributes {worst_analysis['ad_percentage']:.1f}% ({worst_analysis['ad_contribution']:.4f})
- Performance Deficit contributes {worst_analysis['pd_percentage']:.1f}% ({worst_analysis['pd_contribution']:.4f})
- Shortcut Feature Reliance contributes {worst_analysis['sfr_percentage']:.1f}% ({worst_analysis['sfr_contribution']:.4f})
"""

        interpretation += f"""
Method Ranking (best to worst):
"""
        for i, (method, score) in enumerate(method_scores, 1):
            interpretation += f"{i}. {method}: {score:.4f}\n"

        return interpretation.strip()

    def validate_composite_scores(self,
                                method_metrics: Dict[str, List[ERIMetrics]]) -> bool:
        """
        Validate that composite scores are reasonable and consistent.

        Args:
            method_metrics: Dictionary mapping method names to lists of ERIMetrics

        Returns:
            True if validation passes, False otherwise
        """
        try:
            method_scores = self.rank_methods_by_composite(method_metrics)

            if not method_scores:
                self.logger.warning("No composite scores computed for validation")
                return False

            # Check that scores are in reasonable range [0, 1]
            for method, score in method_scores:
                if not (0.0 <= score <= 1.0):
                    self.logger.warning(f"Composite score out of range for {method}: {score}")
                    return False

            # Check that there's reasonable variation between methods
            scores = [score for _, score in method_scores]
            score_range = max(scores) - min(scores)

            if score_range < 0.01:  # Very small range might indicate calculation issues
                self.logger.warning(f"Very small score range detected: {score_range}")
                return False

            self.logger.info(f"Composite score validation passed. Range: {score_range:.4f}")
            return True

        except Exception as e:
            self.logger.error(f"Composite score validation failed: {e}")
            return False
