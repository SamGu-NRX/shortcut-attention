# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Einstellung Rigidity Index (ERI) Metrics Implementation

This module provides comprehensive metrics for measuring cognitive rigidity
in continual learning through the Einstellung Effect paradigm.

Core Metrics:
- Adaptation Delay (AD): Epochs required to reach threshold accuracy
- Performance Deficit (PD): Accuracy drop when shortcuts are removed
- Shortcut Feature Reliance (SFR): Relative dependency on shortcut features
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging


@dataclass
class EinstellungTimelineData:
    """Container for timeline data during training."""
    epoch: int
    task_id: int
    subset_name: str  # e.g., 'T1_all', 'T2_shortcut_normal', etc.
    accuracy: float
    loss: float
    timestamp: float


@dataclass
class EinstellungMetrics:
    """Container for computed ERI metrics."""
    # Core ERI metrics
    adaptation_delay: Optional[float] = None
    performance_deficit: Optional[float] = None
    shortcut_feature_reliance: Optional[float] = None

    # Extended metrics
    task1_accuracy_final: Optional[float] = None
    task2_shortcut_accuracy: Optional[float] = None
    task2_masked_accuracy: Optional[float] = None
    task2_nonshortcut_accuracy: Optional[float] = None

    # Timeline metrics
    epochs_to_convergence: Optional[int] = None
    final_loss: Optional[float] = None

    # Composite ERI score
    eri_score: Optional[float] = None

    def compute_eri_score(self,
                         ad_weight: float = 0.4,
                         pd_weight: float = 0.4,
                         sfr_weight: float = 0.2) -> float:
        """
        Compute composite ERI score from individual metrics.

        Args:
            ad_weight: Weight for Adaptation Delay component
            pd_weight: Weight for Performance Deficit component
            sfr_weight: Weight for Shortcut Feature Reliance component

        Returns:
            Composite ERI score (higher = more rigid)
        """
        if None in [self.adaptation_delay, self.performance_deficit, self.shortcut_feature_reliance]:
            return None

        # Normalize AD to [0,1] range (assuming max reasonable AD is 50 epochs)
        ad_normalized = min(self.adaptation_delay / 50.0, 1.0)

        # PD and SFR are already in [0,1] range
        pd_normalized = self.performance_deficit
        sfr_normalized = self.shortcut_feature_reliance

        self.eri_score = (ad_weight * ad_normalized +
                         pd_weight * pd_normalized +
                         sfr_weight * sfr_normalized)
        return self.eri_score


class EinstellungMetricsCalculator:
    """
    Calculator for Einstellung Rigidity Index (ERI) metrics.

    Tracks performance across different evaluation subsets and computes
    metrics that quantify cognitive rigidity in continual learning.
    """

    def __init__(self, adaptation_threshold: float = 0.8):
        """
        Initialize the metrics calculator.

        Args:
            adaptation_threshold: Accuracy threshold for Adaptation Delay calculation
        """
        self.adaptation_threshold = adaptation_threshold
        self.timeline_data: List[EinstellungTimelineData] = []
        self.logger = logging.getLogger(__name__)

    def add_timeline_data(self,
                         epoch: int,
                         task_id: int,
                         subset_accuracies: Dict[str, float],
                         subset_losses: Dict[str, float],
                         timestamp: float = None) -> None:
        """
        Add timeline data for a training epoch.

        Args:
            epoch: Training epoch number
            task_id: Current task ID
            subset_accuracies: Dictionary of subset name -> accuracy
            subset_losses: Dictionary of subset name -> loss
            timestamp: Unix timestamp (auto-generated if None)
        """
        if timestamp is None:
            timestamp = torch.time()

        for subset_name, accuracy in subset_accuracies.items():
            loss = subset_losses.get(subset_name, 0.0)

            self.timeline_data.append(EinstellungTimelineData(
                epoch=epoch,
                task_id=task_id,
                subset_name=subset_name,
                accuracy=accuracy,
                loss=loss,
                timestamp=timestamp
            ))

    def calculate_adaptation_delay(self, subset_name: str = 'T1_all') -> Optional[float]:
        """
        Calculate Adaptation Delay: epochs required to reach threshold accuracy.

        Args:
            subset_name: Evaluation subset to analyze

        Returns:
            Number of epochs to reach threshold, or None if never reached
        """
        subset_data = [d for d in self.timeline_data if d.subset_name == subset_name]
        if not subset_data:
            self.logger.warning(f"No timeline data found for subset {subset_name}")
            return None

        # Sort by epoch to ensure chronological order
        subset_data.sort(key=lambda x: x.epoch)

        for data_point in subset_data:
            if data_point.accuracy >= self.adaptation_threshold:
                return float(data_point.epoch)

        # Threshold never reached
        return None

    def calculate_performance_deficit(self) -> Optional[float]:
        """
        Calculate Performance Deficit: accuracy drop when shortcuts are removed.

        Compares T2_shortcut_normal vs T2_shortcut_masked accuracy on final epoch.

        Returns:
            Performance deficit as fraction (0-1), or None if data unavailable
        """
        # Get final epoch accuracies for shortcut vs masked comparison
        normal_acc = self._get_final_accuracy('T2_shortcut_normal')
        masked_acc = self._get_final_accuracy('T2_shortcut_masked')

        if normal_acc is None or masked_acc is None:
            self.logger.warning("Insufficient data for Performance Deficit calculation")
            return None

        # PD = (acc_with_shortcuts - acc_without_shortcuts) / acc_with_shortcuts
        if normal_acc == 0:
            return 0.0

        deficit = (normal_acc - masked_acc) / normal_acc
        return max(0.0, deficit)  # Ensure non-negative

    def calculate_shortcut_feature_reliance(self) -> Optional[float]:
        """
        Calculate Shortcut Feature Reliance: relative dependency on shortcuts.

        Compares performance on shortcut vs non-shortcut classes in Task 2.

        Returns:
            SFR score (0-1), or None if data unavailable
        """
        shortcut_acc = self._get_final_accuracy('T2_shortcut_normal')
        nonshortcut_acc = self._get_final_accuracy('T2_nonshortcut_normal')

        if shortcut_acc is None or nonshortcut_acc is None:
            self.logger.warning("Insufficient data for SFR calculation")
            return None

        # SFR = shortcut_accuracy / (shortcut_accuracy + nonshortcut_accuracy)
        total_acc = shortcut_acc + nonshortcut_acc
        if total_acc == 0:
            return 0.5  # Equal reliance when both are zero

        return shortcut_acc / total_acc

    def calculate_comprehensive_metrics(self) -> EinstellungMetrics:
        """
        Calculate all ERI metrics and return comprehensive results.

        Returns:
            EinstellungMetrics object with all computed metrics
        """
        metrics = EinstellungMetrics()

        # Core ERI metrics
        metrics.adaptation_delay = self.calculate_adaptation_delay()
        metrics.performance_deficit = self.calculate_performance_deficit()
        metrics.shortcut_feature_reliance = self.calculate_shortcut_feature_reliance()

        # Extended metrics
        metrics.task1_accuracy_final = self._get_final_accuracy('T1_all')
        metrics.task2_shortcut_accuracy = self._get_final_accuracy('T2_shortcut_normal')
        metrics.task2_masked_accuracy = self._get_final_accuracy('T2_shortcut_masked')
        metrics.task2_nonshortcut_accuracy = self._get_final_accuracy('T2_nonshortcut_normal')

        # Timeline metrics
        metrics.epochs_to_convergence = self._calculate_convergence_epoch()
        metrics.final_loss = self._get_final_loss('T1_all')

        # Compute composite score
        metrics.compute_eri_score()

        return metrics

    def _get_final_accuracy(self, subset_name: str) -> Optional[float]:
        """Get accuracy for the final epoch of a specific subset."""
        subset_data = [d for d in self.timeline_data if d.subset_name == subset_name]
        if not subset_data:
            return None

        # Return accuracy from the latest epoch
        latest_data = max(subset_data, key=lambda x: x.epoch)
        return latest_data.accuracy

    def _get_final_loss(self, subset_name: str) -> Optional[float]:
        """Get loss for the final epoch of a specific subset."""
        subset_data = [d for d in self.timeline_data if d.subset_name == subset_name]
        if not subset_data:
            return None

        latest_data = max(subset_data, key=lambda x: x.epoch)
        return latest_data.loss

    def _calculate_convergence_epoch(self,
                                   subset_name: str = 'T1_all',
                                   window_size: int = 5,
                                   stability_threshold: float = 0.01) -> Optional[int]:
        """
        Calculate epoch where accuracy converged (stopped improving significantly).

        Args:
            subset_name: Subset to analyze
            window_size: Number of epochs to check for stability
            stability_threshold: Maximum change allowed for convergence

        Returns:
            Epoch number where convergence occurred, or None
        """
        subset_data = [d for d in self.timeline_data if d.subset_name == subset_name]
        if len(subset_data) < window_size:
            return None

        subset_data.sort(key=lambda x: x.epoch)

        for i in range(window_size, len(subset_data)):
            # Check if accuracy has been stable over the window
            window_accuracies = [subset_data[j].accuracy for j in range(i-window_size, i)]
            accuracy_range = max(window_accuracies) - min(window_accuracies)

            if accuracy_range <= stability_threshold:
                return subset_data[i-window_size].epoch

        return None

    def export_timeline_data(self) -> List[Dict]:
        """
        Export timeline data for external analysis or visualization.

        Returns:
            List of dictionaries containing timeline data
        """
        return [
            {
                'epoch': d.epoch,
                'task_id': d.task_id,
                'subset_name': d.subset_name,
                'accuracy': d.accuracy,
                'loss': d.loss,
                'timestamp': d.timestamp
            }
            for d in self.timeline_data
        ]

    def get_accuracy_curves(self) -> Dict[str, Tuple[List[int], List[float]]]:
        """
        Get accuracy curves for plotting.

        Returns:
            Dictionary mapping subset_name -> (epochs, accuracies)
        """
        curves = {}

        # Group data by subset
        subset_groups = {}
        for data_point in self.timeline_data:
            if data_point.subset_name not in subset_groups:
                subset_groups[data_point.subset_name] = []
            subset_groups[data_point.subset_name].append(data_point)

        # Convert to curves
        for subset_name, data_points in subset_groups.items():
            data_points.sort(key=lambda x: x.epoch)
            epochs = [d.epoch for d in data_points]
            accuracies = [d.accuracy for d in data_points]
            curves[subset_name] = (epochs, accuracies)

        return curves


def calculate_cross_experiment_eri_statistics(metrics_list: List[EinstellungMetrics]) -> Dict[str, float]:
    """
    Calculate statistical summaries across multiple experiment runs.

    Args:
        metrics_list: List of EinstellungMetrics from different runs

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}

    stats = {}
    metric_names = ['adaptation_delay', 'performance_deficit', 'shortcut_feature_reliance', 'eri_score']

    for metric_name in metric_names:
        values = [getattr(m, metric_name) for m in metrics_list if getattr(m, metric_name) is not None]

        if values:
            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        else:
            stats[metric_name] = {
                'mean': None, 'std': None, 'min': None, 'max': None, 'count': 0
            }

    return stats
