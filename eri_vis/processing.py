"""
ERITimelineProcessor - Metric calculations and analysis for ERI timeline data.

This module provides the ERITimelineProcessor class for computing ERI metrics
including smoothing, confidence intervals, adaptation delays, performance deficits,
and shortcut forgetting rates.

Integrates with the existing Mammoth Einstellung experiment infrastructure:
- Compatible with utils/einstellung_evaluator.py metric definitions
- Supports all existing Mammoth continual learning strategies
- Works with datasets/seq_cifar100_einstellung_224.py evaluation splits
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from .dataset import ERITimelineDataset


@dataclass
class AccuracyCurve:
    """
    Represents an accuracy curve with confidence intervals.

    Attributes:
        epochs: Array of epoch values
        mean_accuracy: Array of mean accuracy values across seeds
        conf_interval: Array of 95% confidence interval half-widths
        method: Method name (e.g., 'sgd', 'ewc_on', 'Scratch_T2')
        split: Split name (e.g., 'T2_shortcut_normal', 'T2_shortcut_masked')
        n_seeds: Number of seeds used to compute statistics
        raw_data: Optional raw per-seed data for debugging
    """
    epochs: np.ndarray
    mean_accuracy: np.ndarray
    conf_interval: np.ndarray
    method: str
    split: str
    n_seeds: int = 0
    raw_data: Optional[np.ndarray] = None  # shape: (n_seeds, n_epochs)


@dataclass
class TimeSeries:
    """
    Represents a time series with optional confidence intervals.

    Attributes:
        epochs: Array of epoch values
        values: Array of time series values
        conf_interval: Array of confidence interval half-widths (may be empty)
        method: Method name
        metric_name: Name of the metric (e.g., 'PD_t', 'SFR_rel')
    """
    epochs: np.ndarray
    values: np.ndarray
    conf_interval: np.ndarray
    method: str
    metric_name: str


class ERITimelineProcessor:
    """
    Processor for computing ERI metrics from timeline data.

    This class provides methods for:
    - Smoothing accuracy curves with configurable window size
    - Computing accuracy curves with 95% confidence intervals
    - Computing Adaptation Delay (AD) with threshold crossing detection
    - Computing Performance Deficit (PD_t) time series
    - Computing Shortcut Forgetting Rate relative (SFR_rel) time series
    """

    def __init__(self, smoothing_window: int = 3, tau: float = 0.6):
        """
        Initialize the processor.

        Args:
            smoothing_window: Window size for smoothing (default 3)
            tau: Threshold for adaptation delay computation (default 0.6)
        """
        self.smoothing_window = smoothing_window
        self.tau = tau
        self.logger = logging.getLogger(__name__)

        if smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be between 0.0 and 1.0")

    def smooth_curve(self, values: np.ndarray, window: Optional[int] = None) -> np.ndarray:
        """
        Apply smoothing to a curve using centered moving average with edge padding.

        Args:
            values: Array of values to smooth
            window: Window size (uses self.smoothing_window if None)

        Returns:
            Smoothed array of same length as input
        """
        if window is None:
            window = self.smoothing_window

        if window <= 1:
            return values.copy()

        if len(values) == 0:
            return values.copy()

        # Use pandas rolling mean with center=True for centered window
        # and min_periods=1 to handle edges
        series = pd.Series(values)
        smoothed = series.rolling(window=window, center=True, min_periods=1).mean()

        return smoothed.values

    def compute_confidence_interval(self, data: np.ndarray, axis: int = 0,
                                  confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and confidence interval using t-distribution.

        Args:
            data: Array of data (seeds x epochs or similar)
            axis: Axis along which to compute statistics
            confidence: Confidence level (default 0.95)

        Returns:
            Tuple of (mean, ci_half_width) arrays
        """
        if data.size == 0:
            return np.array([]), np.array([])

        mean_vals = np.mean(data, axis=axis)

        if data.shape[axis] <= 1:
            # Single sample - no confidence interval
            ci_half_width = np.zeros_like(mean_vals)
        else:
            # Multiple samples - use t-distribution
            std_vals = np.std(data, axis=axis, ddof=1)  # Sample standard deviation
            n_samples = data.shape[axis]

            # t-critical value for 95% CI
            alpha = 1 - confidence
            t_critical = stats.t.ppf(1 - alpha/2, df=n_samples - 1)

            # Standard error of the mean
            sem = std_vals / np.sqrt(n_samples)

            # Confidence interval half-width
            ci_half_width = t_critical * sem

        return mean_vals, ci_half_width

    def compute_accuracy_curves(self, ds: ERITimelineDataset) -> Dict[str, AccuracyCurve]:
        """
        Compute accuracy curves with confidence intervals for all method-split combinations.

        Args:
            ds: ERITimelineDataset containing the timeline data

        Returns:
            Dictionary mapping "{method}_{split}" to AccuracyCurve objects

        Raises:
            ValueError: If dataset is empty or missing required data
        """
        if len(ds.data) == 0:
            raise ValueError("Dataset is empty")

        curves = {}

        # Process each method-split combination
        for method in ds.methods:
            for split in ds.splits:
                try:
                    curve = self._compute_single_accuracy_curve(ds, method, split)
                    if curve is not None:
                        key = f"{method}_{split}"
                        curves[key] = curve
                except Exception as e:
                    self.logger.warning(f"Failed to compute curve for {method}_{split}: {e}")
                    continue

        if not curves:
            raise ValueError("No valid accuracy curves could be computed")

        return curves

    def _compute_single_accuracy_curve(self, ds: ERITimelineDataset,
                                     method: str, split: str) -> Optional[AccuracyCurve]:
        """Compute accuracy curve for a single method-split combination."""
        # Filter data for this method-split combination
        method_split_data = ds.data[
            (ds.data['method'] == method) & (ds.data['split'] == split)
        ].copy()

        if len(method_split_data) == 0:
            self.logger.warning(f"No data found for {method}_{split}")
            return None

        # Get unique epochs and seeds
        unique_epochs = np.sort(method_split_data['epoch_eff'].unique())
        unique_seeds = np.sort(method_split_data['seed'].unique())

        if len(unique_epochs) == 0:
            self.logger.warning(f"No epochs found for {method}_{split}")
            return None

        # Create matrix: seeds x epochs
        acc_matrix = np.full((len(unique_seeds), len(unique_epochs)), np.nan)

        # Fill matrix with accuracy values
        for i, seed in enumerate(unique_seeds):
            seed_data = method_split_data[method_split_data['seed'] == seed]
            for j, epoch in enumerate(unique_epochs):
                epoch_data = seed_data[seed_data['epoch_eff'] == epoch]
                if len(epoch_data) > 0:
                    # Take mean if multiple entries (shouldn't happen with clean data)
                    acc_matrix[i, j] = epoch_data['acc'].mean()

        # Check for missing data
        missing_fraction = np.isnan(acc_matrix).sum() / acc_matrix.size
        if missing_fraction > 0.1:  # More than 10% missing
            self.logger.warning(
                f"High missing data fraction ({missing_fraction:.1%}) for {method}_{split}"
            )

        # Handle missing data by forward/backward fill per seed
        for i in range(acc_matrix.shape[0]):
            seed_row = acc_matrix[i, :]
            if np.all(np.isnan(seed_row)):
                self.logger.warning(f"All NaN for {method}_{split}, seed {unique_seeds[i]}")
                continue

            # Forward fill then backward fill
            mask = ~np.isnan(seed_row)
            if np.any(mask):
                # Forward fill
                last_valid = None
                for j in range(len(seed_row)):
                    if mask[j]:
                        last_valid = seed_row[j]
                    elif last_valid is not None:
                        seed_row[j] = last_valid

                # Backward fill
                last_valid = None
                for j in range(len(seed_row) - 1, -1, -1):
                    if mask[j]:
                        last_valid = seed_row[j]
                    elif last_valid is not None:
                        seed_row[j] = last_valid

                acc_matrix[i, :] = seed_row

        # Remove seeds that are still all NaN
        valid_seeds_mask = ~np.all(np.isnan(acc_matrix), axis=1)
        if not np.any(valid_seeds_mask):
            self.logger.warning(f"No valid seeds found for {method}_{split}")
            return None

        acc_matrix = acc_matrix[valid_seeds_mask, :]
        valid_seeds = unique_seeds[valid_seeds_mask]

        # Apply smoothing to each seed's curve
        smoothed_matrix = np.zeros_like(acc_matrix)
        for i in range(acc_matrix.shape[0]):
            smoothed_matrix[i, :] = self.smooth_curve(acc_matrix[i, :])

        # Compute mean and confidence intervals
        mean_acc, ci_half_width = self.compute_confidence_interval(smoothed_matrix, axis=0)

        return AccuracyCurve(
            epochs=unique_epochs,
            mean_accuracy=mean_acc,
            conf_interval=ci_half_width,
            method=method,
            split=split,
            n_seeds=len(valid_seeds),
            raw_data=smoothed_matrix
        )

    def compute_adaptation_delays(self, curves: Dict[str, AccuracyCurve]) -> Dict[str, float]:
        """
        Compute Adaptation Delay (AD) for each method.

        AD is defined as the difference in epochs where methods first cross
        the threshold τ on the shortcut normal split:
        AD = E_CL(τ) - E_S(τ)

        Args:
            curves: Dictionary of AccuracyCurve objects

        Returns:
            Dictionary mapping method names to AD values (NaN if censored)
        """
        ad_values = {}

        # Find Scratch_T2 crossing epoch for reference
        scratch_key = None
        for key in curves.keys():
            if key.startswith('Scratch_T2_') and 'shortcut_normal' in key:
                scratch_key = key
                break

        if scratch_key is None:
            self.logger.warning("No Scratch_T2 shortcut_normal curve found for AD computation")
            return ad_values  # Return empty dict
        else:
            scratch_crossing = self._find_threshold_crossing(curves[scratch_key], self.tau)

        # Compute AD for each method
        for key, curve in curves.items():
            if not key.endswith('_T2_shortcut_normal') and not 'shortcut_normal' in key:
                continue  # Only compute AD for shortcut_normal split

            method = curve.method
            if method == 'Scratch_T2':
                continue  # Skip scratch baseline

            method_crossing = self._find_threshold_crossing(curve, self.tau)

            if np.isnan(method_crossing) or np.isnan(scratch_crossing):
                ad_values[method] = np.nan
                if np.isnan(method_crossing):
                    self.logger.warning(f"Censored run detected for {method} (no threshold crossing)")
            else:
                ad_values[method] = method_crossing - scratch_crossing

        return ad_values

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

    def compute_performance_deficits(self, curves: Dict[str, AccuracyCurve],
                                   scratch_key: str = "Scratch_T2") -> Dict[str, TimeSeries]:
        """
        Compute Performance Deficit (PD_t) time series.

        PD_t(e) = A_S(e) - A_CL(e) where:
        - A_S(e) is Scratch_T2 accuracy at epoch e
        - A_CL(e) is continual learning method accuracy at epoch e

        Args:
            curves: Dictionary of AccuracyCurve objects
            scratch_key: Key prefix for scratch baseline method

        Returns:
            Dictionary mapping method names to PD_t TimeSeries objects
        """
        pd_series = {}

        # Find scratch baseline curve for shortcut_normal split
        scratch_curve = None
        for key, curve in curves.items():
            if key.startswith(scratch_key) and 'shortcut_normal' in key:
                scratch_curve = curve
                break

        if scratch_curve is None:
            self.logger.warning(f"No {scratch_key} shortcut_normal curve found for PD_t computation")
            return pd_series

        # Compute PD_t for each continual learning method
        for key, curve in curves.items():
            if not 'shortcut_normal' in key:
                continue  # Only compute PD_t for shortcut_normal split

            method = curve.method
            if method == scratch_key:
                continue  # Skip scratch baseline

            try:
                # Align epochs between scratch and method curves
                aligned_epochs, scratch_aligned, method_aligned = self._align_curves(
                    scratch_curve, curve
                )

                if len(aligned_epochs) == 0:
                    self.logger.warning(f"No overlapping epochs for PD_t computation: {method}")
                    continue

                # Compute PD_t = A_S - A_CL
                pd_values = scratch_aligned - method_aligned

                pd_series[method] = TimeSeries(
                    epochs=aligned_epochs,
                    values=pd_values,
                    conf_interval=np.array([]),  # No CI for derived metric
                    method=method,
                    metric_name='PD_t'
                )

            except Exception as e:
                self.logger.warning(f"Failed to compute PD_t for {method}: {e}")
                continue

        return pd_series

    def compute_sfr_relative(self, curves: Dict[str, AccuracyCurve],
                           scratch_key: str = "Scratch_T2") -> Dict[str, TimeSeries]:
        """
        Compute Shortcut Forgetting Rate relative (SFR_rel) time series.

        SFR_rel(e) = Δ_CL(e) - Δ_S(e) where:
        - Δ_M(e) = Acc(M, SC_patched, e) - Acc(M, SC_masked, e)
        - M is the method (CL or Scratch)

        Args:
            curves: Dictionary of AccuracyCurve objects
            scratch_key: Key prefix for scratch baseline method

        Returns:
            Dictionary mapping method names to SFR_rel TimeSeries objects
        """
        sfr_series = {}

        # Find scratch baseline curves
        scratch_patched = None
        scratch_masked = None

        for key, curve in curves.items():
            if key.startswith(scratch_key):
                if 'shortcut_normal' in key:
                    scratch_patched = curve
                elif 'shortcut_masked' in key:
                    scratch_masked = curve

        if scratch_patched is None or scratch_masked is None:
            self.logger.warning(f"Missing {scratch_key} curves for SFR_rel computation")
            return sfr_series

        # Compute Δ_S(e) for scratch baseline
        try:
            aligned_epochs_s, patched_s, masked_s = self._align_curves(
                scratch_patched, scratch_masked
            )
            if len(aligned_epochs_s) == 0:
                self.logger.warning("No overlapping epochs for scratch Δ computation")
                return sfr_series

            delta_scratch = patched_s - masked_s

        except Exception as e:
            self.logger.warning(f"Failed to compute Δ_S: {e}")
            return sfr_series

        # Compute SFR_rel for each continual learning method
        for method in set(curve.method for curve in curves.values()):
            if method == scratch_key:
                continue  # Skip scratch baseline

            # Find method curves
            method_patched = None
            method_masked = None

            for key, curve in curves.items():
                if curve.method == method:
                    if 'shortcut_normal' in key:
                        method_patched = curve
                    elif 'shortcut_masked' in key:
                        method_masked = curve

            if method_patched is None or method_masked is None:
                self.logger.warning(f"Missing curves for {method} SFR_rel computation")
                continue

            try:
                # Compute Δ_CL(e) for this method
                aligned_epochs_cl, patched_cl, masked_cl = self._align_curves(
                    method_patched, method_masked
                )

                if len(aligned_epochs_cl) == 0:
                    self.logger.warning(f"No overlapping epochs for {method} Δ computation")
                    continue

                delta_method = patched_cl - masked_cl

                # Align Δ_CL and Δ_S to common epochs
                common_epochs = np.intersect1d(aligned_epochs_cl, aligned_epochs_s)
                if len(common_epochs) == 0:
                    self.logger.warning(f"No common epochs for {method} SFR_rel computation")
                    continue

                # Interpolate both deltas to common epochs
                delta_cl_interp = np.interp(common_epochs, aligned_epochs_cl, delta_method)
                delta_s_interp = np.interp(common_epochs, aligned_epochs_s, delta_scratch)

                # Compute SFR_rel = Δ_CL - Δ_S
                sfr_rel_values = delta_cl_interp - delta_s_interp

                sfr_series[method] = TimeSeries(
                    epochs=common_epochs,
                    values=sfr_rel_values,
                    conf_interval=np.array([]),  # No CI for derived metric
                    method=method,
                    metric_name='SFR_rel'
                )

            except Exception as e:
                self.logger.warning(f"Failed to compute SFR_rel for {method}: {e}")
                continue

        return sfr_series

    def _align_curves(self, curve1: AccuracyCurve, curve2: AccuracyCurve) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align two curves to common epochs using interpolation.

        Args:
            curve1: First AccuracyCurve
            curve2: Second AccuracyCurve

        Returns:
            Tuple of (common_epochs, curve1_aligned, curve2_aligned)
        """
        # Find overlapping epoch range
        min_epoch = max(curve1.epochs.min(), curve2.epochs.min())
        max_epoch = min(curve1.epochs.max(), curve2.epochs.max())

        if min_epoch >= max_epoch:
            return np.array([]), np.array([]), np.array([])

        # Create common epoch grid
        # Use the finer resolution of the two curves
        n_points1 = len(curve1.epochs)
        n_points2 = len(curve2.epochs)
        n_common = max(n_points1, n_points2)

        common_epochs = np.linspace(min_epoch, max_epoch, n_common)

        # Interpolate both curves to common epochs
        curve1_aligned = np.interp(common_epochs, curve1.epochs, curve1.mean_accuracy)
        curve2_aligned = np.interp(common_epochs, curve2.epochs, curve2.mean_accuracy)

        return common_epochs, curve1_aligned, curve2_aligned

    def get_processing_summary(self, curves: Dict[str, AccuracyCurve]) -> Dict[str, any]:
        """
        Get summary statistics about the processed curves.

        Args:
            curves: Dictionary of processed AccuracyCurve objects

        Returns:
            Dictionary containing processing summary statistics
        """
        if not curves:
            return {'n_curves': 0, 'methods': [], 'splits': []}

        methods = set()
        splits = set()
        n_seeds_per_curve = {}
        epoch_ranges = {}

        for key, curve in curves.items():
            methods.add(curve.method)
            splits.add(curve.split)
            n_seeds_per_curve[key] = curve.n_seeds
            epoch_ranges[key] = (curve.epochs.min(), curve.epochs.max())

        return {
            'n_curves': len(curves),
            'methods': sorted(list(methods)),
            'splits': sorted(list(splits)),
            'smoothing_window': self.smoothing_window,
            'tau_threshold': self.tau,
            'n_seeds_per_curve': n_seeds_per_curve,
            'epoch_ranges': epoch_ranges,
            'curves_per_method': {method: sum(1 for c in curves.values() if c.method == method)
                                for method in methods},
            'curves_per_split': {split: sum(1 for c in curves.values() if c.split == split)
                               for split in splits}
        }
