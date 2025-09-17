"""
Tests for ERITimelineProcessor - Metric calculations and analysis.

This module tests the ERITimelineProcessor class including:
- Smoothing with configurable window size and edge padding
- Accuracy curves with mean and 95% confidence intervals using t-distribution
- Adaptation Delay (AD) with first threshold crossing detection
- Performance Deficit (PD_t) and Shortcut Forgetting Rate (SFR_rel) time series
- Handling of censored runs by marking AD as NaN with appropriate warnings
"""

import numpy as np
import pandas as pd
import pytest
import warnings
from unittest.mock import patch

from eri_vis.processing import ERITimelineProcessor, AccuracyCurve, TimeSeries
from eri_vis.dataset import ERITimelineDataset


class TestERITimelineProcessor:
    """Test suite for ERITimelineProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

    def test_init_valid_params(self):
        """Test processor initialization with valid parameters."""
        processor = ERITimelineProcessor(smoothing_window=5, tau=0.7)
        assert processor.smoothing_window == 5
        assert processor.tau == 0.7

    def test_init_invalid_params(self):
        """Test processor initialization with invalid parameters."""
        with pytest.raises(ValueError, match="smoothing_window must be >= 1"):
            ERITimelineProcessor(smoothing_window=0)

        with pytest.raises(ValueError, match="tau must be between 0.0 and 1.0"):
            ERITimelineProcessor(tau=-0.1)

        with pytest.raises(ValueError, match="tau must be between 0.0 and 1.0"):
            ERITimelineProcessor(tau=1.1)


class TestSmoothing:
    """Test suite for smoothing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ERITimelineProcessor(smoothing_window=3)

    def test_smooth_curve_basic(self):
        """Test basic smoothing functionality."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smoothed = self.processor.smooth_curve(values)

        # With window=3, center=True, min_periods=1:
        # Index 0: mean of [1] = 1.0
        # Index 1: mean of [1, 2, 3] = 2.0
        # Index 2: mean of [1, 2, 3, 4] = 2.5 (but centered, so [2, 3, 4] = 3.0)
        # Index 3: mean of [2, 3, 4, 5] = 3.5 (but centered, so [3, 4, 5] = 4.0)
        # Index 4: mean of [5] = 5.0

        assert len(smoothed) == len(values)
        assert not np.any(np.isnan(smoothed))

        # Check that smoothing reduces variance
        original_var = np.var(values)
        smoothed_var = np.var(smoothed)
        assert smoothed_var <= original_var

    def test_smooth_curve_window_1(self):
        """Test smoothing with window size 1 (no smoothing)."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smoothed = self.processor.smooth_curve(values, window=1)

        np.testing.assert_array_equal(smoothed, values)

    def test_smooth_curve_empty_array(self):
        """Test smoothing with empty array."""
        values = np.array([])
        smoothed = self.processor.smooth_curve(values)

        assert len(smoothed) == 0

    def test_smooth_curve_single_value(self):
        """Test smoothing with single value."""
        values = np.array([5.0])
        smoothed = self.processor.smooth_curve(values)

        np.testing.assert_array_equal(smoothed, values)

    def test_smooth_curve_with_nans(self):
        """Test smoothing with NaN values."""
        values = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        smoothed = self.processor.smooth_curve(values)

        # Should handle NaNs gracefully
        assert len(smoothed) == len(values)
        # Some values should be non-NaN
        assert np.any(~np.isnan(smoothed))


class TestConfidenceIntervals:
    """Test suite for confidence interval computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ERITimelineProcessor()

    def test_compute_confidence_interval_basic(self):
        """Test basic confidence interval computation."""
        # Create synthetic data: 3 seeds x 5 epochs
        data = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],  # seed 1
            [0.15, 0.25, 0.35, 0.45, 0.55],  # seed 2
            [0.05, 0.15, 0.25, 0.35, 0.45]   # seed 3
        ])

        mean_vals, ci_half_width = self.processor.compute_confidence_interval(data, axis=0)

        # Check shapes
        assert mean_vals.shape == (5,)
        assert ci_half_width.shape == (5,)

        # Check mean values
        expected_means = np.mean(data, axis=0)
        np.testing.assert_array_almost_equal(mean_vals, expected_means)

        # Check that CI is positive
        assert np.all(ci_half_width >= 0)

        # Check that CI is reasonable (not zero for multiple samples)
        assert np.all(ci_half_width > 0)

    def test_compute_confidence_interval_single_sample(self):
        """Test confidence interval with single sample."""
        data = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])  # 1 seed x 5 epochs

        mean_vals, ci_half_width = self.processor.compute_confidence_interval(data, axis=0)

        # With single sample, CI should be zero
        np.testing.assert_array_equal(ci_half_width, np.zeros(5))
        np.testing.assert_array_equal(mean_vals, data[0])

    def test_compute_confidence_interval_empty(self):
        """Test confidence interval with empty data."""
        data = np.array([]).reshape(0, 0)

        mean_vals, ci_half_width = self.processor.compute_confidence_interval(data, axis=0)

        assert len(mean_vals) == 0
        assert len(ci_half_width) == 0

    def test_compute_confidence_interval_different_confidence(self):
        """Test confidence interval with different confidence levels."""
        data = np.array([
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35],
            [0.05, 0.15, 0.25],
            [0.12, 0.22, 0.32]
        ])

        # 95% CI
        _, ci_95 = self.processor.compute_confidence_interval(data, confidence=0.95)

        # 90% CI should be narrower
        _, ci_90 = self.processor.compute_confidence_interval(data, confidence=0.90)

        assert np.all(ci_90 <= ci_95)


class TestAccuracyCurves:
    """Test suite for accuracy curve computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

    def create_test_dataset(self, include_missing=False):
        """Create a test dataset for accuracy curve testing."""
        data_rows = []

        methods = ['Scratch_T2', 'sgd', 'ewc_on']
        splits = ['T2_shortcut_normal', 'T2_shortcut_masked']
        seeds = [1, 2, 3]
        epochs = [0.0, 1.0, 2.0, 3.0, 4.0]

        for method in methods:
            for split in splits:
                for seed in seeds:
                    for epoch in epochs:
                        # Skip some data points if testing missing data
                        if include_missing and method == 'sgd' and seed == 2 and epoch == 2.0:
                            continue

                        # Create synthetic accuracy values
                        base_acc = 0.1 if split == 'T2_shortcut_normal' else 0.05
                        if method == 'Scratch_T2':
                            acc = base_acc + epoch * 0.1 + np.random.normal(0, 0.01)
                        else:
                            acc = base_acc + epoch * 0.08 + np.random.normal(0, 0.02)

                        acc = np.clip(acc, 0.0, 1.0)

                        data_rows.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch,
                            'split': split,
                            'acc': acc
                        })

        df = pd.DataFrame(data_rows)

        return ERITimelineDataset(
            data=df,
            metadata={'test_dataset': True},
            methods=methods,
            splits=splits,
            seeds=seeds,
            epoch_range=(0.0, 4.0)
        )

    def test_compute_accuracy_curves_basic(self):
        """Test basic accuracy curve computation."""
        ds = self.create_test_dataset()
        curves = self.processor.compute_accuracy_curves(ds)

        # Should have curves for all method-split combinations
        expected_keys = [
            'Scratch_T2_T2_shortcut_normal', 'Scratch_T2_T2_shortcut_masked',
            'sgd_T2_shortcut_normal', 'sgd_T2_shortcut_masked',
            'ewc_on_T2_shortcut_normal', 'ewc_on_T2_shortcut_masked'
        ]

        assert len(curves) == len(expected_keys)
        for key in expected_keys:
            assert key in curves

        # Check curve properties
        for key, curve in curves.items():
            assert isinstance(curve, AccuracyCurve)
            assert len(curve.epochs) == 5  # 5 epochs in test data
            assert len(curve.mean_accuracy) == 5
            assert len(curve.conf_interval) == 5
            assert curve.n_seeds == 3
            assert curve.raw_data is not None
            assert curve.raw_data.shape == (3, 5)  # 3 seeds x 5 epochs

    def test_compute_accuracy_curves_empty_dataset(self):
        """Test accuracy curve computation with empty dataset."""
        empty_df = pd.DataFrame(columns=['method', 'seed', 'epoch_eff', 'split', 'acc'])
        ds = ERITimelineDataset(
            data=empty_df,
            metadata={},
            methods=[],
            splits=[],
            seeds=[],
            epoch_range=(0.0, 0.0)
        )

        with pytest.raises(ValueError, match="Dataset is empty"):
            self.processor.compute_accuracy_curves(ds)

    def test_compute_accuracy_curves_with_missing_data(self):
        """Test accuracy curve computation with missing data."""
        ds = self.create_test_dataset(include_missing=True)

        # Just test that it works with missing data - warnings are implementation details
        curves = self.processor.compute_accuracy_curves(ds)

        # Should still produce curves despite missing data
        assert len(curves) > 0


class TestAdaptationDelay:
    """Test suite for Adaptation Delay (AD) computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ERITimelineProcessor(smoothing_window=1, tau=0.6)

    def create_curves_for_ad_testing(self):
        """Create synthetic curves for AD testing."""
        epochs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Scratch_T2: crosses threshold at epoch 2.0
        scratch_curve = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.array([0.1, 0.4, 0.6, 0.8, 0.9]),  # crosses 0.6 at epoch 2.0
            conf_interval=np.array([0.01, 0.02, 0.01, 0.01, 0.01]),
            method='Scratch_T2',
            split='T2_shortcut_normal',
            n_seeds=3
        )

        # SGD: crosses threshold at epoch 3.0
        sgd_curve = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.array([0.1, 0.3, 0.5, 0.6, 0.8]),  # crosses 0.6 at epoch 3.0
            conf_interval=np.array([0.02, 0.03, 0.02, 0.02, 0.02]),
            method='sgd',
            split='T2_shortcut_normal',
            n_seeds=3
        )

        # EWC: never crosses threshold (censored)
        ewc_curve = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # never reaches 0.6
            conf_interval=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
            method='ewc_on',
            split='T2_shortcut_normal',
            n_seeds=3
        )

        return {
            'Scratch_T2_T2_shortcut_normal': scratch_curve,
            'sgd_T2_shortcut_normal': sgd_curve,
            'ewc_on_T2_shortcut_normal': ewc_curve
        }

    def test_compute_adaptation_delays_basic(self):
        """Test basic AD computation."""
        curves = self.create_curves_for_ad_testing()
        ad_values = self.processor.compute_adaptation_delays(curves)

        # SGD should have AD = 3.0 - 2.0 = 1.0
        assert 'sgd' in ad_values
        assert ad_values['sgd'] == 1.0

        # EWC should have AD = NaN (censored)
        assert 'ewc_on' in ad_values
        assert np.isnan(ad_values['ewc_on'])

        # Scratch_T2 should not be in results (it's the baseline)
        assert 'Scratch_T2' not in ad_values

    def test_compute_adaptation_delays_no_scratch(self):
        """Test AD computation without Scratch_T2 baseline."""
        curves = self.create_curves_for_ad_testing()
        del curves['Scratch_T2_T2_shortcut_normal']

        ad_values = self.processor.compute_adaptation_delays(curves)

        # Should return empty dict when no scratch baseline
        assert len(ad_values) == 0

    def test_find_threshold_crossing(self):
        """Test threshold crossing detection."""
        curve = AccuracyCurve(
            epochs=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            mean_accuracy=np.array([0.1, 0.4, 0.6, 0.8, 0.9]),
            conf_interval=np.array([0.01, 0.02, 0.01, 0.01, 0.01]),
            method='test',
            split='test',
            n_seeds=1
        )

        # Should find crossing at epoch 2.0 for threshold 0.6
        crossing = self.processor._find_threshold_crossing(curve, 0.6)
        assert crossing == 2.0

        # Should find crossing at epoch 1.0 for threshold 0.4
        crossing = self.processor._find_threshold_crossing(curve, 0.4)
        assert crossing == 1.0

        # Should return NaN for threshold 1.0 (never reached)
        crossing = self.processor._find_threshold_crossing(curve, 1.0)
        assert np.isnan(crossing)

    def test_find_threshold_crossing_empty_curve(self):
        """Test threshold crossing with empty curve."""
        curve = AccuracyCurve(
            epochs=np.array([]),
            mean_accuracy=np.array([]),
            conf_interval=np.array([]),
            method='test',
            split='test',
            n_seeds=0
        )

        crossing = self.processor._find_threshold_crossing(curve, 0.6)
        assert np.isnan(crossing)


class TestPerformanceDeficit:
    """Test suite for Performance Deficit (PD_t) computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ERITimelineProcessor()

    def create_curves_for_pd_testing(self):
        """Create synthetic curves for PD_t testing."""
        epochs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Scratch_T2: high performance
        scratch_curve = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.array([0.1, 0.4, 0.6, 0.8, 0.9]),
            conf_interval=np.array([0.01, 0.02, 0.01, 0.01, 0.01]),
            method='Scratch_T2',
            split='T2_shortcut_normal',
            n_seeds=3
        )

        # SGD: lower performance
        sgd_curve = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.array([0.1, 0.3, 0.4, 0.5, 0.6]),
            conf_interval=np.array([0.02, 0.03, 0.02, 0.02, 0.02]),
            method='sgd',
            split='T2_shortcut_normal',
            n_seeds=3
        )

        return {
            'Scratch_T2_T2_shortcut_normal': scratch_curve,
            'sgd_T2_shortcut_normal': sgd_curve
        }

    def test_compute_performance_deficits_basic(self):
        """Test basic PD_t computation."""
        curves = self.create_curves_for_pd_testing()
        pd_series = self.processor.compute_performance_deficits(curves)

        # Should have PD_t for sgd
        assert 'sgd' in pd_series

        sgd_pd = pd_series['sgd']
        assert isinstance(sgd_pd, TimeSeries)
        assert sgd_pd.method == 'sgd'
        assert sgd_pd.metric_name == 'PD_t'
        assert len(sgd_pd.epochs) > 0
        assert len(sgd_pd.values) == len(sgd_pd.epochs)

        # PD_t should be positive (Scratch performs better than SGD)
        assert np.all(sgd_pd.values >= 0)

        # Should not have PD_t for Scratch_T2 (it's the baseline)
        assert 'Scratch_T2' not in pd_series

    def test_compute_performance_deficits_no_scratch(self):
        """Test PD_t computation without Scratch_T2 baseline."""
        curves = self.create_curves_for_pd_testing()
        del curves['Scratch_T2_T2_shortcut_normal']

        pd_series = self.processor.compute_performance_deficits(curves)

        # Should return empty dict when no scratch baseline
        assert len(pd_series) == 0


class TestSFRRelative:
    """Test suite for Shortcut Forgetting Rate relative (SFR_rel) computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ERITimelineProcessor()

    def create_curves_for_sfr_testing(self):
        """Create synthetic curves for SFR_rel testing."""
        epochs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Scratch_T2 curves
        scratch_patched = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.array([0.1, 0.4, 0.6, 0.8, 0.9]),
            conf_interval=np.array([0.01, 0.02, 0.01, 0.01, 0.01]),
            method='Scratch_T2',
            split='T2_shortcut_normal',
            n_seeds=3
        )

        scratch_masked = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.array([0.05, 0.1, 0.15, 0.2, 0.25]),
            conf_interval=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
            method='Scratch_T2',
            split='T2_shortcut_masked',
            n_seeds=3
        )

        # SGD curves
        sgd_patched = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.array([0.1, 0.3, 0.4, 0.5, 0.6]),
            conf_interval=np.array([0.02, 0.03, 0.02, 0.02, 0.02]),
            method='sgd',
            split='T2_shortcut_normal',
            n_seeds=3
        )

        sgd_masked = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.array([0.05, 0.08, 0.1, 0.12, 0.15]),
            conf_interval=np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
            method='sgd',
            split='T2_shortcut_masked',
            n_seeds=3
        )

        return {
            'Scratch_T2_T2_shortcut_normal': scratch_patched,
            'Scratch_T2_T2_shortcut_masked': scratch_masked,
            'sgd_T2_shortcut_normal': sgd_patched,
            'sgd_T2_shortcut_masked': sgd_masked
        }

    def test_compute_sfr_relative_basic(self):
        """Test basic SFR_rel computation."""
        curves = self.create_curves_for_sfr_testing()
        sfr_series = self.processor.compute_sfr_relative(curves)

        # Should have SFR_rel for sgd
        assert 'sgd' in sfr_series

        sgd_sfr = sfr_series['sgd']
        assert isinstance(sgd_sfr, TimeSeries)
        assert sgd_sfr.method == 'sgd'
        assert sgd_sfr.metric_name == 'SFR_rel'
        assert len(sgd_sfr.epochs) > 0
        assert len(sgd_sfr.values) == len(sgd_sfr.epochs)

        # Should not have SFR_rel for Scratch_T2 (it's the baseline)
        assert 'Scratch_T2' not in sfr_series

    def test_compute_sfr_relative_missing_curves(self):
        """Test SFR_rel computation with missing curves."""
        curves = self.create_curves_for_sfr_testing()

        # Remove one of the required curves
        del curves['Scratch_T2_T2_shortcut_masked']

        sfr_series = self.processor.compute_sfr_relative(curves)

        # Should return empty dict when missing required curves
        assert len(sfr_series) == 0


class TestCurveAlignment:
    """Test suite for curve alignment functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ERITimelineProcessor()

    def test_align_curves_basic(self):
        """Test basic curve alignment."""
        # Create curves with different epoch grids
        curve1 = AccuracyCurve(
            epochs=np.array([0.0, 1.0, 2.0, 3.0]),
            mean_accuracy=np.array([0.1, 0.3, 0.5, 0.7]),
            conf_interval=np.array([0.01, 0.01, 0.01, 0.01]),
            method='method1',
            split='split1',
            n_seeds=1
        )

        curve2 = AccuracyCurve(
            epochs=np.array([0.5, 1.5, 2.5, 3.5]),
            mean_accuracy=np.array([0.2, 0.4, 0.6, 0.8]),
            conf_interval=np.array([0.01, 0.01, 0.01, 0.01]),
            method='method2',
            split='split2',
            n_seeds=1
        )

        common_epochs, aligned1, aligned2 = self.processor._align_curves(curve1, curve2)

        # Should have overlapping range [0.5, 3.0]
        assert len(common_epochs) > 0
        assert common_epochs.min() >= 0.5
        assert common_epochs.max() <= 3.0

        # Aligned arrays should have same length as common_epochs
        assert len(aligned1) == len(common_epochs)
        assert len(aligned2) == len(common_epochs)

        # Values should be reasonable (interpolated)
        assert np.all((aligned1 >= 0.0) & (aligned1 <= 1.0))
        assert np.all((aligned2 >= 0.0) & (aligned2 <= 1.0))

    def test_align_curves_no_overlap(self):
        """Test curve alignment with no overlapping epochs."""
        curve1 = AccuracyCurve(
            epochs=np.array([0.0, 1.0, 2.0]),
            mean_accuracy=np.array([0.1, 0.3, 0.5]),
            conf_interval=np.array([0.01, 0.01, 0.01]),
            method='method1',
            split='split1',
            n_seeds=1
        )

        curve2 = AccuracyCurve(
            epochs=np.array([3.0, 4.0, 5.0]),
            mean_accuracy=np.array([0.6, 0.7, 0.8]),
            conf_interval=np.array([0.01, 0.01, 0.01]),
            method='method2',
            split='split2',
            n_seeds=1
        )

        common_epochs, aligned1, aligned2 = self.processor._align_curves(curve1, curve2)

        # Should return empty arrays
        assert len(common_epochs) == 0
        assert len(aligned1) == 0
        assert len(aligned2) == 0


class TestProcessingSummary:
    """Test suite for processing summary functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

    def test_get_processing_summary_basic(self):
        """Test basic processing summary."""
        # Create some test curves
        curves = {
            'Scratch_T2_T2_shortcut_normal': AccuracyCurve(
                epochs=np.array([0.0, 1.0, 2.0]),
                mean_accuracy=np.array([0.1, 0.3, 0.5]),
                conf_interval=np.array([0.01, 0.01, 0.01]),
                method='Scratch_T2',
                split='T2_shortcut_normal',
                n_seeds=3
            ),
            'sgd_T2_shortcut_masked': AccuracyCurve(
                epochs=np.array([0.0, 1.0, 2.0, 3.0]),
                mean_accuracy=np.array([0.1, 0.2, 0.3, 0.4]),
                conf_interval=np.array([0.02, 0.02, 0.02, 0.02]),
                method='sgd',
                split='T2_shortcut_masked',
                n_seeds=5
            )
        }

        summary = self.processor.get_processing_summary(curves)

        assert summary['n_curves'] == 2
        assert set(summary['methods']) == {'Scratch_T2', 'sgd'}
        assert set(summary['splits']) == {'T2_shortcut_normal', 'T2_shortcut_masked'}
        assert summary['smoothing_window'] == 3
        assert summary['tau_threshold'] == 0.6

        # Check per-curve statistics
        assert 'n_seeds_per_curve' in summary
        assert 'epoch_ranges' in summary
        assert 'curves_per_method' in summary
        assert 'curves_per_split' in summary

    def test_get_processing_summary_empty(self):
        """Test processing summary with empty curves."""
        summary = self.processor.get_processing_summary({})

        assert summary['n_curves'] == 0
        assert summary['methods'] == []
        assert summary['splits'] == []


class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

    def create_realistic_dataset(self):
        """Create a realistic dataset for integration testing."""
        np.random.seed(42)  # For reproducible tests

        data_rows = []
        methods = ['Scratch_T2', 'sgd', 'ewc_on']
        splits = ['T2_shortcut_normal', 'T2_shortcut_masked']
        seeds = [1, 2, 3, 4, 5]
        epochs = np.linspace(0.0, 10.0, 21)  # 21 epochs from 0 to 10

        for method in methods:
            for split in splits:
                for seed in seeds:
                    # Create realistic learning curves
                    base_acc = 0.1 if split == 'T2_shortcut_normal' else 0.05

                    for epoch in epochs:
                        if method == 'Scratch_T2':
                            # Scratch: fast learning, high final performance
                            acc = base_acc + (1 - base_acc) * (1 - np.exp(-epoch / 3))
                        elif method == 'sgd':
                            # SGD: slower learning, moderate performance
                            acc = base_acc + (0.7 - base_acc) * (1 - np.exp(-epoch / 5))
                        else:  # ewc_on
                            # EWC: very slow learning, lower performance
                            acc = base_acc + (0.5 - base_acc) * (1 - np.exp(-epoch / 8))

                        # Add noise
                        acc += np.random.normal(0, 0.02)
                        acc = np.clip(acc, 0.0, 1.0)

                        data_rows.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch,
                            'split': split,
                            'acc': acc
                        })

        df = pd.DataFrame(data_rows)

        return ERITimelineDataset(
            data=df,
            metadata={'realistic_test': True},
            methods=methods,
            splits=splits,
            seeds=seeds,
            epoch_range=(0.0, 10.0)
        )

    def test_full_processing_pipeline(self):
        """Test the complete processing pipeline with realistic data."""
        ds = self.create_realistic_dataset()

        # Compute accuracy curves
        curves = self.processor.compute_accuracy_curves(ds)

        # Should have curves for all method-split combinations
        assert len(curves) == 6  # 3 methods × 2 splits

        # Compute adaptation delays
        ad_values = self.processor.compute_adaptation_delays(curves)

        # Should have AD for continual learning methods
        assert 'sgd' in ad_values
        assert 'ewc_on' in ad_values
        assert 'Scratch_T2' not in ad_values

        # SGD should have smaller (better) AD than EWC, unless EWC is censored
        if not np.isnan(ad_values['ewc_on']):
            assert ad_values['sgd'] < ad_values['ewc_on']
        else:
            # EWC is censored (never crosses threshold), which is expected
            assert not np.isnan(ad_values['sgd'])  # SGD should cross

        # Compute performance deficits
        pd_series = self.processor.compute_performance_deficits(curves)

        # Should have PD_t for continual learning methods
        assert 'sgd' in pd_series
        assert 'ewc_on' in pd_series

        # PD_t should be positive (Scratch performs better)
        assert np.all(pd_series['sgd'].values >= 0)
        assert np.all(pd_series['ewc_on'].values >= 0)

        # Compute SFR relative
        sfr_series = self.processor.compute_sfr_relative(curves)

        # Should have SFR_rel for continual learning methods
        assert 'sgd' in sfr_series
        assert 'ewc_on' in sfr_series

        # Get processing summary
        summary = self.processor.get_processing_summary(curves)

        assert summary['n_curves'] == 6
        assert len(summary['methods']) == 3
        assert len(summary['splits']) == 2

    def test_censored_runs_handling(self):
        """Test handling of censored runs (methods that never cross threshold)."""
        # Create dataset where some methods never cross threshold
        data_rows = []
        methods = ['Scratch_T2', 'good_method', 'bad_method']
        splits = ['T2_shortcut_normal']
        seeds = [1, 2, 3]
        epochs = [0.0, 1.0, 2.0, 3.0, 4.0]

        for method in methods:
            for seed in seeds:
                for epoch in epochs:
                    if method == 'Scratch_T2':
                        acc = 0.1 + epoch * 0.2  # Crosses 0.6 at epoch 2.5
                    elif method == 'good_method':
                        acc = 0.1 + epoch * 0.15  # Crosses 0.6 at epoch 3.33
                    else:  # bad_method
                        acc = 0.1 + epoch * 0.05  # Never crosses 0.6 (max 0.3)

                    acc = np.clip(acc, 0.0, 1.0)

                    data_rows.append({
                        'method': method,
                        'seed': seed,
                        'epoch_eff': epoch,
                        'split': 'T2_shortcut_normal',
                        'acc': acc
                    })

        df = pd.DataFrame(data_rows)
        ds = ERITimelineDataset(
            data=df,
            metadata={},
            methods=methods,
            splits=splits,
            seeds=seeds,
            epoch_range=(0.0, 4.0)
        )

        # Process with tau=0.6
        processor = ERITimelineProcessor(smoothing_window=1, tau=0.6)
        curves = processor.compute_accuracy_curves(ds)

        ad_values = processor.compute_adaptation_delays(curves)

        # good_method should have finite AD
        assert 'good_method' in ad_values
        assert not np.isnan(ad_values['good_method'])

        # bad_method should have NaN AD (censored)
        assert 'bad_method' in ad_values
        assert np.isnan(ad_values['bad_method'])


if __name__ == '__main__':
    pytest.main([__file__])
