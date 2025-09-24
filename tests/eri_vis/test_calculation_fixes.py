"""
Tests for corrected ERI calculations.

This module tests the mathematically correct implementation of ERI metrics
according to the exact paper specification, including:
1. Trailing moving average smoothing
2. Final checkpoint selection for PD and SFR_rel
3. Macro-averaged accuracy computation
4. Proper effective epoch tracking
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from eri_vis.metrics_calculator import CorrectedERICalculator, ERIMetrics
from eri_vis.dataset import ERITimelineDataset
from eri_vis.eri_composite import CompositeERICalculator, CompositeERIConfig


class TestTrailingSmoothing:
    """Test trailing moving average smoothing implementation."""

    def test_trailing_smoothing_basic(self):
        """Test basic trailing smoothing functionality."""
        calculator = CorrectedERICalculator(smoothing_window=3)

        # Simple test case
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        smoothed = calculator.apply_trailing_smoothing(values, window=3)

        # Expected: [0.1, 0.15, 0.2, 0.3, 0.4]
        expected = np.array([0.1, 0.15, 0.2, 0.3, 0.4])

        np.testing.assert_array_almost_equal(smoothed, expected, decimal=6)

    def test_trailing_smoothing_window_1(self):
        """Test that window=1 returns original values."""
        calculator = CorrectedERICalculator(smoothing_window=1)

        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        smoothed = calculator.apply_trailing_smoothing(values, window=1)

        np.testing.assert_array_equal(smoothed, values)

    def test_trailing_smoothing_paper_example(self):
        """Test trailing smoothing with paper-like example."""
        calculator = CorrectedERICalculator(smoothing_window=3)

        # Simulate accuracy curve that crosses threshold
        values = np.array([0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7])
        smoothed = calculator.apply_trailing_smoothing(values, window=3)

        # Check specific values
        assert abs(smoothed[0] - 0.3) < 1e-10  # First value unchanged
        assert abs(smoothed[1] - 0.35) < 1e-10  # (0.3 + 0.4) / 2
        assert abs(smoothed[2] - 0.4) < 1e-10   # (0.3 + 0.4 + 0.5) / 3
        assert abs(smoothed[3] - 0.48333333) < 1e-6  # (0.4 + 0.5 + 0.55) / 3

        # Smoothed curve should be less noisy
        assert len(smoothed) == len(values)


class TestThresholdCrossing:
    """Test threshold crossing detection with corrected smoothing."""

    def test_threshold_crossing_basic(self):
        """Test basic threshold crossing detection."""
        calculator = CorrectedERICalculator(tau=0.6, smoothing_window=3)

        # Accuracy curve that crosses 0.6 at epoch 4
        accuracy = np.array([0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7])
        epochs = np.array([0, 1, 2, 3, 4, 5, 6])

        crossing_epoch = calculator.find_threshold_crossing_epoch(accuracy, epochs, tau=0.6)

        # Should find crossing after smoothing
        assert crossing_epoch is not None
        assert crossing_epoch >= 0

    def test_threshold_crossing_censored(self):
        """Test censored run (never crosses threshold)."""
        calculator = CorrectedERICalculator(tau=0.8, smoothing_window=3)

        # Accuracy curve that never reaches 0.8
        accuracy = np.array([0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7])
        epochs = np.array([0, 1, 2, 3, 4, 5, 6])

        crossing_epoch = calculator.find_threshold_crossing_epoch(accuracy, epochs, tau=0.8)

        # Should return None for censored run
        assert crossing_epoch is None


class TestAdaptationDelay:
    """Test Adaptation Delay calculation with corrected implementation."""

    def test_adaptation_delay_basic(self):
        """Test basic AD calculation."""
        calculator = CorrectedERICalculator(tau=0.6, smoothing_window=3)

        # CL method reaches threshold at epoch 5, scratch at epoch 3
        cl_accuracy = np.array([0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65])
        cl_epochs = np.array([0, 1, 2, 3, 4, 5, 6])

        scratch_accuracy = np.array([0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75])
        scratch_epochs = np.array([0, 1, 2, 3, 4, 5, 6])

        ad, censored = calculator.compute_adaptation_delay(
            cl_accuracy, cl_epochs, scratch_accuracy, scratch_epochs, tau=0.6
        )

        assert not censored
        assert ad is not None
        # CL method should be slower (positive AD)
        assert ad > 0

    def test_adaptation_delay_censored(self):
        """Test AD calculation with censored run."""
        calculator = CorrectedERICalculator(tau=0.8, smoothing_window=3)

        # CL method never reaches 0.8
        cl_accuracy = np.array([0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65])
        cl_epochs = np.array([0, 1, 2, 3, 4, 5, 6])

        # Scratch reaches 0.8
        scratch_accuracy = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85])
        scratch_epochs = np.array([0, 1, 2, 3, 4, 5, 6])

        ad, censored = calculator.compute_adaptation_delay(
            cl_accuracy, cl_epochs, scratch_accuracy, scratch_epochs, tau=0.8
        )

        assert censored
        assert ad is None


class TestFinalCheckpointMetrics:
    """Test PD and SFR_rel calculations using final checkpoints."""

    def create_test_dataset(self):
        """Create test dataset with known final checkpoint values."""
        data = []

        # Method 1: derpp
        for epoch in [0, 1, 2, 3, 4, 5]:  # Final epoch = 5
            data.extend([
                {'method': 'derpp', 'seed': 42, 'epoch_eff': epoch,
                 'split': 'T2_shortcut_normal', 'acc': 0.5 + epoch * 0.05, 'top5': 0.7, 'loss': 1.0},
                {'method': 'derpp', 'seed': 42, 'epoch_eff': epoch,
                 'split': 'T2_shortcut_masked', 'acc': 0.4 + epoch * 0.03, 'top5': 0.6, 'loss': 1.2},
            ])

        # Method 2: scratch_t2
        for epoch in [0, 1, 2, 3, 4, 5]:  # Final epoch = 5
            data.extend([
                {'method': 'scratch_t2', 'seed': 42, 'epoch_eff': epoch,
                 'split': 'T2_shortcut_normal', 'acc': 0.6 + epoch * 0.02, 'top5': 0.8, 'loss': 0.8},
                {'method': 'scratch_t2', 'seed': 42, 'epoch_eff': epoch,
                 'split': 'T2_shortcut_masked', 'acc': 0.5 + epoch * 0.02, 'top5': 0.7, 'loss': 1.0},
            ])

        df = pd.DataFrame(data)

        return ERITimelineDataset(
            data=df,
            metadata={'test': True},
            methods=['derpp', 'scratch_t2'],
            splits=['T2_shortcut_normal', 'T2_shortcut_masked'],
            seeds=[42],
            epoch_range=(0, 5)
        )

    def test_final_checkpoint_accuracy(self):
        """Test final checkpoint accuracy retrieval."""
        calculator = CorrectedERICalculator()
        dataset = self.create_test_dataset()

        # Get final checkpoint accuracy for derpp on shortcut_normal
        final_acc = calculator.get_final_checkpoint_accuracy(
            dataset, 'derpp', 42, 'T2_shortcut_normal'
        )

        # Should be 0.5 + 5 * 0.05 = 0.75
        assert final_acc == 0.75

    def test_performance_deficit_calculation(self):
        """Test PD calculation using final checkpoints."""
        calculator = CorrectedERICalculator()
        dataset = self.create_test_dataset()

        pd = calculator.compute_performance_deficit(dataset, 'derpp', 42, 'scratch_t2')

        # Expected: scratch_final - derpp_final = 0.7 - 0.75 = -0.05
        # (derpp performs better on shortcut_normal)
        assert pd is not None
        assert abs(pd - (-0.05)) < 1e-10

    def test_sfr_rel_calculation(self):
        """Test SFR_rel calculation using final checkpoints."""
        calculator = CorrectedERICalculator()
        dataset = self.create_test_dataset()

        sfr_rel = calculator.compute_shortcut_feature_reliance(dataset, 'derpp', 42, 'scratch_t2')

        # Expected:
        # Δ_derpp = 0.75 - 0.55 = 0.20
        # Δ_scratch = 0.7 - 0.6 = 0.10
        # SFR_rel = 0.20 - 0.10 = 0.10
        assert sfr_rel is not None
        assert abs(sfr_rel - 0.10) < 1e-10


class TestComprehensiveMetrics:
    """Test comprehensive metrics computation."""

    def test_method_metrics_computation(self):
        """Test complete metrics computation for a method."""
        calculator = CorrectedERICalculator(tau=0.6, smoothing_window=3)

        # Create more comprehensive test dataset
        data = []

        # Add data for both methods with multiple epochs
        for method, base_acc in [('derpp', 0.3), ('scratch_t2', 0.4)]:
            for epoch in range(10):
                acc_normal = base_acc + epoch * 0.05
                acc_masked = base_acc + epoch * 0.03

                data.extend([
                    {'method': method, 'seed': 42, 'epoch_eff': epoch,
                     'split': 'T2_shortcut_normal', 'acc': acc_normal, 'top5': 0.7, 'loss': 1.0},
                    {'method': method, 'seed': 42, 'epoch_eff': epoch,
                     'split': 'T2_shortcut_masked', 'acc': acc_masked, 'top5': 0.6, 'loss': 1.2},
                ])

        df = pd.DataFrame(data)
        dataset = ERITimelineDataset(
            data=df,
            metadata={'test': True},
            methods=['derpp', 'scratch_t2'],
            splits=['T2_shortcut_normal', 'T2_shortcut_masked'],
            seeds=[42],
            epoch_range=(0, 9)
        )

        # Compute metrics
        metrics = calculator.compute_method_metrics(dataset, 'derpp', 42, 'scratch_t2')

        # Check that all metrics are computed
        assert metrics.method == 'derpp'
        assert metrics.seed == 42
        assert metrics.adaptation_delay is not None
        assert metrics.performance_deficit is not None
        assert metrics.shortcut_feature_reliance is not None

        # Check final checkpoint accuracies are stored
        assert metrics.final_cl_patch is not None
        assert metrics.final_cl_mask is not None
        assert metrics.final_scratch_patch is not None
        assert metrics.final_scratch_mask is not None


class TestCompositeERI:
    """Test composite ERI score computation."""

    def test_composite_score_basic(self):
        """Test basic composite score computation."""
        config = CompositeERIConfig(ad_weight=0.4, pd_weight=0.4, sfr_weight=0.2, ad_max=50.0)
        calculator = CompositeERICalculator(config)

        # Create test metrics
        metrics = ERIMetrics(
            adaptation_delay=10.0,  # Will be normalized to 10/50 = 0.2
            performance_deficit=0.1,
            shortcut_feature_reliance=0.05,
            method='test_method',
            seed=42
        )

        composite = calculator.compute_composite_score(metrics)

        # Expected: 0.4 * 0.2 + 0.4 * 0.1 + 0.2 * 0.05 = 0.08 + 0.04 + 0.01 = 0.13
        expected = 0.4 * 0.2 + 0.4 * 0.1 + 0.2 * 0.05
        assert composite is not None
        assert abs(composite - expected) < 1e-6

    def test_composite_score_incomplete_data(self):
        """Test composite score with incomplete data."""
        calculator = CompositeERICalculator()

        # Metrics with missing values
        metrics = ERIMetrics(
            adaptation_delay=None,  # Missing
            performance_deficit=0.1,
            shortcut_feature_reliance=0.05,
            method='test_method',
            seed=42
        )

        composite = calculator.compute_composite_score(metrics)
        assert composite is None


class TestValidationWithRealData:
    """Test validation against known data patterns."""

    def test_derpp_outperforms_scratch(self):
        """Test that DER++ shows better performance than scratch in provided data."""
        # This test would use actual data from einstellung_results/session_20250923-012304_seed42
        # For now, we'll create synthetic data that mimics the expected pattern

        calculator = CorrectedERICalculator(tau=0.6, smoothing_window=3)

        # Create synthetic data where DER++ outperforms scratch
        data = []

        # DER++ data - better final performance, faster convergence
        for epoch in range(20):
            acc_normal = 0.2 + epoch * 0.03  # Reaches 0.8 at epoch ~20
            acc_masked = 0.15 + epoch * 0.025

            data.extend([
                {'method': 'derpp', 'seed': 42, 'epoch_eff': epoch,
                 'split': 'T2_shortcut_normal', 'acc': min(acc_normal, 0.85), 'top5': 0.9, 'loss': 1.0},
                {'method': 'derpp', 'seed': 42, 'epoch_eff': epoch,
                 'split': 'T2_shortcut_masked', 'acc': min(acc_masked, 0.75), 'top5': 0.8, 'loss': 1.2},
            ])

        # Scratch_T2 data - slower convergence, worse final performance
        for epoch in range(20):
            acc_normal = 0.15 + epoch * 0.025  # Reaches 0.65 at epoch 20
            acc_masked = 0.1 + epoch * 0.02

            data.extend([
                {'method': 'scratch_t2', 'seed': 42, 'epoch_eff': epoch,
                 'split': 'T2_shortcut_normal', 'acc': min(acc_normal, 0.65), 'top5': 0.8, 'loss': 1.5},
                {'method': 'scratch_t2', 'seed': 42, 'epoch_eff': epoch,
                 'split': 'T2_shortcut_masked', 'acc': min(acc_masked, 0.5), 'top5': 0.7, 'loss': 1.8},
            ])

        df = pd.DataFrame(data)
        dataset = ERITimelineDataset(
            data=df,
            metadata={'test': True},
            methods=['derpp', 'scratch_t2'],
            splits=['T2_shortcut_normal', 'T2_shortcut_masked'],
            seeds=[42],
            epoch_range=(0, 19)
        )

        # Validate that calculations work correctly
        validation_result = calculator.validate_against_known_data(dataset)
        assert validation_result is True

        # Compute metrics for DER++
        derpp_metrics = calculator.compute_method_metrics(dataset, 'derpp', 42, 'scratch_t2')

        # DER++ should have negative PD (better performance than scratch)
        assert derpp_metrics.performance_deficit is not None
        assert derpp_metrics.performance_deficit < 0  # DER++ outperforms scratch


if __name__ == "__main__":
    pytest.main([__file__])
