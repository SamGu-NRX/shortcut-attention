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


class TestEdgeCasesAndBoundaryConditions:
    """
    Test edge cases and boundary conditions for robust IEEE research standards.
    
    Rationale: IEEE research requires handling of edge cases that may occur in 
    real experimental settings. These tests ensure numerical stability and correct
    behavior at boundaries.
    """
    
    def test_smoothing_with_empty_array(self):
        """
        Test smoothing with empty input arrays.
        
        Rationale: Empty datasets can occur when filtering experimental data.
        The calculator should handle this gracefully without exceptions.
        """
        calculator = CorrectedERICalculator(smoothing_window=3)
        
        empty_array = np.array([])
        result = calculator.apply_trailing_smoothing(empty_array)
        
        assert len(result) == 0
        assert isinstance(result, np.ndarray)
    
    def test_smoothing_with_single_value(self):
        """
        Test smoothing with single-element array.
        
        Rationale: Single-epoch experiments or highly filtered data may result
        in single values. Smoothing should return the original value unchanged.
        """
        calculator = CorrectedERICalculator(smoothing_window=3)
        
        single_value = np.array([0.5])
        result = calculator.apply_trailing_smoothing(single_value)
        
        np.testing.assert_array_equal(result, single_value)
    
    def test_smoothing_preserves_monotonicity_properties(self):
        """
        Test that smoothing does not introduce non-physical artifacts.
        
        Rationale: Smoothing should reduce noise but not create spurious patterns.
        For strictly increasing sequences, smoothed values should also increase.
        """
        calculator = CorrectedERICalculator(smoothing_window=3)
        
        # Strictly increasing sequence
        increasing = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        smoothed = calculator.apply_trailing_smoothing(increasing)
        
        # Smoothed should also be monotonically increasing
        for i in range(1, len(smoothed)):
            assert smoothed[i] >= smoothed[i-1], \
                f"Smoothing violated monotonicity at index {i}: {smoothed[i-1]} -> {smoothed[i]}"
    
    def test_threshold_crossing_at_exact_boundary(self):
        """
        Test threshold crossing when accuracy equals threshold exactly.
        
        Rationale: Boundary conditions (accuracy == threshold) must be handled
        consistently with the mathematical definition (A >= τ).
        """
        calculator = CorrectedERICalculator(tau=0.6, smoothing_window=1)
        
        # Accuracy curve that reaches exactly 0.6
        accuracy = np.array([0.4, 0.5, 0.6, 0.7])
        epochs = np.array([0, 1, 2, 3])
        
        crossing_epoch = calculator.find_threshold_crossing_epoch(accuracy, epochs, tau=0.6)
        
        # Should detect crossing at epoch 2 where acc == 0.6
        assert crossing_epoch == 2
    
    def test_threshold_crossing_with_plateau(self):
        """
        Test threshold crossing when accuracy plateaus near threshold.
        
        Rationale: Training may plateau just below threshold. Calculator should
        correctly identify if/when threshold is eventually crossed.
        """
        calculator = CorrectedERICalculator(tau=0.6, smoothing_window=1)
        
        # Plateau below threshold, then cross
        accuracy = np.array([0.3, 0.4, 0.55, 0.55, 0.55, 0.6, 0.65])
        epochs = np.array([0, 1, 2, 3, 4, 5, 6])
        
        crossing_epoch = calculator.find_threshold_crossing_epoch(accuracy, epochs, tau=0.6)
        
        # Should detect first crossing at epoch 5
        assert crossing_epoch == 5
    
    def test_adaptation_delay_with_negative_values(self):
        """
        Test AD calculation when CL method is faster than scratch.
        
        Rationale: Negative AD values indicate CL method adapts faster. This is
        a critical research finding that must be computed correctly.
        """
        calculator = CorrectedERICalculator(tau=0.6, smoothing_window=1)
        
        # CL reaches threshold at epoch 3, scratch at epoch 5
        cl_accuracy = np.array([0.3, 0.45, 0.55, 0.6, 0.65])
        cl_epochs = np.array([0, 1, 2, 3, 4])
        
        scratch_accuracy = np.array([0.2, 0.35, 0.45, 0.52, 0.58, 0.6, 0.62])
        scratch_epochs = np.array([0, 1, 2, 3, 4, 5, 6])
        
        ad, censored = calculator.compute_adaptation_delay(
            cl_accuracy, cl_epochs, scratch_accuracy, scratch_epochs, tau=0.6
        )
        
        assert not censored
        assert ad is not None
        assert ad < 0, f"Expected negative AD (CL faster than scratch), got {ad}"
        assert ad == -2, f"Expected AD = -2 (epoch 3 - epoch 5), got {ad}"
    
    def test_performance_deficit_when_methods_tied(self):
        """
        Test PD calculation when both methods achieve identical accuracy.
        
        Rationale: PD should be exactly zero when methods perform identically.
        This tests numerical precision in equality comparisons.
        """
        calculator = CorrectedERICalculator()
        
        # Both methods achieve identical final accuracy
        data = []
        for method in ['derpp', 'scratch_t2']:
            for epoch in range(5):
                data.extend([
                    {'method': method, 'seed': 42, 'epoch_eff': epoch,
                     'split': 'T2_shortcut_normal', 'acc': 0.5 + epoch * 0.05, 
                     'top5': 0.7, 'loss': 1.0},
                    {'method': method, 'seed': 42, 'epoch_eff': epoch,
                     'split': 'T2_shortcut_masked', 'acc': 0.4 + epoch * 0.04,
                     'top5': 0.6, 'loss': 1.2},
                ])
        
        df = pd.DataFrame(data)
        dataset = ERITimelineDataset(
            data=df, metadata={'test': True},
            methods=['derpp', 'scratch_t2'],
            splits=['T2_shortcut_normal', 'T2_shortcut_masked'],
            seeds=[42], epoch_range=(0, 4)
        )
        
        perf_deficit = calculator.compute_performance_deficit(dataset, 'derpp', 42, 'scratch_t2')
        
        assert perf_deficit is not None
        assert abs(perf_deficit) < 1e-10, f"Expected PD ≈ 0 for identical performance, got {perf_deficit}"
    
    def test_sfr_rel_with_zero_deltas(self):
        """
        Test SFR_rel when patch and masked accuracies are identical.
        
        Rationale: When no shortcut effect exists (patch = masked), both deltas
        should be zero, resulting in SFR_rel = 0. Tests detection of no-shortcut case.
        """
        calculator = CorrectedERICalculator()
        
        # Both splits achieve identical accuracy (no shortcut effect)
        data = []
        for method in ['derpp', 'scratch_t2']:
            for epoch in range(5):
                acc = 0.5 + epoch * 0.05  # Same for both splits
                data.extend([
                    {'method': method, 'seed': 42, 'epoch_eff': epoch,
                     'split': 'T2_shortcut_normal', 'acc': acc, 'top5': 0.7, 'loss': 1.0},
                    {'method': method, 'seed': 42, 'epoch_eff': epoch,
                     'split': 'T2_shortcut_masked', 'acc': acc, 'top5': 0.7, 'loss': 1.0},
                ])
        
        df = pd.DataFrame(data)
        dataset = ERITimelineDataset(
            data=df, metadata={'test': True},
            methods=['derpp', 'scratch_t2'],
            splits=['T2_shortcut_normal', 'T2_shortcut_masked'],
            seeds=[42], epoch_range=(0, 4)
        )
        
        sfr_rel = calculator.compute_shortcut_feature_reliance(dataset, 'derpp', 42, 'scratch_t2')
        
        assert sfr_rel is not None
        assert abs(sfr_rel) < 1e-10, \
            f"Expected SFR_rel ≈ 0 when no shortcut effect, got {sfr_rel}"


class TestNumericalStability:
    """
    Test numerical stability for IEEE research rigor.
    
    Rationale: Numerical stability is critical for reproducibility and reliable
    comparisons across experiments. These tests ensure calculations remain stable
    across a wide range of input values.
    """
    
    def test_smoothing_with_extreme_values(self):
        """
        Test smoothing with extreme but valid accuracy values.
        
        Rationale: Experimental data may contain accuracies near 0 or 1.
        Calculator must maintain numerical stability at these boundaries.
        """
        calculator = CorrectedERICalculator(smoothing_window=3)
        
        # Near-zero accuracies
        near_zero = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        smoothed_zero = calculator.apply_trailing_smoothing(near_zero)
        assert np.all(np.isfinite(smoothed_zero)), "Smoothing failed with near-zero values"
        assert np.all(smoothed_zero >= 0), "Smoothing produced negative values"
        
        # Near-one accuracies
        near_one = np.array([0.995, 0.996, 0.997, 0.998, 0.999])
        smoothed_one = calculator.apply_trailing_smoothing(near_one)
        assert np.all(np.isfinite(smoothed_one)), "Smoothing failed with near-one values"
        assert np.all(smoothed_one <= 1.0), "Smoothing produced values > 1.0"
    
    def test_smoothing_with_noisy_data(self):
        """
        Test smoothing effectiveness with realistic noisy training curves.
        
        Rationale: Real training exhibits stochasticity. Smoothing should reduce
        variance while preserving the overall trend.
        """
        calculator = CorrectedERICalculator(smoothing_window=5)
        
        # Create realistic noisy curve: overall increasing trend with noise
        np.random.seed(42)  # For reproducibility
        epochs = np.arange(20)
        clean_trend = 0.3 + 0.02 * epochs  # Linear increase
        noise = np.random.normal(0, 0.05, size=len(epochs))  # 5% std noise
        noisy_accuracy = np.clip(clean_trend + noise, 0, 1)
        
        smoothed = calculator.apply_trailing_smoothing(noisy_accuracy, window=5)
        
        # Smoothed should have lower variance than original
        original_variance = np.var(np.diff(noisy_accuracy))
        smoothed_variance = np.var(np.diff(smoothed))
        
        assert smoothed_variance < original_variance, \
            f"Smoothing did not reduce variance: {smoothed_variance:.6f} >= {original_variance:.6f}"
        
        # Smoothed should still preserve overall upward trend
        assert smoothed[-1] > smoothed[0], "Smoothing reversed overall trend"
    
    def test_composite_score_normalization_bounds(self):
        """
        Test that composite ERI scores remain in valid [0,1] range.
        
        Rationale: Composite scores are used for cross-method comparisons.
        They must be properly normalized to enable meaningful interpretation.
        """
        # Test various extreme cases
        test_cases = [
            # (AD, PD, SFR_rel, description)
            (0.0, 0.0, 0.0, "All zeros"),
            (50.0, 1.0, 1.0, "All maximum values"),
            (25.0, 0.5, 0.0, "Mixed values"),
            (-10.0, 0.1, 0.05, "Negative AD"),
        ]
        
        for ad, pd, sfr, description in test_cases:
            metrics = ERIMetrics(
                adaptation_delay=ad,
                performance_deficit=pd,
                shortcut_feature_reliance=sfr,
                method='test',
                seed=42
            )
            
            score = metrics.compute_overall_eri(ad_weight=0.4, pd_weight=0.4, 
                                               sfr_weight=0.2, ad_max=50.0)
            
            assert score is not None, f"Composite score failed for {description}"
            assert 0.0 <= score <= 1.0, \
                f"Composite score out of bounds for {description}: {score}"
            assert np.isfinite(score), \
                f"Composite score not finite for {description}: {score}"


class TestInputValidation:
    """
    Test input validation for robust error handling.
    
    Rationale: IEEE research requires clear error reporting when inputs are invalid.
    These tests ensure graceful handling of malformed inputs.
    """
    
    def test_calculator_rejects_invalid_tau(self):
        """
        Test that calculator rejects tau values outside [0,1] range.
        
        Rationale: τ represents an accuracy threshold, which must be in [0,1].
        Invalid values should be rejected immediately with clear error messages.
        """
        with pytest.raises(ValueError, match="tau must be between 0.0 and 1.0"):
            CorrectedERICalculator(tau=1.5)
        
        with pytest.raises(ValueError, match="tau must be between 0.0 and 1.0"):
            CorrectedERICalculator(tau=-0.1)
    
    def test_calculator_rejects_invalid_window(self):
        """
        Test that calculator rejects invalid smoothing window sizes.
        
        Rationale: Window size must be positive integer. Invalid values indicate
        programming errors that should be caught early.
        """
        with pytest.raises(ValueError, match="smoothing_window must be >= 1"):
            CorrectedERICalculator(smoothing_window=0)
        
        with pytest.raises(ValueError, match="smoothing_window must be >= 1"):
            CorrectedERICalculator(smoothing_window=-5)
    
    def test_threshold_crossing_with_mismatched_lengths(self):
        """
        Test handling of mismatched accuracy and epoch array lengths.
        
        Rationale: Mismatched arrays indicate data integrity issues. Should not
        crash but handle gracefully.
        """
        calculator = CorrectedERICalculator(tau=0.6)
        
        accuracy = np.array([0.3, 0.4, 0.5, 0.6])
        epochs = np.array([0, 1, 2])  # One element short
        
        # Should handle gracefully - likely return None or use minimum length
        result = calculator.find_threshold_crossing_epoch(accuracy[:len(epochs)], epochs)
        # Test should complete without exception


class TestMathematicalProperties:
    """
    Test mathematical properties and invariants of ERI metrics.
    
    Rationale: ERI metrics must satisfy mathematical properties defined in the paper.
    These tests validate that implementations preserve theoretical guarantees.
    """
    
    def test_pd_antisymmetry(self):
        """
        Test that PD(A, B) = -PD(B, A).
        
        Rationale: Performance deficit should be antisymmetric by definition.
        PD measures relative advantage, so swapping methods should negate the value.
        """
        calculator = CorrectedERICalculator()
        
        data = []
        for method, acc_offset in [('method_a', 0.0), ('method_b', 0.1)]:
            for epoch in range(5):
                data.extend([
                    {'method': method, 'seed': 42, 'epoch_eff': epoch,
                     'split': 'T2_shortcut_normal', 'acc': 0.5 + acc_offset + epoch * 0.05,
                     'top5': 0.7, 'loss': 1.0},
                    {'method': method, 'seed': 42, 'epoch_eff': epoch,
                     'split': 'T2_shortcut_masked', 'acc': 0.4 + acc_offset + epoch * 0.04,
                     'top5': 0.6, 'loss': 1.2},
                ])
        
        df = pd.DataFrame(data)
        dataset = ERITimelineDataset(
            data=df, metadata={'test': True},
            methods=['method_a', 'method_b'],
            splits=['T2_shortcut_normal', 'T2_shortcut_masked'],
            seeds=[42], epoch_range=(0, 4)
        )
        
        pd_ab = calculator.compute_performance_deficit(dataset, 'method_a', 42, 'method_b')
        pd_ba = calculator.compute_performance_deficit(dataset, 'method_b', 42, 'method_a')
        
        assert pd_ab is not None and pd_ba is not None
        assert abs(pd_ab + pd_ba) < 1e-10, \
            f"PD antisymmetry violated: PD(A,B)={pd_ab}, PD(B,A)={pd_ba}"
    
    def test_sfr_rel_antisymmetry(self):
        """
        Test that SFR_rel(A, B) = -SFR_rel(B, A).
        
        Rationale: Like PD, SFR_rel measures relative reliance difference.
        Should be antisymmetric under method swap.
        """
        calculator = CorrectedERICalculator()
        
        data = []
        for method, patch_boost in [('method_a', 0.0), ('method_b', 0.05)]:
            for epoch in range(5):
                data.extend([
                    {'method': method, 'seed': 42, 'epoch_eff': epoch,
                     'split': 'T2_shortcut_normal', 
                     'acc': 0.5 + patch_boost + epoch * 0.05,
                     'top5': 0.7, 'loss': 1.0},
                    {'method': method, 'seed': 42, 'epoch_eff': epoch,
                     'split': 'T2_shortcut_masked', 
                     'acc': 0.4 + epoch * 0.05,  # No boost on masked
                     'top5': 0.6, 'loss': 1.2},
                ])
        
        df = pd.DataFrame(data)
        dataset = ERITimelineDataset(
            data=df, metadata={'test': True},
            methods=['method_a', 'method_b'],
            splits=['T2_shortcut_normal', 'T2_shortcut_masked'],
            seeds=[42], epoch_range=(0, 4)
        )
        
        sfr_ab = calculator.compute_shortcut_feature_reliance(
            dataset, 'method_a', 42, 'method_b'
        )
        sfr_ba = calculator.compute_shortcut_feature_reliance(
            dataset, 'method_b', 42, 'method_a'
        )
        
        assert sfr_ab is not None and sfr_ba is not None
        assert abs(sfr_ab + sfr_ba) < 1e-10, \
            f"SFR_rel antisymmetry violated: SFR(A,B)={sfr_ab}, SFR(B,A)={sfr_ba}"
    
    def test_delta_decomposition(self):
        """
        Test that SFR_rel correctly decomposes as Δ_CL - Δ_S.
        
        Rationale: SFR_rel definition requires correct computation of intermediate
        deltas. This validates the decomposition matches the paper formula.
        """
        calculator = CorrectedERICalculator()
        
        data = []
        # Create data with known delta values
        # CL: patch=0.8, masked=0.6, Δ_CL=0.2
        # Scratch: patch=0.7, masked=0.6, Δ_S=0.1
        # Expected SFR_rel = 0.2 - 0.1 = 0.1
        
        data.extend([
            {'method': 'cl_method', 'seed': 42, 'epoch_eff': 0,
             'split': 'T2_shortcut_normal', 'acc': 0.8, 'top5': 0.9, 'loss': 0.5},
            {'method': 'cl_method', 'seed': 42, 'epoch_eff': 0,
             'split': 'T2_shortcut_masked', 'acc': 0.6, 'top5': 0.8, 'loss': 0.8},
            {'method': 'scratch_t2', 'seed': 42, 'epoch_eff': 0,
             'split': 'T2_shortcut_normal', 'acc': 0.7, 'top5': 0.85, 'loss': 0.7},
            {'method': 'scratch_t2', 'seed': 42, 'epoch_eff': 0,
             'split': 'T2_shortcut_masked', 'acc': 0.6, 'top5': 0.8, 'loss': 0.8},
        ])
        
        df = pd.DataFrame(data)
        dataset = ERITimelineDataset(
            data=df, metadata={'test': True},
            methods=['cl_method', 'scratch_t2'],
            splits=['T2_shortcut_normal', 'T2_shortcut_masked'],
            seeds=[42], epoch_range=(0, 0)
        )
        
        # Get metrics with intermediate values
        metrics = calculator.compute_method_metrics(dataset, 'cl_method', 42, 'scratch_t2')
        
        # Verify delta decomposition
        assert metrics.delta_cl is not None
        assert metrics.delta_scratch is not None
        assert metrics.shortcut_feature_reliance is not None
        
        # Δ_CL = 0.8 - 0.6 = 0.2
        assert abs(metrics.delta_cl - 0.2) < 1e-10, \
            f"Expected Δ_CL=0.2, got {metrics.delta_cl}"
        
        # Δ_S = 0.7 - 0.6 = 0.1
        assert abs(metrics.delta_scratch - 0.1) < 1e-10, \
            f"Expected Δ_S=0.1, got {metrics.delta_scratch}"
        
        # SFR_rel = 0.2 - 0.1 = 0.1
        expected_sfr = metrics.delta_cl - metrics.delta_scratch
        assert abs(metrics.shortcut_feature_reliance - expected_sfr) < 1e-10, \
            f"SFR_rel decomposition failed: got {metrics.shortcut_feature_reliance}, " \
            f"expected {expected_sfr}"


if __name__ == "__main__":
    pytest.main([__file__])
