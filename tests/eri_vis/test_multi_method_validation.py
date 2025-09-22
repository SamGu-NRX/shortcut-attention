"""
Test suite for validating multi-method ERI visualization capabilities.

This test validates that the existing ERI visualization system correctly handles
multi-method datasets including baseline methods (Scratch_T2, Interleaved) and
continual learning methods for comparative analysis.

Tests cover:
- ERITimelineProcessor with multi-method datasets
- compute_performance_deficits() using Scratch_T2 as baseline
- compute_sfr_relative() calculating relative metrics properly
- ERIDynamicsPlotter generating comparative plots with multiple methods
- ERIHeatmapPlotter creating robustness heatmaps across all methods
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

from eri_vis.data_loader import ERIDataLoader
from eri_vis.dataset import ERITimelineDataset
from eri_vis.processing import ERITimelineProcessor, AccuracyCurve, TimeSeries
from eri_vis.plot_dynamics import ERIDynamicsPlotter
from eri_vis.plot_heatmap import ERIHeatmapPlotter


class TestMultiMethodERIValidation:
    """Test suite for multi-method ERI visualization validation."""

    @pytest.fixture
    def multi_method_dataset(self):
        """Create a multi-method dataset with baseline and continual learning methods."""
        # Define methods including baselines
        methods = ['Scratch_T2', 'Interleaved', 'sgd', 'derpp', 'ewc_on']
        splits = ['T2_shortcut_normal', 'T2_shortcut_masked']
        seeds = [42, 43, 44]
        epochs = np.linspace(0, 10, 21)  # 21 epochs from 0 to 10

        rows = []

        for method in methods:
            for split in splits:
                for seed in seeds:
                    for epoch in epochs:
                        # Create realistic accuracy trajectories
                        if method == 'Scratch_T2':
                            # Optimal performance on shortcut task
                            if split == 'T2_shortcut_normal':
                                acc = 0.1 + 0.8 * (1 - np.exp(-epoch/2))  # Fast learning to ~0.9
                            else:  # maske
                               acc = 0.05 + 0.4 * (1 - np.exp(-epoch/3))  # Slower to ~0.45
                        elif method == 'Interleaved':
                            # Good performance but slightly worse than Scratch_T2
                            if split == 'T2_shortcut_normal':
                                acc = 0.1 + 0.7 * (1 - np.exp(-epoch/2.5))  # To ~0.8
                            else:  # masked
                                acc = 0.05 + 0.35 * (1 - np.exp(-epoch/3.5))  # To ~0.4
                        elif method == 'sgd':
                            # Poor performance due to catastrophic forgetting
                            if split == 'T2_shortcut_normal':
                                acc = 0.1 + 0.5 * (1 - np.exp(-epoch/3))  # To ~0.6
                            else:  # masked
                                acc = 0.05 + 0.2 * (1 - np.exp(-epoch/4))  # To ~0.25
                        elif method == 'derpp':
                            # Better continual learning performance
                            if split == 'T2_shortcut_normal':
                                acc = 0.1 + 0.65 * (1 - np.exp(-epoch/2.2))  # To ~0.75
                            else:  # masked
                                acc = 0.05 + 0.3 * (1 - np.exp(-epoch/3.2))  # To ~0.35
                        elif method == 'ewc_on':
                            # Moderate continual learning performance
                            if split == 'T2_shortcut_normal':
                                acc = 0.1 + 0.6 * (1 - np.exp(-epoch/2.8))  # To ~0.7
                            else:  # masked
                                acc = 0.05 + 0.25 * (1 - np.exp(-epoch/3.8))  # To ~0.3

                        # Add some noise
                        noise = np.random.normal(0, 0.02, 1)[0]
                        acc = np.clip(acc + noise, 0, 1)

                        rows.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch,
                            'split': split,
                            'acc': acc
                        })

        df = pd.DataFrame(rows)

        # Create dataset
        loader = ERIDataLoader()
        loader.validate_format(df)

        return ERITimelineDataset(
            data=df,
            metadata={'source': 'test_multi_method'},
            methods=methods,
            splits=splits,
            seeds=seeds,
            epoch_range=(0.0, 10.0)
        )

    def test_timeline_processor_multi_method_curves(self, multi_method_dataset):
        """Test ERITimelineProcessor with multi-method datasets."""
        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

        # Compute accuracy curves
        curves = processor.compute_accuracy_curves(multi_method_dataset)

        # Validate that all method-split combinations are present
        expected_keys = []
        for method in multi_method_dataset.methods:
            for split in multi_method_dataset.splits:
                expected_keys.append(f"{method}_{split}")

        assert len(curves) == len(expected_keys), f"Expected {len(expected_keys)} curves, got {len(curves)}"

        # Validate each curve
        for key, curve in curves.items():
            assert isinstance(curve, AccuracyCurve)
            assert len(curve.epochs) > 0
            assert len(curve.mean_accuracy) == len(curve.epochs)
            assert curve.n_seeds == 3  # We have 3 seeds
            assert 0 <= np.min(curve.mean_accuracy) <= np.max(curve.mean_accuracy) <= 1

        # Validate baseline methods are present
        scratch_keys = [k for k in curves.keys() if k.startswith('Scratch_T2_')]
        interleaved_keys = [k for k in curves.keys() if k.startswith('Interleaved_')]

        assert len(scratch_keys) == 2, f"Expected 2 Scratch_T2 curves, got {len(scratch_keys)}"
        assert len(interleaved_keys) == 2, f"Expected 2 Interleaved curves, got {len(interleaved_keys)}"

    def test_compute_performance_deficits_with_scratch_baseline(self, multi_method_dataset):
        """Test compute_performance_deficits() correctly uses Scratch_T2 as baseline."""
        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

        # Compute accuracy curves
        curves = processor.compute_accuracy_curves(multi_method_dataset)

        # Compute performance deficits
        pd_series = processor.compute_performance_deficits(curves, scratch_key="Scratch_T2")

        # Validate PD_t series
        expected_methods = ['Interleaved', 'sgd', 'derpp', 'ewc_on']  # Excluding Scratch_T2

        assert len(pd_series) == len(expected_methods), f"Expected PD_t for {len(expected_methods)} methods, got {len(pd_series)}"

        for method in expected_methods:
            assert method in pd_series, f"Missing PD_t series for method: {method}"

            series = pd_series[method]
            assert isinstance(series, TimeSeries)
            assert series.metric_name == 'PD_t'
            assert series.method == method
            assert len(series.epochs) > 0
            assert len(series.values) == len(series.epochs)

            # PD_t should be positive for continual learning methods (worse than Scratch_T2)
            # except possibly for Interleaved which might be close
            if method != 'Interleaved':
                assert np.mean(series.values) > 0, f"Expected positive PD_t for {method}, got {np.mean(series.values)}"

    def test_compute_sfr_relative_calculations(self, multi_method_dataset):
        """Test compute_sfr_relative() calculates relative metrics properly."""
        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

        # Compute accuracy curves
        curves = processor.compute_accuracy_curves(multi_method_dataset)

        # Compute SFR relative
        sfr_series = processor.compute_sfr_relative(curves, scratch_key="Scratch_T2")

        # Validate SFR_rel series
        expected_methods = ['Interleaved', 'sgd', 'derpp', 'ewc_on']  # Excluding Scratch_T2

        assert len(sfr_series) == len(expected_methods), f"Expected SFR_rel for {len(expected_methods)} methods, got {len(sfr_series)}"

        for method in expected_methods:
            assert method in sfr_series, f"Missing SFR_rel series for method: {method}"

            series = sfr_series[method]
            assert isinstance(series, TimeSeries)
            assert series.metric_name == 'SFR_rel'
            assert series.method == method
            assert len(series.epochs) > 0
            assert len(series.values) == len(series.epochs)

            # SFR_rel values should be reasonable (not all zero, finite)
            assert not np.all(series.values == 0), f"SFR_rel for {method} is all zeros"
            assert np.all(np.isfinite(series.values)), f"SFR_rel for {method} contains non-finite values"

    def test_dynamics_plotter_comparative_plots(self, multi_method_dataset):
        """Test ERIDynamicsPlotter generates comparative plots with multiple methods."""
        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)
        plotter = ERIDynamicsPlotter()

        # Compute all required data
        curves = processor.compute_accuracy_curves(multi_method_dataset)

        # Separate patched and masked curves
        patched_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}
        masked_curves = {k: v for k, v in curves.items() if 'shortcut_masked' in k}

        # Compute derived metrics
        pd_series = processor.compute_performance_deficits(curves)
        sfr_series = processor.compute_sfr_relative(curves)
        ad_values = processor.compute_adaptation_delays(curves)

        # Create dynamics figure
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_dynamics.pdf")

            fig = plotter.create_dynamics_figure(
                patched_curves=patched_curves,
                masked_curves=masked_curves,
                pd_series=pd_series,
                sfr_series=sfr_series,
                ad_values=ad_values,
                tau=0.6,
                title="Multi-Method ERI Dynamics Test",
                save_path=save_path
            )

            # Validate figure structure
            assert len(fig.axes) == 3, f"Expected 3 panels, got {len(fig.axes)}"

            # Validate that file was saved
            assert os.path.exists(save_path), "Figure was not saved"

            # Validate panel A (accuracy trajectories)
            ax_a = fig.axes[0]
            lines = ax_a.get_lines()
            assert len(lines) >= len(multi_method_dataset.methods), f"Expected at least {len(multi_method_dataset.methods)} lines in panel A"

            # Check that legend exists and has entries
            legend = ax_a.get_legend()
            assert legend is not None, "Panel A missing legend"
            assert len(legend.get_texts()) > 0, "Panel A legend is empty"

            # Validate panel B (PD_t)
            ax_b = fig.axes[1]
            b_lines = ax_b.get_lines()
            # Should have lines for each continual learning method (excluding Scratch_T2)
            expected_pd_lines = len([m for m in multi_method_dataset.methods if m != 'Scratch_T2'])
            assert len(b_lines) >= expected_pd_lines, f"Expected at least {expected_pd_lines} lines in panel B"

            # Validate panel C (SFR_rel)
            ax_c = fig.axes[2]
            c_lines = ax_c.get_lines()
            # Should have lines for each continual learning method (excluding Scratch_T2)
            expected_sfr_lines = len([m for m in multi_method_dataset.methods if m != 'Scratch_T2'])
            assert len(c_lines) >= expected_sfr_lines, f"Expected at least {expected_sfr_lines} lines in panel C"

            plt.close(fig)

    def test_heatmap_plotter_robustness_analysis(self, multi_method_dataset):
        """Test ERIHeatmapPlotter creates robustness heatmaps across all methods."""
        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)
        heatmap_plotter = ERIHeatmapPlotter()

        # Compute accuracy curves
        curves = processor.compute_accuracy_curves(multi_method_dataset)

        # Filter for shortcut_normal curves only (needed for AD computation)
        shortcut_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}

        # Define tau range for sensitivity analysis
        taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

        # Compute tau sensitivity
        sensitivity_result = heatmap_plotter.compute_tau_sensitivity(
            shortcut_curves, taus, baseline_method="Scratch_T2"
        )

        # Validate sensitivity result
        expected_methods = [m for m in multi_method_dataset.methods if m != 'Scratch_T2']
        assert sensitivity_result.methods == sorted(expected_methods), f"Expected methods {sorted(expected_methods)}, got {sensitivity_result.methods}"
        assert len(sensitivity_result.taus) == len(taus), f"Expected {len(taus)} tau values, got {len(sensitivity_result.taus)}"
        assert sensitivity_result.ad_matrix.shape == (len(expected_methods), len(taus)), f"Unexpected AD matrix shape: {sensitivity_result.ad_matrix.shape}"
        assert sensitivity_result.baseline_method == "Scratch_T2"

        # Create heatmap
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_heatmap.pdf")

            fig = heatmap_plotter.create_tau_sensitivity_heatmap(
                sensitivity_result,
                title="Multi-Method AD(τ) Sensitivity Test"
            )

            # Save the figure
            heatmap_plotter.save_heatmap(fig, save_path)

            # Validate figure structure
            assert len(fig.axes) >= 1, "Heatmap figure should have at least 1 axis"

            # Validate that file was saved
            assert os.path.exists(save_path), "Heatmap was not saved"

            # Validate heatmap content
            ax = fig.axes[0]

            # Check that colorbar exists
            assert hasattr(fig, 'colorbar') or len(fig.axes) > 1, "Heatmap should have a colorbar"

            # Check axis labels
            assert ax.get_xlabel(), "Heatmap missing x-axis label"
            assert ax.get_ylabel(), "Heatmap missing y-axis label"
            assert ax.get_title(), "Heatmap missing title"

            # Check tick labels
            x_labels = [t.get_text() for t in ax.get_xticklabels()]
            y_labels = [t.get_text() for t in ax.get_yticklabels()]

            assert len(x_labels) == len(taus), f"Expected {len(taus)} x-tick labels, got {len(x_labels)}"
            assert len(y_labels) == len(expected_methods), f"Expected {len(expected_methods)} y-tick labels, got {len(y_labels)}"

            plt.close(fig)

    def test_adaptation_delays_computation(self, multi_method_dataset):
        """Test adaptation delay computation with multiple methods."""
        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

        # Compute accuracy curves
        curves = processor.compute_accuracy_curves(multi_method_dataset)

        # Compute adaptation delays
        ad_values = processor.compute_adaptation_delays(curves)

        # Validate AD values
        expected_methods = [m for m in multi_method_dataset.methods if m != 'Scratch_T2']

        # Should have AD values for continual learning methods
        for method in expected_methods:
            assert method in ad_values, f"Missing AD value for method: {method}"

            ad_val = ad_values[method]
            # AD should be finite (not NaN) for our synthetic data
            # and positive for continual learning methods (they should be slower than Scratch_T2)
            if not np.isnan(ad_val):
                assert ad_val >= 0, f"Expected non-negative AD for {method}, got {ad_val}"

    def test_baseline_detection_and_handling(self, multi_method_dataset):
        """Test that baseline methods are correctly detected and handled."""
        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

        # Compute accuracy curves
        curves = processor.compute_accuracy_curves(multi_method_dataset)

        # Test with correct baseline
        pd_series = processor.compute_performance_deficits(curves, scratch_key="Scratch_T2")
        assert len(pd_series) > 0, "Should compute PD_t when Scratch_T2 baseline is present"

        sfr_series = processor.compute_sfr_relative(curves, scratch_key="Scratch_T2")
        assert len(sfr_series) > 0, "Should compute SFR_rel when Scratch_T2 baseline is present"

        # Test with missing baseline
        curves_no_scratch = {k: v for k, v in curves.items() if not k.startswith('Scratch_T2')}

        pd_series_no_baseline = processor.compute_performance_deficits(curves_no_scratch, scratch_key="Scratch_T2")
        assert len(pd_series_no_baseline) == 0, "Should not compute PD_t when baseline is missing"

        sfr_series_no_baseline = processor.compute_sfr_relative(curves_no_scratch, scratch_key="Scratch_T2")
        assert len(sfr_series_no_baseline) == 0, "Should not compute SFR_rel when baseline is missing"

    def test_multi_method_data_consistency(self, multi_method_dataset):
        """Test data consistency across multiple methods."""
        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

        # Compute accuracy curves
        curves = processor.compute_accuracy_curves(multi_method_dataset)

        # Check that all methods have the same splits
        methods_splits = {}
        for key, curve in curves.items():
            method = curve.method
            if method not in methods_splits:
                methods_splits[method] = set()
            methods_splits[method].add(curve.split)

        # All methods should have the same splits
        expected_splits = set(multi_method_dataset.splits)
        for method, splits in methods_splits.items():
            assert splits == expected_splits, f"Method {method} has inconsistent splits: {splits} vs {expected_splits}"

        # Check epoch alignment across methods
        epoch_ranges = {}
        for key, curve in curves.items():
            method = curve.method
            epoch_range = (curve.epochs.min(), curve.epochs.max())
            if method not in epoch_ranges:
                epoch_ranges[method] = epoch_range
            else:
                # Should be consistent within method
                assert abs(epoch_ranges[method][0] - epoch_range[0]) < 0.1, f"Inconsistent epoch start for {method}"
                assert abs(epoch_ranges[method][1] - epoch_range[1]) < 0.1, f"Inconsistent epoch end for {method}"

    def test_processing_summary_multi_method(self, multi_method_dataset):
        """Test processing summary with multi-method data."""
        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

        # Compute accuracy curves
        curves = processor.compute_accuracy_curves(multi_method_dataset)

        # Get processing summary
        summary = processor.get_processing_summary(curves)

        # Validate summary
        assert summary['n_curves'] == len(curves)
        assert set(summary['methods']) == set(multi_method_dataset.methods)
        assert set(summary['splits']) == set(multi_method_dataset.splits)
        assert summary['smoothing_window'] == 3
        assert summary['tau_threshold'] == 0.6

        # Check curves per method
        for method in multi_method_dataset.methods:
            expected_count = len(multi_method_dataset.splits)  # One curve per split
            assert summary['curves_per_method'][method] == expected_count, f"Expected {expected_count} curves for {method}"

        # Check curves per split
        for split in multi_method_dataset.splits:
            expected_count = len(multi_method_dataset.methods)  # One curve per method
            assert summary['curves_per_split'][split] == expected_count, f"Expected {expected_count} curves for {split}"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
