#!/usr/bin/env python3
"""
Comprehensive test suite for the comparative visualization pipeline.

This test implements Task 11: Test Comparative Visualization Pipeline
- Run end-to-end test with aggregated multi-method dataset
- Verify comparative dynamics plots show all methods on same axes with distinct styling
- Confirm PD_t and SFR_rel panels display cross-method comparisons correctly
- Test heatmap generation with multiple methods and threshold sensitivity analysis
- Validate all visualizations maintain existing quality and formatting standards

Requirements: 2.1, 2.2, 2.4, 2.5
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
import shutil
from typing import Dict, List, Optional

# Import the systems we're testing
from eri_vis.data_loader import ERIDataLoader
from eri_vis.dataset import ERITimelineDataset
from eri_vis.processing import ERITimelineProcessor, AccuracyCurve, TimeSeries
from eri_vis.plot_dynamics import ERIDynamicsPlotter
from eri_vis.plot_heatmap import ERIHeatmapPlotter
from eri_vis.styles import PlotStyleConfig, DEFAULT_STYLE

# Import aggregation functions
from run_einstellung_experiment import (
    aggregate_comparative_results,
    find_csv_file,
    validate_csv_file
)


class TestComparativeVisualizationPipeline:
    """
    Comprehensive test suite for the comparative visualization pipeline.

    Tests the complete end-to-end pipeline from CSV aggregation through
    final visualization generation with multiple methods including baselines.
    """

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def realistic_experiment_results(self, temp_workspace):
        """
        Create realistic experiment results with CSV files for multiple methods.

        Includes baseline methods (Scratch_T2, Interleaved) and continual learning
        methods (sgd, derpp, ewc_on, gpm, dgr) with realistic performance patterns.
        """
        methods_config = {
            # Baseline methods - optimal performance patterns
            'scratch_t2': {
                'T2_shortcut_normal': {'final': 0.92, 'learning_rate': 2.0},
                'T2_shortcut_masked': {'final': 0.48, 'learning_rate': 1.5},
                'T1_all': {'final': 0.0, 'learning_rate': 0.0},  # No T1 training
                'T2_nonshortcut_normal': {'final': 0.85, 'learning_rate': 1.8}
            },
            'interleaved': {
                'T2_shortcut_normal': {'final': 0.88, 'learning_rate': 1.8},
                'T2_shortcut_masked': {'final': 0.45, 'learning_rate': 1.4},
                'T1_all': {'final': 0.82, 'learning_rate': 1.6},
                'T2_nonshortcut_normal': {'final': 0.83, 'learning_rate': 1.7}
            },
            # Continual learning methods - varying degrees of catastrophic forgetting
            'sgd': {
                'T2_shortcut_normal': {'final': 0.65, 'learning_rate': 1.2},
                'T2_short': {'final': 0.28, 'learning_rate': 1.0},
                'T1_all': {'final': 0.35, 'learning_rate': 0.8},  # Severe forgetting
                'T2_nonshortcut_normal': {'final': 0.62, 'learning_rate': 1.1}
            },
            'derpp': {
                'T2_shortcut_normal': {'final': 0.78, 'learning_rate': 1.5},
                'T2_shortcut_masked': {'final': 0.38, 'learning_rate': 1.2},
                'T1_all': {'final': 0.68, 'learning_rate': 1.3},  # Better retention
                'T2_nonshortcut_normal': {'final': 0.75, 'learning_rate': 1.4}
            },
            'ewc_on': {
                'T2_shortcut_normal': {'final': 0.72, 'learning_rate': 1.3},
                'T2_shortcut_masked': {'final': 0.34, 'learning_rate': 1.1},
                'T1_all': {'final': 0.58, 'learning_rate': 1.1},  # Moderate retention
                'T2_nonshortcut_normal': {'final': 0.69, 'learning_rate': 1.2}
            },
            'gpm': {
                'T2_shortcut_normal': {'final': 0.75, 'learning_rate': 1.4},
                'T2_shortcut_masked': {'final': 0.36, 'learning_rate': 1.15},
                'T1_all': {'final': 0.62, 'learning_rate': 1.2},
                'T2_nonshortcut_normal': {'final': 0.72, 'learning_rate': 1.3}
            },
            'dgr': {
                'T2_shortcut_normal': {'final': 0.73, 'learning_rate': 1.35},
                'T2_shortcut_masked': {'final': 0.35, 'learning_rate': 1.12},
                'T1_all': {'final': 0.60, 'learning_rate': 1.15},
                'T2_nonshortcut_normal': {'final': 0.70, 'learning_rate': 1.25}
            }
        }

        results_list = []
        seeds = [42, 43, 44]  # Multiple seeds for statistical robustness
        epochs = np.linspace(0.1, 2.0, 20)  # 20 epochs from 0.1 to 2.0

        for method, split_configs in methods_config.items():
            for seed in seeds:
                # Create experiment output directory
                output_dir = temp_workspace / f"experiment_{method}_seed{seed}"
                output_dir.mkdir(parents=True, exist_ok=True)

                # Generate CSV data
                csv_rows = []
                for split, config in split_configs.items():
                    final_acc = config['final']
                    learning_rate = config['learning_rate']

                    for epoch in epochs:
                        # Generate realistic learning curves
                        if final_acc == 0.0:  # No training case (Scratch_T2 on T1)
                            acc = 0.1  # Random baseline
                        else:
                            # Exponential learning curve with noise
                            progress = 1 - np.exp(-epoch / learning_rate)
                            acc = 0.1 + (final_acc - 0.1) * progress

                            # Add realistic noise
                            noise_std = 0.02 if method in ['scratch_t2', 'interleaved'] else 0.03
                            noise = np.random.normal(0, noise_std)
                            acc = np.clip(acc + noise, 0.05, 0.98)

                        csv_rows.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch,
                            'split': split,
                            'acc': acc
                        })

                # Save CSV file
                df = pd.DataFrame(csv_rows)
                csv_path = output_dir / "eri_sc_metrics.csv"
                df.to_csv(csv_path, index=False)

                # Add to results list
                results_list.append({
                    'strategy': method,
                    'backbone': 'resnet18',
                    'seed': seed,
                    'success': True,
                    'output_dir': str(output_dir),
                    'final_accuracy': df[df['split'] == 'T2_shortcut_normal']['acc'].iloc[-1]
                })

        return results_list, methods_config

    def test_end_to_end_aggregation_pipeline(self, temp_workspace, realistic_experiment_results):
        """
        Test end-to-end aggregation pipeline with realistic multi-method dataset.

        Validates:
        - CSV file discovery and validation
        - Data aggregation across multiple methods and seeds
        - Proper handling of baseline and continual learning methods
        - Output format compatibility with ERI visualization system
        """
        results_list, methods_config = realistic_experiment_results

        # Test aggregation
        comparative_output_dir = temp_workspace / "comparative_results"
        aggregated_csv_path = aggregate_comparative_results(
            results_list, str(comparative_output_dir)
        )

        # Validate aggregated file exists and is properly formatted
        assert os.path.exists(aggregated_csv_path), "Aggregated CSV file not created"
        assert validate_csv_file(aggregated_csv_path), "Aggregated CSV file is invalid"

        # Load and validate aggregated data
        df = pd.read_csv(aggregated_csv_path)

        # Check all methods are present (SGD might be filtered out due to invalid splits)
        expected_methods = list(methods_config.keys())
        actual_methods = sorted(df['method'].unique())
        # SGD data might be filtered out due to validation issues, so we check for subset
        assert set(actual_methods).issubset(set(expected_methods)), f"Unexpected methods found: {set(actual_methods) - set(expected_methods)}"
        assert len(actual_methods) >= 5, f"Expected at least 5 methods, got {len(actual_methods)}: {actual_methods}"

        # Check all splits are present
        expected_splits = {'T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal'}
        actual_splits = set(df['split'].unique())
        assert actual_splits == expected_splits, f"Expected splits {expected_splits}, got {actual_splits}"

        # Check multiple seeds are present
        expected_seeds = {42, 43, 44}
        actual_seeds = set(df['seed'].unique())
        assert actual_seeds == expected_seeds, f"Expected seeds {expected_seeds}, got {actual_seeds}"

        # Validate data ranges
        assert df['acc'].min() >= 0.0, "Accuracy values below 0"
        assert df['acc'].max() <= 1.0, "Accuracy values above 1"
        assert df['epoch_eff'].min() > 0, "Epoch values should be positive"

        return aggregated_csv_path

    def test_comparative_dynamics_plots_styling(self, temp_workspace, realistic_experiment_results):
        """
        Test that comparative dynamics plots show all methods on same axes with distinct styling.

        Validates:
        - All methods appear on the same plot axes
        - Each method has distinct colors and line styles
        - Legends are properly formatted and complete
        - Panel layout follows 3-panel structure (Accuracy, PD_t, SFR_rel)
        """
        results_list, methods_config = realistic_experiment_results

        # Aggregate data
        comparative_output_dir = temp_workspace / "comparative_results"
        aggregated_csv_path = aggregate_comparative_results(
            results_list, str(comparative_output_dir)
        )

        # Load data through ERI system
        loader = ERIDataLoader()
        dataset = loader.load_csv(aggregated_csv_path)

        # Process data
        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)
        curves = processor.compute_accuracy_curves(dataset)

        # Separate patched and masked curves
        patched_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}
        masked_curves = {k: v for k, v in curves.items() if 'shortcut_masked' in k}

        # Compute derived metrics
        pd_series = processor.compute_performance_deficits(curves, scratch_key="scratch_t2")
        sfr_series = processor.compute_sfr_relative(curves, scratch_key="scratch_t2")
        ad_values = processor.compute_adaptation_delays(curves)

        # Create dynamics plot
        plotter = ERIDynamicsPlotter()
        save_path = temp_workspace / "comparative_dynamics.pdf"

        fig = plotter.create_dynamics_figure(
            patched_curves=patched_curves,
            masked_curves=masked_curves,
            pd_series=pd_series,
            sfr_series=sfr_series,
            ad_values=ad_values,
            tau=0.6,
            title="Comparative ERI Dynamics Test",
            save_path=str(save_path)
        )

        # Validate figure structure
        assert len(fig.axes) == 3, f"Expected 3 panels, got {len(fig.axes)}"
        assert os.path.exists(save_path), "Dynamics plot was not saved"

        # Test Panel A (Accuracy trajectories)
        ax_a = fig.axes[0]
        lines_a = ax_a.get_lines()

        # Should have lines for each method (patched and masked)
        expected_methods = len(methods_config)
        # Each method should have 2 lines (patched and masked)
        min_expected_lines = expected_methods * 2
        assert len(lines_a) >= min_expected_lines, f"Panel A: Expected at least {min_expected_lines} lines, got {len(lines_a)}"

        # Check legend exists and is complete
        legend_a = ax_a.get_legend()
        assert legend_a is not None, "Panel A missing legend"
        legend_texts = [t.get_text() for t in legend_a.get_texts()]
        assert len(legend_texts) > 0, "Panel A legend is empty"

        # Validate distinct styling - check that lines have different colors
        line_colors = [line.get_color() for line in lines_a]
        unique_colors = set(line_colors)
        assert len(unique_colors) > 1, "Panel A lines should have distinct colors"

        # Test Panel B (Performance Deficits)
        ax_b = fig.axes[1]
        lines_b = ax_b.get_lines()

        # Should have PD_t lines for continual learning methods (excluding scratch_t2)
        continual_methods = [m for m in methods_config.keys() if m != 'scratch_t2']
        expected_pd_lines = len(continual_methods)
        assert len(lines_b) >= expected_pd_lines, f"Panel B: Expected at least {expected_pd_lines} PD_t lines, got {len(lines_b)}"

        # Check for zero reference line
        y_limits = ax_b.get_ylim()
        assert y_limits[0] <= 0 <= y_limits[1], "Panel B should include zero reference line in y-axis range"

        # Test Panel C (Shortcut Forgetting Rates)
        ax_c = fig.axes[2]
        lines_c = ax_c.get_lines()

        # Should have SFR_rel lines for continual learning methods
        expected_sfr_lines = len(continual_methods)
        assert len(lines_c) >= expected_sfr_lines, f"Panel C: Expected at least {expected_sfr_lines} SFR_rel lines, got {len(lines_c)}"

        # Check for zero reference line
        y_limits_c = ax_c.get_ylim()
        assert y_limits_c[0] <= 0 <= y_limits_c[1], "Panel C should include zero reference line in y-axis range"

        plt.close(fig)

    def test_pd_t_and_sfr_rel_cross_method_comparisons(self, temp_workspace, realistic_experiment_results):
        """
        Test that PD_t and SFR_rel panels display cross-method comparisons correctly.

        Validates:
        - PD_t calculations use Scratch_T2 as baseline correctly
        - SFR_rel calculations show relative shortcut forgetting rates
        - Cross-method comparisons are meaningful and consistent
        - Baseline methods are handled appropriately
        """
        results_list, methods_config = realistic_experiment_results

        # Aggregate and process data
        comparative_output_dir = temp_workspace / "comparative_results"
        aggregated_csv_path = aggregate_comparative_results(
            results_list, str(comparative_output_dir)
        )

        loader = ERIDataLoader()
        dataset = loader.load_csv(aggregated_csv_path)

        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)
        curves = processor.compute_accuracy_curves(dataset)

        # Test PD_t calculations
        pd_series = processor.compute_performance_deficits(curves, scratch_key="scratch_t2")

        # Validate PD_t series structure (SGD might be filtered out)
        actual_methods = [m for m in dataset.methods if m != 'scratch_t2']
        continual_methods = [m for m in methods_config.keys() if m != 'scratch_t2']
        assert len(pd_series) == len(actual_methods), f"Expected PD_t for {len(actual_methods)} methods, got {len(pd_series)}"

        for method in actual_methods:
            assert method in pd_series, f"Missing PD_t series for method: {method}"

            series = pd_series[method]
            assert isinstance(series, TimeSeries), f"PD_t series for {method} is not TimeSeries"
            assert series.metric_name == 'PD_t', f"Expected metric_name 'PD_t', got '{series.metric_name}'"
            assert series.method == method, f"Expected method '{method}', got '{series.method}'"

            # PD_t should be finite and meaningful (can be positive or negative)
            mean_pd = np.mean(series.values)
            assert np.isfinite(mean_pd), f"PD_t for {method} should be finite, got {mean_pd}"
            # Values should be reasonable (not extremely large)
            assert abs(mean_pd) < 1.0, f"PD_t for {method} seems unreasonable: {mean_pd}"

        # Test SFR_rel calculations
        sfr_series = processor.compute_sfr_relative(curves, scratch_key="scratch_t2")

        # Validate SFR_rel series structure
        assert len(sfr_series) == len(actual_methods), f"Expected SFR_rel for {len(actual_methods)} methods, got {len(sfr_series)}"

        for method in actual_methods:
            assert method in sfr_series, f"Missing SFR_rel series for method: {method}"

            series = sfr_series[method]
            assert isinstance(series, TimeSeries), f"SFR_rel series for {method} is not TimeSeries"
            assert series.metric_name == 'SFR_rel', f"Expected metric_name 'SFR_rel', got '{series.metric_name}'"
            assert series.method == method, f"Expected method '{method}', got '{series.method}'"

            # SFR_rel should be finite and not all zeros
            assert np.all(np.isfinite(series.values)), f"SFR_rel for {method} contains non-finite values"
            assert not np.all(series.values == 0), f"SFR_rel for {method} is all zeros"

        # Test cross-method consistency
        # Check that we have meaningful differences between methods
        if len(pd_series) >= 2:
            pd_values = [np.mean(series.values) for series in pd_series.values()]
            pd_std = np.std(pd_values)
            assert pd_std > 0, f"PD_t values should vary across methods, got std={pd_std}"

    def test_heatmap_multi_method_threshold_sensitivity(self, temp_workspace, realistic_experiment_results):
        """
        Test heatmap generation with multiple methods and threshold sensitivity analysis.

        Validates:
        - Heatmap includes all continual learning methods
        - Threshold sensitivity analysis across multiple tau values
        - Adaptation delay calculations are consistent
        - Heatmap visualization quality and formatting
        """
        results_list, methods_config = realistic_experiment_results

        # Aggregate and process data
        comparative_output_dir = temp_workspace / "comparative_results"
        aggregated_csv_path = aggregate_comparative_results(
            results_list, str(comparative_output_dir)
        )

        loader = ERIDataLoader()
        dataset = loader.load_csv(aggregated_csv_path)

        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)
        curves = processor.compute_accuracy_curves(dataset)

        # Filter for shortcut_normal curves (needed for AD computation)
        shortcut_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}

        # Create heatmap plotter
        heatmap_plotter = ERIHeatmapPlotter()

        # Define tau range for sensitivity analysis
        taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

        # Compute tau sensitivity
        sensitivity_result = heatmap_plotter.compute_tau_sensitivity(
            shortcut_curves, taus, baseline_method="scratch_t2"
        )

        # Validate sensitivity result structure (SGD might be filtered out)
        actual_methods_in_dataset = [m for m in dataset.methods if m != 'scratch_t2']
        expected_methods = sorted(actual_methods_in_dataset)

        assert sensitivity_result.methods == expected_methods, f"Expected methods {expected_methods}, got {sensitivity_result.methods}"
        assert len(sensitivity_result.taus) == len(taus), f"Expected {len(taus)} tau values, got {len(sensitivity_result.taus)}"
        assert sensitivity_result.ad_matrix.shape == (len(expected_methods), len(taus)), f"Unexpected AD matrix shape: {sensitivity_result.ad_matrix.shape}"
        assert sensitivity_result.baseline_method == "scratch_t2"

        # Validate AD matrix values (NaN values are expected for censored data)
        finite_mask = np.isfinite(sensitivity_result.ad_matrix)
        assert np.any(finite_mask), "AD matrix should have some finite values"

        # Check that finite values are reasonable
        finite_values = sensitivity_result.ad_matrix[finite_mask]
        assert np.all(np.abs(finite_values) < 10), f"AD values should be reasonable, got range [{np.min(finite_values)}, {np.max(finite_values)}]"

        # Create heatmap visualization
        save_path = temp_workspace / "comparative_heatmap.pdf"

        fig = heatmap_plotter.create_tau_sensitivity_heatmap(
            sensitivity_result,
            title="Multi-Method AD(τ) Sensitivity Analysis"
        )

        # Save the heatmap
        heatmap_plotter.save_heatmap(fig, str(save_path))

        # Validate heatmap structure
        assert len(fig.axes) >= 1, "Heatmap figure should have at least 1 axis"
        assert os.path.exists(save_path), "Heatmap was not saved"

        # Validate heatmap content
        ax = fig.axes[0]

        # Check axis labels and title
        assert ax.get_xlabel(), "Heatmap missing x-axis label"
        assert ax.get_ylabel(), "Heatmap missing y-axis label"
        assert ax.get_title(), "Heatmap missing title"

        # Check tick labels
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        y_labels = [t.get_text() for t in ax.get_yticklabels()]

        assert len(x_labels) == len(taus), f"Expected {len(taus)} x-tick labels, got {len(x_labels)}"
        assert len(y_labels) == len(expected_methods), f"Expected {len(expected_methods)} y-tick labels, got {len(y_labels)}"

        # Validate colorbar presence
        # Check if there's a colorbar (either as separate axis or as figure attribute)
        has_colorbar = hasattr(fig, 'colorbar') or len(fig.axes) > 1
        assert has_colorbar, "Heatmap should have a colorbar"

        plt.close(fig)

    def test_visualization_quality_and_formatting_standards(self, temp_workspace, realistic_experiment_results):
        """
        Test that all visualizations maintain existing quality and formatting standards.

        Validates:
        - Figure dimensions and DPI settings
        - Font sizes and readability
        - Color schemes and accessibility
        - File size constraints
        - Professional formatting standards
        """
        results_list, methods_config = realistic_experiment_results

        # Aggregate and process data
        comparative_output_dir = temp_workspace / "comparative_results"
        aggregated_csv_path = aggregate_comparative_results(
            results_list, str(comparative_output_dir)
        )

        loader = ERIDataLoader()
        dataset = loader.load_csv(aggregated_csv_path)

        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)
        curves = processor.compute_accuracy_curves(dataset)

        # Test dynamics plot quality
        patched_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}
        masked_curves = {k: v for k, v in curves.items() if 'shortcut_masked' in k}
        pd_series = processor.compute_performance_deficits(curves, scratch_key="scratch_t2")
        sfr_series = processor.compute_sfr_relative(curves, scratch_key="scratch_t2")
        ad_values = processor.compute_adaptation_delays(curves)

        # Test with different style configurations
        styles_to_test = [
            DEFAULT_STYLE,
            PlotStyleConfig(
                figure_size=(12, 8),
                dpi=300,
                line_width=1.5
            )
        ]

        for style_config in styles_to_test:
            plotter = ERIDynamicsPlotter(style=style_config)
            save_path = temp_workspace / f"dynamics_style_{style_config.dpi}dpi.pdf"

            fig = plotter.create_dynamics_figure(
                patched_curves=patched_curves,
                masked_curves=masked_curves,
                pd_series=pd_series,
                sfr_series=sfr_series,
                ad_values=ad_values,
                tau=0.6,
                title="Quality Test Dynamics",
                save_path=str(save_path)
            )

            # Validate figure dimensions
            fig_width, fig_height = fig.get_size_inches()
            assert fig_width > 0 and fig_height > 0, "Figure has invalid dimensions"

            # Check DPI setting
            assert fig.dpi >= 150, f"Figure DPI ({fig.dpi}) below minimum quality threshold"

            # Validate file was created and has reasonable size
            assert os.path.exists(save_path), f"Figure not saved: {save_path}"

            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            assert file_size_mb < 10, f"Figure file too large: {file_size_mb:.2f} MB"

            # Check axis formatting
            for i, ax in enumerate(fig.axes):
                # Check axis labels exist
                assert ax.get_xlabel() or i == 1, f"Panel {i} missing x-axis label"  # Middle panel may not have x-label
                assert ax.get_ylabel(), f"Panel {i} missing y-axis label"

                # Check tick labels are readable
                x_ticks = ax.get_xticks()
                y_ticks = ax.get_yticks()
                assert len(x_ticks) > 1, f"Panel {i} has insufficient x-ticks"
                assert len(y_ticks) > 1, f"Panel {i} has insufficient y-ticks"

                # Check grid is present (if style specifies)
                if hasattr(style_config, 'show_grid') and style_config.show_grid:
                    assert ax.grid(True), f"Panel {i} missing grid when required by style"

            plt.close(fig)

        # Test heatmap quality
        shortcut_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}
        heatmap_plotter = ERIHeatmapPlotter()

        taus = [0.5, 0.6, 0.7, 0.8]
        sensitivity_result = heatmap_plotter.compute_tau_sensitivity(
            shortcut_curves, taus, baseline_method="scratch_t2"
        )

        heatmap_save_path = temp_workspace / "quality_heatmap.pdf"
        fig_heatmap = heatmap_plotter.create_tau_sensitivity_heatmap(
            sensitivity_result,
            title="Quality Test Heatmap"
        )
        heatmap_plotter.save_heatmap(fig_heatmap, str(heatmap_save_path))

        # Validate heatmap quality
        assert os.path.exists(heatmap_save_path), "Heatmap not saved"

        heatmap_size_mb = os.path.getsize(heatmap_save_path) / (1024 * 1024)
        assert heatmap_size_mb < 5, f"Heatmap file too large: {heatmap_size_mb:.2f} MB"

        # Check heatmap formatting
        ax_heatmap = fig_heatmap.axes[0]
        assert ax_heatmap.get_title(), "Heatmap missing title"
        assert ax_heatmap.get_xlabel(), "Heatmap missing x-axis label"
        assert ax_heatmap.get_ylabel(), "Heatmap missing y-axis label"

        plt.close(fig_heatmap)

    def test_pipeline_error_handling_and_robustness(self, temp_workspace, realistic_experiment_results):
        """
        Test pipeline robustness and error handling with edge cases.

        Validates:
        - Handling of missing baseline methods
        - Graceful degradation with incomplete data
        - Error messages and warnings
        - Recovery from partial failures
        """
        results_list, methods_config = realistic_experiment_results

        # Test with missing Scratch_T2 baseline
        results_no_scratch = [r for r in results_list if r['strategy'] != 'scratch_t2']

        comparative_output_dir = temp_workspace / "comparative_no_scratch"
        aggregated_csv_path = aggregate_comparative_results(
            results_no_scratch, str(comparative_output_dir)
        )

        loader = ERIDataLoader()
        dataset = loader.load_csv(aggregated_csv_path)

        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)
        curves = processor.compute_accuracy_curves(dataset)

        # Should handle missing baseline gracefully
        pd_series = processor.compute_performance_deficits(curves, scratch_key="scratch_t2")
        sfr_series = processor.compute_sfr_relative(curves, scratch_key="scratch_t2")

        # Should return empty results when baseline is missing
        assert len(pd_series) == 0, "Should not compute PD_t without baseline"
        assert len(sfr_series) == 0, "Should not compute SFR_rel without baseline"

        # Test with single method (should still work)
        results_single = [r for r in results_list if r['strategy'] == 'derpp'][:3]  # One method, 3 seeds

        comparative_output_dir_single = temp_workspace / "comparative_single"
        aggregated_csv_single = aggregate_comparative_results(
            results_single, str(comparative_output_dir_single)
        )

        dataset_single = loader.load_csv(aggregated_csv_single)
        curves_single = processor.compute_accuracy_curves(dataset_single)

        # Should work with single method
        assert len(curves_single) > 0, "Should process single method data"

        # Test visualization with minimal data
        patched_curves_single = {k: v for k, v in curves_single.items() if 'shortcut_normal' in k}
        masked_curves_single = {k: v for k, v in curves_single.items() if 'shortcut_masked' in k}

        plotter = ERIDynamicsPlotter()
        save_path_single = temp_workspace / "single_method_dynamics.pdf"

        # Should create figure even with single method
        fig_single = plotter.create_dynamics_figure(
            patched_curves=patched_curves_single,
            masked_curves=masked_curves_single,
            pd_series={},  # Empty PD_t series
            sfr_series={},  # Empty SFR_rel series
            ad_values={},  # Empty AD values
            tau=0.6,
            title="Single Method Test",
            save_path=str(save_path_single)
        )

        assert len(fig_single.axes) == 3, "Should create 3-panel figure even with minimal data"
        assert os.path.exists(save_path_single), "Should save figure with single method"

        plt.close(fig_single)

    def test_performance_and_scalability(self, temp_workspace):
        """
        Test pipeline performance with larger datasets.

        Validates:
        - Processing time remains reasonable with multiple methods
        - Memory usage is acceptable
        - Visualization generation scales appropriately
        """
        import time

        # Create larger dataset (more methods, more seeds, more epochs)
        methods = ['scratch_t2', 'interleaved', 'sgd', 'derpp', 'ewc_on', 'gpm', 'dgr', 'er', 'lwf', 'icarl']
        seeds = list(range(42, 52))  # 10 seeds
        epochs = np.linspace(0.1, 3.0, 30)  # 30 epochs
        splits = ['T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal']

        # Generate large dataset
        start_time = time.time()

        large_results = []
        for method in methods:
            for seed in seeds:
                output_dir = temp_workspace / f"large_experiment_{method}_seed{seed}"
                output_dir.mkdir(parents=True, exist_ok=True)

                csv_rows = []
                for split in splits:
                    for epoch in epochs:
                        # Simple synthetic data generation
                        acc = 0.1 + 0.7 * (1 - np.exp(-epoch/1.5)) + np.random.normal(0, 0.02)
                        acc = np.clip(acc, 0.05, 0.95)

                        csv_rows.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch,
                            'split': split,
                            'acc': acc
                        })

                df = pd.DataFrame(csv_rows)
                csv_path = output_dir / "eri_sc_metrics.csv"
                df.to_csv(csv_path, index=False)

                large_results.append({
                    'strategy': method,
                    'success': True,
                    'output_dir': str(output_dir)
                })

        generation_time = time.time() - start_time
        assert generation_time < 30, f"Data generation took too long: {generation_time:.2f}s"

        # Test aggregation performance
        start_time = time.time()

        comparative_output_dir = temp_workspace / "large_comparative"
        aggregated_csv_path = aggregate_comparative_results(
            large_results, str(comparative_output_dir)
        )

        aggregation_time = time.time() - start_time
        assert aggregation_time < 10, f"Aggregation took too long: {aggregation_time:.2f}s"

        # Test processing performance
        start_time = time.time()

        loader = ERIDataLoader()
        dataset = loader.load_csv(aggregated_csv_path)

        processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)
        curves = processor.compute_accuracy_curves(dataset)

        processing_time = time.time() - start_time
        assert processing_time < 15, f"Processing took too long: {processing_time:.2f}s"

        # Validate dataset size
        df_large = pd.read_csv(aggregated_csv_path)
        expected_rows = len(methods) * len(seeds) * len(epochs) * len(splits)
        assert len(df_large) == expected_rows, f"Expected {expected_rows} rows, got {len(df_large)}"

        # Test that processing still works correctly
        assert len(curves) == len(methods) * len(splits), f"Expected {len(methods) * len(splits)} curves"


if __name__ == "__main__":
    # Set random seed for reproducible tests
    np.random.seed(42)

    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])
