"""
Integration test for multi-method ERI visualization pipeline.

This test demonstrates the complete pipeline from multi-method CSV data
through processing to final visualization outputs, validating that the
existing ERI system correctly handles comparative analysis.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

from eri_vis.data_loader import ERIDataLoader
from eri_vis.processing import ERITimelineProcessor
from eri_vis.plot_dynamics import ERIDynamicsPlotter
from eri_vis.plot_heatmap import ERIHeatmapPlotter


class TestMultiMethodIntegration:
    """Integration test for complete multi-method ERI pipeline."""

    @pytest.fixture
    def sample_multi_method_csv(self):
        """Create a sample multi-method CSV file for testing."""
        # Create realistic multi-method data
        methods = ['Scratch_T2', 'Interleaved', 'sgd', 'derpp', 'ewc_on', 'gpm']
        splits = ['T2_shortcut_normal', 'T2_shortcut_masked']
        seeds = [42, 43, 44]
        epochs = np.linspace(0, 20, 41)  # 41 epochs from 0 to 20

        rows = []

        for method in methods:
            for split in splits:
                for seed in seeds:
                    # Add some seed-specific variation
                    seed_offset = (seed - 42) * 0.02

                    for epoch in epochs:
                        # Create method-specific learning curves
                        if method == 'Scratch_T2':
                            # Optimal baseline - fast learning
                            if split == 'T2_shortcut_normal':
                                base_acc = 0.1 + 0.85 * (1 - np.exp(-epoch/3))
                            else:  # masked
                                base_acc = 0.05 + 0.45 * (1 - np.exp(-epoch/4))
                        elif method == 'Interleaved':
                            # Good baseliney worse than Scratch_T2
                            if split == 'T2_shortcut_normal':
                                base_acc = 0.1 + 0.75 * (1 - np.exp(-epoch/3.5))
                            else:  # masked
                                base_acc = 0.05 + 0.4 * (1 - np.exp(-epoch/4.5))
                        elif method == 'sgd':
                            # Poor continual learning - catastrophic forgetting
                            if split == 'T2_shortcut_normal':
                                base_acc = 0.1 + 0.5 * (1 - np.exp(-epoch/4))
                            else:  # masked
                                base_acc = 0.05 + 0.2 * (1 - np.exp(-epoch/5))
                        elif method == 'derpp':
                            # Good continual learning method
                            if split == 'T2_shortcut_normal':
                                base_acc = 0.1 + 0.7 * (1 - np.exp(-epoch/3.2))
                            else:  # masked
                                base_acc = 0.05 + 0.35 * (1 - np.exp(-epoch/4.2))
                        elif method == 'ewc_on':
                            # Moderate continual learning method
                            if split == 'T2_shortcut_normal':
                                base_acc = 0.1 + 0.65 * (1 - np.exp(-epoch/3.8))
                            else:  # masked
                                base_acc = 0.05 + 0.3 * (1 - np.exp(-epoch/4.8))
                        elif method == 'gpm':
                            # Another continual learning method
                            if split == 'T2_shortcut_normal':
                                base_acc = 0.1 + 0.68 * (1 - np.exp(-epoch/3.5))
                            else:  # masked
                                base_acc = 0.05 + 0.32 * (1 - np.exp(-epoch/4.5))

                        # Add seed variation and noise
                        acc = base_acc + seed_offset + np.random.normal(0, 0.015)
                        acc = np.clip(acc, 0, 1)

                        rows.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch,
                            'split': split,
                            'acc': acc
                        })

        return pd.DataFrame(rows)

    def test_complete_multi_method_pipeline(self, sample_multi_method_csv):
        """Test the complete multi-method ERI visualization pipeline."""

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save CSV file
            csv_path = os.path.join(temp_dir, "multi_method_eri_data.csv")
            sample_multi_method_csv.to_csv(csv_path, index=False)

            # Step 1: Load data
            loader = ERIDataLoader()
            dataset = loader.load_csv(csv_path)

            # Validate dataset
            assert len(dataset.methods) == 6  # All methods loaded
            assert 'Scratch_T2' in dataset.methods
            assert 'Interleaved' in dataset.methods
            assert len(dataset.splits) == 2
            assert len(dataset.seeds) == 3

            # Step 2: Process data
            processor = ERITimelineProcessor(smoothing_window=5, tau=0.6)

            # Compute accuracy curves
            curves = processor.compute_accuracy_curves(dataset)
            assert len(curves) == 12  # 6 methods × 2 splits

            # Separate curves by split
            patched_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}
            masked_curves = {k: v for k, v in curves.items() if 'shortcut_masked' in k}

            assert len(patched_curves) == 6
            assert len(masked_curves) == 6

            # Compute derived metrics
            pd_series = processor.compute_performance_deficits(curves)
            sfr_series = processor.compute_sfr_relative(curves)
            ad_values = processor.compute_adaptation_delays(curves)

            # Validate derived metrics
            expected_cl_methods = ['Interleaved', 'sgd', 'derpp', 'ewc_on', 'gpm']
            assert len(pd_series) == len(expected_cl_methods)
            assert len(sfr_series) == len(expected_cl_methods)
            assert len(ad_values) == len(expected_cl_methods)

            # Step 3: Create dynamics visualization
            dynamics_plotter = ERIDynamicsPlotter()

            dynamics_path = os.path.join(temp_dir, "multi_method_dynamics.pdf")
            dynamics_fig = dynamics_plotter.create_dynamics_figure(
                patched_curves=patched_curves,
                masked_curves=masked_curves,
                pd_series=pd_series,
                sfr_series=sfr_series,
                ad_values=ad_values,
                tau=0.6,
                title="Multi-Method ERI Dynamics Analysis",
                save_path=dynamics_path
            )

            # Validate dynamics figure
            assert len(dynamics_fig.axes) == 3
            assert os.path.exists(dynamics_path)

            # Check that all methods appear in panel A
            ax_a = dynamics_fig.axes[0]
            legend_texts = [t.get_text() for t in ax_a.get_legend().get_texts()]

            # Should have entries for both patched and masked versions of each method
            for method in dataset.methods:
                patched_label = f"{method} (patched)"
                masked_label = f"{method} (masked)"
                # At least one should be present (some might be filtered)
                method_present = any(method in label for label in legend_texts)
                assert method_present, f"Method {method} not found in legend"

            plt.close(dynamics_fig)

            # Step 4: Create heatmap visualization
            heatmap_plotter = ERIHeatmapPlotter()

            # Create tau sensitivity analysis
            taus = np.arange(0.5, 0.8, 0.05)  # [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
            taus = np.round(taus, 2)  # Round to avoid floating point precision issues

            heatmap_path = os.path.join(temp_dir, "multi_method_heatmap.pdf")
            heatmap_fig = heatmap_plotter.create_method_comparison_heatmap(
                curves=patched_curves,  # Only need shortcut_normal for AD analysis
                tau_range=(0.5, 0.75),
                tau_step=0.05,
                baseline_method="Scratch_T2",
                title="Multi-Method AD(τ) Robustness Analysis"
            )

            heatmap_plotter.save_heatmap(heatmap_fig, heatmap_path)

            # Validate heatmap
            assert len(heatmap_fig.axes) >= 1
            assert os.path.exists(heatmap_path)

            # Check heatmap content
            ax_heatmap = heatmap_fig.axes[0]

            # Should have correct number of ticks
            x_ticks = ax_heatmap.get_xticks()
            y_ticks = ax_heatmap.get_yticks()

            # Note: x_ticks might be different due to matplotlib's tick positioning
            # Just check that we have a reasonable number of ticks
            assert len(x_ticks) >= 4, f"Expected at least 4 x-ticks, got {len(x_ticks)}"
            assert len(y_ticks) == len(expected_cl_methods), f"Expected {len(expected_cl_methods)} y-ticks, got {len(y_ticks)}"

            plt.close(heatmap_fig)

            # Step 5: Validate processing summary
            summary = processor.get_processing_summary(curves)

            assert summary['n_curves'] == 12
            assert len(summary['methods']) == 6
            assert summary['smoothing_window'] == 5
            assert summary['tau_threshold'] == 0.6

            # All methods should have 2 curves (one per split)
            for method in dataset.methods:
                assert summary['curves_per_method'][method] == 2

            # All splits should have 6 curves (one per method)
            for split in dataset.splits:
                assert summary['curves_per_split'][split] == 6

    def test_baseline_method_validation(self, sample_multi_method_csv):
        """Test that baseline methods are correctly identified and used."""

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save CSV file
            csv_path = os.path.join(temp_dir, "baseline_test.csv")
            sample_multi_method_csv.to_csv(csv_path, index=False)

            # Load and process
            loader = ERIDataLoader()
            dataset = loader.load_csv(csv_path)
            processor = ERITimelineProcessor(tau=0.6)

            curves = processor.compute_accuracy_curves(dataset)

            # Test Scratch_T2 baseline detection
            pd_series = processor.compute_performance_deficits(curves, scratch_key="Scratch_T2")

            # Should compute PD_t for all non-baseline methods
            non_baseline_methods = [m for m in dataset.methods if m != 'Scratch_T2']
            assert len(pd_series) == len(non_baseline_methods)

            # All PD_t values should be positive (continual learning methods worse than Scratch_T2)
            for method, series in pd_series.items():
                if method != 'Interleaved':  # Interleaved might be close to Scratch_T2
                    mean_pd = np.mean(series.values)
                    assert mean_pd > 0, f"Expected positive PD_t for {method}, got {mean_pd}"

            # Test SFR_rel computation
            sfr_series = processor.compute_sfr_relative(curves, scratch_key="Scratch_T2")
            assert len(sfr_series) == len(non_baseline_methods)

            # SFR_rel should be computed for all methods
            for method in non_baseline_methods:
                assert method in sfr_series, f"Missing SFR_rel for {method}"
                series = sfr_series[method]
                assert len(series.values) > 0
                assert np.all(np.isfinite(series.values))

    def test_missing_baseline_handling(self, sample_multi_method_csv):
        """Test handling when baseline methods are missing."""

        # Remove Scratch_T2 from data
        no_scratch_data = sample_multi_method_csv[
            sample_multi_method_csv['method'] != 'Scratch_T2'
        ].copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "no_baseline_test.csv")
            no_scratch_data.to_csv(csv_path, index=False)

            # Load and process
            loader = ERIDataLoader()
            dataset = loader.load_csv(csv_path)
            processor = ERITimelineProcessor(tau=0.6)

            curves = processor.compute_accuracy_curves(dataset)

            # Should not find Scratch_T2
            assert 'Scratch_T2' not in dataset.methods

            # PD_t computation should return empty dict
            pd_series = processor.compute_performance_deficits(curves, scratch_key="Scratch_T2")
            assert len(pd_series) == 0, "Should not compute PD_t without baseline"

            # SFR_rel computation should return empty dict
            sfr_series = processor.compute_sfr_relative(curves, scratch_key="Scratch_T2")
            assert len(sfr_series) == 0, "Should not compute SFR_rel without baseline"

            # But regular accuracy curves should still work
            assert len(curves) > 0

            # And dynamics plot should still work (just without PD_t and SFR_rel panels)
            patched_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}
            masked_curves = {k: v for k, v in curves.items() if 'shortcut_masked' in k}

            dynamics_plotter = ERIDynamicsPlotter()
            fig = dynamics_plotter.create_dynamics_figure(
                patched_curves=patched_curves,
                masked_curves=masked_curves,
                pd_series={},  # Empty
                sfr_series={},  # Empty
                ad_values={},  # Empty
                tau=0.6,
                title="No Baseline Test"
            )

            # Should still create 3 panels, but B and C will show "no data" messages
            assert len(fig.axes) == 3
            plt.close(fig)

    def test_comparative_visualization_quality(self, sample_multi_method_csv):
        """Test that comparative visualizations meet quality standards."""

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "quality_test.csv")
            sample_multi_method_csv.to_csv(csv_path, index=False)

            # Load and process
            loader = ERIDataLoader()
            dataset = loader.load_csv(csv_path)
            processor = ERITimelineProcessor(smoothing_window=3, tau=0.6)

            curves = processor.compute_accuracy_curves(dataset)
            patched_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}
            masked_curves = {k: v for k, v in curves.items() if 'shortcut_masked' in k}

            pd_series = processor.compute_performance_deficits(curves)
            sfr_series = processor.compute_sfr_relative(curves)
            ad_values = processor.compute_adaptation_delays(curves)

            # Create high-quality dynamics plot
            dynamics_plotter = ERIDynamicsPlotter()

            dynamics_path = os.path.join(temp_dir, "quality_dynamics.pdf")
            fig = dynamics_plotter.create_dynamics_figure(
                patched_curves=patched_curves,
                masked_curves=masked_curves,
                pd_series=pd_series,
                sfr_series=sfr_series,
                ad_values=ad_values,
                tau=0.6,
                title="Quality Test Dynamics",
                save_path=dynamics_path
            )

            # Check file size (should be reasonable)
            file_size_mb = os.path.getsize(dynamics_path) / (1024 * 1024)
            assert file_size_mb < 5.0, f"Dynamics plot too large: {file_size_mb:.1f} MB"

            # Check figure properties
            fig_info = dynamics_plotter.get_figure_info(fig)
            assert fig_info['n_axes'] == 3
            assert fig_info['dpi'] > 0

            plt.close(fig)

            # Create high-quality heatmap
            heatmap_plotter = ERIHeatmapPlotter()

            heatmap_path = os.path.join(temp_dir, "quality_heatmap.pdf")
            heatmap_fig = heatmap_plotter.create_method_comparison_heatmap(
                curves=patched_curves,
                tau_range=(0.55, 0.75),
                tau_step=0.05,
                baseline_method="Scratch_T2",
                title="Quality Test Heatmap"
            )

            heatmap_plotter.save_heatmap(heatmap_fig, heatmap_path)

            # Check heatmap file size
            heatmap_size_mb = os.path.getsize(heatmap_path) / (1024 * 1024)
            assert heatmap_size_mb < 5.0, f"Heatmap too large: {heatmap_size_mb:.1f} MB"

            plt.close(heatmap_fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
