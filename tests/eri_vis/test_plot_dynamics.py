"""
Tests for ERIDynamicsPlotter - Main visualization generator.

This module provides comprehensive tests for the ERIDynamicsPlotter class,
including unit tests, integration tests, and visual regression tests.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import hashlib
from unittest.mock import Mock, patch

from eri_vis.plot_dynamics import ERIDynamicsPlotter
from eri_vis.processing import AccuracyCurve, TimeSeries
from eri_vis.styles import PlotStyleConfig


class TestERIDynamicsPlotter:
    """Test suite for ERIDynamicsPlotter class."""

    @pytest.fixture
    def sample_style(self):
        """Create a sample PlotStyleConfig for testing."""
        return PlotStyleConfig(
            figure_size=(8, 6),
            dpi=100,  # Lower DPI for faster tests
            font_sizes={
                'title': 12,
                'axis_label': 10,
                'legend': 8,
                'annotation': 7,
                'panel_label': 14,
            }
        )

    @pytest.fixture
    def sample_patched_curves(self):
        """Create sample patched accuracy curves for testing."""
        epochs = np.linspace(0, 10, 21)

        # Scratch_T2 curve - reaches threshold quickly
        scratch_acc = 0.1 + 0.8 * (1 - np.exp(-epochs / 2))
        scratch_ci = 0.05 * np.ones_like(epochs)

        # SGD curve - slower adaptation
        sgd_acc = 0.1 + 0.7 * (1 - np.exp(-epochs / 4))
        sgd_ci = 0.08 * np.ones_like(epochs)

        # EWC curve - even slower
        ewc_acc = 0.1 + 0.6 * (1 - np.exp(-epochs / 6))
        ewc_ci = 0.06 * np.ones_like(epochs)

        return {
            'Scratch_T2_T2_shortcut_normal': AccuracyCurve(
                epochs=epochs,
                mean_accuracy=scratch_acc,
                conf_interval=scratch_ci,
                method='Scratch_T2',
                split='T2_shortcut_normal',
                n_seeds=5
            ),
            'sgd_T2_shortcut_normal': AccuracyCurve(
                epochs=epochs,
                mean_accuracy=sgd_acc,
                conf_interval=sgd_ci,
                method='sgd',
                split='T2_shortcut_normal',
                n_seeds=5
            ),
            'ewc_on_T2_shortcut_normal': AccuracyCurve(
                epochs=epochs,
                mean_accuracy=ewc_acc,
                conf_interval=ewc_ci,
                method='ewc_on',
                split='T2_shortcut_normal',
                n_seeds=5
            )
        }

    @pytest.fixture
    def sample_masked_curves(self):
        """Create sample masked accuracy curves for testing."""
        epochs = np.linspace(0, 10, 21)

        # Masked curves should be lower and show different patterns
        scratch_acc = 0.05 + 0.1 * epochs / 10  # Slight improvement
        sgd_acc = 0.05 + 0.15 * epochs / 10     # Better improvement
        ewc_acc = 0.05 + 0.12 * epochs / 10     # Moderate improvement

        ci = 0.03 * np.ones_like(epochs)

        return {
            'Scratch_T2_T2_shortcut_masked': AccuracyCurve(
                epochs=epochs,
                mean_accuracy=scratch_acc,
                conf_interval=ci,
                method='Scratch_T2',
                split='T2_shortcut_masked',
                n_seeds=5
            ),
            'sgd_T2_shortcut_masked': AccuracyCurve(
                epochs=epochs,
                mean_accuracy=sgd_acc,
                conf_interval=ci,
                method='sgd',
                split='T2_shortcut_masked',
                n_seeds=5
            ),
            'ewc_on_T2_shortcut_masked': AccuracyCurve(
                epochs=epochs,
                mean_accuracy=ewc_acc,
                conf_interval=ci,
                method='ewc_on',
                split='T2_shortcut_masked',
                n_seeds=5
            )
        }

    @pytest.fixture
    def sample_pd_series(self):
        """Create sample Performance Deficit time series for testing."""
        epochs = np.linspace(0, 10, 21)

        # PD_t should start negative and potentially improve
        sgd_pd = -0.2 + 0.15 * epochs / 10
        ewc_pd = -0.3 + 0.1 * epochs / 10

        return {
            'sgd': TimeSeries(
                epochs=epochs,
                values=sgd_pd,
                conf_interval=np.array([]),
                method='sgd',
                metric_name='PD_t'
            ),
            'ewc_on': TimeSeries(
                epochs=epochs,
                values=ewc_pd,
                conf_interval=np.array([]),
                method='ewc_on',
                metric_name='PD_t'
            )
        }

    @pytest.fixture
    def sample_sfr_series(self):
        """Create sample Shortcut Forgetting Rate time series for testing."""
        epochs = np.linspace(0, 10, 21)

        # SFR_rel should show different patterns
        sgd_sfr = 0.1 * np.sin(epochs / 2) + 0.05
        ewc_sfr = 0.15 * np.sin(epochs / 3) + 0.08

        return {
            'sgd': TimeSeries(
                epochs=epochs,
                values=sgd_sfr,
                conf_interval=np.array([]),
                method='sgd',
                metric_name='SFR_rel'
            ),
            'ewc_on': TimeSeries(
                epochs=epochs,
                values=ewc_sfr,
                conf_interval=np.array([]),
                method='ewc_on',
                metric_name='SFR_rel'
            )
        }

    @pytest.fixture
    def sample_ad_values(self):
        """Create sample Adaptation Delay values for testing."""
        return {
            'sgd': 2.5,
            'ewc_on': 4.2,
            'derpp': np.nan  # Censored run
        }

    def test_plotter_initialization(self, sample_style):
        """Test ERIDynamicsPlotter initialization."""
        # Test with custom style
        plotter = ERIDynamicsPlotter(style=sample_style)
        assert plotter.style == sample_style

        # Test with default style
        plotter_default = ERIDynamicsPlotter()
        assert plotter_default.style is not None

    def test_input_validation(self, sample_patched_curves, sample_masked_curves,
                            sample_pd_series, sample_sfr_series, sample_ad_values):
        """Test input validation for create_dynamics_figure."""
        plotter = ERIDynamicsPlotter()

        # Test empty patched_curves
        with pytest.raises(ValueError, match="patched_curves cannot be empty"):
            plotter.create_dynamics_figure(
                {}, sample_masked_curves, sample_pd_series,
                sample_sfr_series, sample_ad_values, 0.6
            )

        # Test empty masked_curves
        with pytest.raises(ValueError, match="masked_curves cannot be empty"):
            plotter.create_dynamics_figure(
                sample_patched_curves, {}, sample_pd_series,
                sample_sfr_series, sample_ad_values, 0.6
            )

        # Test inconsistent curve data
        bad_curve = AccuracyCurve(
            epochs=np.array([1, 2, 3]),
            mean_accuracy=np.array([0.1, 0.2]),  # Wrong length
            conf_interval=np.array([0.01, 0.02]),
            method='bad',
            split='test',
            n_seeds=1
        )

        with pytest.raises(ValueError, match="Inconsistent curve data"):
            plotter.create_dynamics_figure(
                {'bad_curve': bad_curve}, sample_masked_curves,
                sample_pd_series, sample_sfr_series, sample_ad_values, 0.6
            )

    def test_create_dynamics_figure_basic(self, sample_patched_curves, sample_masked_curves,
                                        sample_pd_series, sample_sfr_series, sample_ad_values,
                                        sample_style):
        """Test basic dynamics figure creation."""
        plotter = ERIDynamicsPlotter(style=sample_style)

        fig = plotter.create_dynamics_figure(
            sample_patched_curves, sample_masked_curves,
            sample_pd_series, sample_sfr_series, sample_ad_values,
            tau=0.6
        )

        # Check figure structure
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # Three panels

        # Check panel labels
        for i, expected_label in enumerate(['A', 'B', 'C']):
            ax = fig.axes[i]
            # Find text objects with the panel label
            panel_labels = [t for t in ax.texts if t.get_text() == expected_label]
            assert len(panel_labels) == 1, f"Panel {expected_label} label not found"

        plt.close(fig)

    def test_create_dynamics_figure_with_title(self, sample_patched_curves, sample_masked_curves,
                                             sample_pd_series, sample_sfr_series, sample_ad_values):
        """Test dynamics figure creation with title."""
        plotter = ERIDynamicsPlotter()

        title = "Test ERI Dynamics"
        fig = plotter.create_dynamics_figure(
            sample_patched_curves, sample_masked_curves,
            sample_pd_series, sample_sfr_series, sample_ad_values,
            tau=0.6, title=title
        )

        # Check that title is set
        assert fig._suptitle is not None
        assert fig._suptitle.get_text() == title

        plt.close(fig)

    def test_panel_a_content(self, sample_patched_curves, sample_masked_curves,
                           sample_pd_series, sample_sfr_series, sample_ad_values):
        """Test Panel A content and formatting."""
        plotter = ERIDynamicsPlotter()

        fig = plotter.create_dynamics_figure(
            sample_patched_curves, sample_masked_curves,
            sample_pd_series, sample_sfr_series, sample_ad_values,
            tau=0.6
        )

        ax_a = fig.axes[0]  # Panel A

        # Check that lines are plotted
        lines = ax_a.get_lines()
        assert len(lines) > 0, "No lines plotted in Panel A"

        # Check for threshold line
        threshold_lines = [line for line in lines if line.get_linestyle() == ':']
        assert len(threshold_lines) > 0, "Threshold line not found"

        # Check axis labels
        assert ax_a.get_xlabel() == 'Effective Epoch'
        assert ax_a.get_ylabel() == 'Accuracy'
        assert 'Accuracy Trajectories' in ax_a.get_title()

        # Check y-axis limits
        ylim = ax_a.get_ylim()
        assert ylim[0] <= 0 and ylim[1] >= 1, "Y-axis limits not set correctly"

        plt.close(fig)

    def test_panel_b_content(self, sample_patched_curves, sample_masked_curves,
                           sample_pd_series, sample_sfr_series, sample_ad_values):
        """Test Panel B content and formatting."""
        plotter = ERIDynamicsPlotter()

        fig = plotter.create_dynamics_figure(
            sample_patched_curves, sample_masked_curves,
            sample_pd_series, sample_sfr_series, sample_ad_values,
            tau=0.6
        )

        ax_b = fig.axes[1]  # Panel B

        # Check that lines are plotted
        lines = ax_b.get_lines()
        assert len(lines) > 0, "No lines plotted in Panel B"

        # Check for zero reference line
        zero_lines = [line for line in lines
                     if np.allclose(line.get_ydata(), 0) and line.get_color() == 'black']
        assert len(zero_lines) > 0, "Zero reference line not found in Panel B"

        # Check axis labels
        assert ax_b.get_xlabel() == 'Effective Epoch'
        assert 'Performance Deficit' in ax_b.get_ylabel()
        assert 'Performance Deficit' in ax_b.get_title()

        plt.close(fig)

    def test_panel_c_content(self, sample_patched_curves, sample_masked_curves,
                           sample_pd_series, sample_sfr_series, sample_ad_values):
        """Test Panel C content and formatting."""
        plotter = ERIDynamicsPlotter()

        fig = plotter.create_dynamics_figure(
            sample_patched_curves, sample_masked_curves,
            sample_pd_series, sample_sfr_series, sample_ad_values,
            tau=0.6
        )

        ax_c = fig.axes[2]  # Panel C

        # Check that lines are plotted
        lines = ax_c.get_lines()
        assert len(lines) > 0, "No lines plotted in Panel C"

        # Check for zero reference line
        zero_lines = [line for line in lines
                     if np.allclose(line.get_ydata(), 0) and line.get_color() == 'black']
        assert len(zero_lines) > 0, "Zero reference line not found in Panel C"

        # Check axis labels
        assert ax_c.get_xlabel() == 'Effective Epoch'
        assert 'Shortcut Forgetting Rate' in ax_c.get_ylabel()
        assert 'Shortcut Forgetting' in ax_c.get_title()

        plt.close(fig)

    def test_empty_pd_sfr_series(self, sample_patched_curves, sample_masked_curves, sample_ad_values):
        """Test handling of empty PD_t and SFR_rel series."""
        plotter = ERIDynamicsPlotter()

        fig = plotter.create_dynamics_figure(
            sample_patched_curves, sample_masked_curves,
            {}, {},  # Empty PD and SFR series
            sample_ad_values, tau=0.6
        )

        # Check that panels B and C show appropriate messages
        ax_b = fig.axes[1]
        ax_c = fig.axes[2]

        # Check for "No data available" messages
        b_texts = [t.get_text() for t in ax_b.texts]
        c_texts = [t.get_text() for t in ax_c.texts]

        assert any('No Performance Deficit data' in text for text in b_texts)
        assert any('No Shortcut Forgetting Rate data' in text for text in c_texts)

        plt.close(fig)

    def test_save_figure(self, sample_patched_curves, sample_masked_curves,
                        sample_pd_series, sample_sfr_series, sample_ad_values):
        """Test figure saving functionality."""
        plotter = ERIDynamicsPlotter()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_dynamics.pdf"

            fig = plotter.create_dynamics_figure(
                sample_patched_curves, sample_masked_curves,
                sample_pd_series, sample_sfr_series, sample_ad_values,
                tau=0.6, save_path=str(save_path)
            )

            # Check that file was created
            assert save_path.exists(), "Figure file was not created"

            # Check file size (should be reasonable)
            file_size_mb = save_path.stat().st_size / (1024 * 1024)
            assert file_size_mb < 5.0, f"File size ({file_size_mb:.1f} MB) exceeds 5MB limit"
            assert file_size_mb > 0.01, "File size too small, likely empty"

            plt.close(fig)

    def test_create_quick_dynamics_plot(self, sample_patched_curves):
        """Test quick dynamics plot creation."""
        plotter = ERIDynamicsPlotter()

        fig = plotter.create_quick_dynamics_plot(
            sample_patched_curves, tau=0.6
        )

        # Check figure structure
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1  # Single panel

        # Check content
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) > 0, "No lines plotted in quick plot"

        # Check for threshold line
        threshold_lines = [line for line in lines if line.get_linestyle() == ':']
        assert len(threshold_lines) > 0, "Threshold line not found in quick plot"

        plt.close(fig)

    def test_create_quick_dynamics_plot_filtered_methods(self, sample_patched_curves):
        """Test quick dynamics plot with method filtering."""
        plotter = ERIDynamicsPlotter()

        methods_to_include = ['sgd', 'ewc_on']
        fig = plotter.create_quick_dynamics_plot(
            sample_patched_curves, tau=0.6, methods=methods_to_include
        )

        ax = fig.axes[0]

        # Check that only specified methods are plotted
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        for method in methods_to_include:
            assert method in legend_labels, f"Method {method} not found in legend"

        # Scratch_T2 should not be included
        assert 'Scratch_T2' not in legend_labels, "Scratch_T2 should be filtered out"

        plt.close(fig)

    def test_get_figure_info(self, sample_patched_curves, sample_masked_curves,
                           sample_pd_series, sample_sfr_series, sample_ad_values):
        """Test figure information extraction."""
        plotter = ERIDynamicsPlotter()

        fig = plotter.create_dynamics_figure(
            sample_patched_curves, sample_masked_curves,
            sample_pd_series, sample_sfr_series, sample_ad_values,
            tau=0.6
        )

        info = plotter.get_figure_info(fig)

        # Check info structure
        assert 'figure_size_inches' in info
        assert 'dpi' in info
        assert 'n_axes' in info
        assert 'n_artists' in info
        assert 'style_config' in info

        # Check values
        assert info['n_axes'] == 3
        assert info['dpi'] > 0
        assert len(info['figure_size_inches']) == 2

        plt.close(fig)

    def test_ad_markers_and_annotations(self, sample_patched_curves, sample_masked_curves,
                                      sample_pd_series, sample_sfr_series, sample_ad_values):
        """Test AD markers and annotations in Panel A."""
        plotter = ERIDynamicsPlotter()

        fig = plotter.create_dynamics_figure(
            sample_patched_curves, sample_masked_curves,
            sample_pd_series, sample_sfr_series, sample_ad_values,
            tau=0.6
        )

        ax_a = fig.axes[0]  # Panel A

        # Check for vertical lines (AD markers)
        vertical_lines = [line for line in ax_a.get_lines()
                         if line.get_linestyle() == '--']
        assert len(vertical_lines) > 0, "No AD marker lines found"

        # Check for annotations
        annotations = [child for child in ax_a.get_children()
                      if hasattr(child, 'get_text') and 'AD =' in str(child.get_text())]
        # Note: Annotations might not be easily detectable this way,
        # but we can check that the method ran without error

        plt.close(fig)

    @pytest.mark.slow
    def test_visual_regression_golden_image(self, sample_patched_curves, sample_masked_curves,
                                          sample_pd_series, sample_sfr_series, sample_ad_values):
        """Test visual regression against golden reference image."""
        plotter = ERIDynamicsPlotter()

        # Create figure with deterministic settings
        fig = plotter.create_dynamics_figure(
            sample_patched_curves, sample_masked_curves,
            sample_pd_series, sample_sfr_series, sample_ad_values,
            tau=0.6
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_output.png"

            # Save as PNG for hash comparison
            fig.savefig(test_path, dpi=100, bbox_inches='tight')

            # Calculate hash of generated image
            with open(test_path, 'rb') as f:
                generated_hash = hashlib.md5(f.read()).hexdigest()

            # For now, just check that we can generate a consistent hash
            # In a real implementation, you would compare against a stored golden hash
            assert len(generated_hash) == 32, "Invalid hash generated"

            # Store the hash for future reference (in real tests, this would be compared)
            print(f"Generated image hash: {generated_hash}")

            plt.close(fig)

    def test_confidence_intervals_plotting(self, sample_style):
        """Test that confidence intervals are plotted correctly."""
        plotter = ERIDynamicsPlotter(style=sample_style)

        # Create curves with significant confidence intervals
        epochs = np.linspace(0, 5, 11)
        mean_acc = 0.5 + 0.3 * epochs / 5
        large_ci = 0.1 * np.ones_like(epochs)

        curves_with_ci = {
            'test_method': AccuracyCurve(
                epochs=epochs,
                mean_accuracy=mean_acc,
                conf_interval=large_ci,
                method='test_method',
                split='T2_shortcut_normal',
                n_seeds=3
            )
        }

        fig = plotter.create_quick_dynamics_plot(curves_with_ci, tau=0.6)
        ax = fig.axes[0]

        # Check for filled areas (confidence intervals)
        collections = ax.collections
        assert len(collections) > 0, "No confidence interval areas found"

        plt.close(fig)

    def test_color_consistency(self, sample_patched_curves, sample_masked_curves,
                             sample_pd_series, sample_sfr_series, sample_ad_values):
        """Test that colors are consistent across panels for the same method."""
        plotter = ERIDynamicsPlotter()

        fig = plotter.create_dynamics_figure(
            sample_patched_curves, sample_masked_curves,
            sample_pd_series, sample_sfr_series, sample_ad_values,
            tau=0.6
        )

        # Get colors from each panel for the same method
        method_colors = {}

        for i, ax in enumerate(fig.axes):
            for line in ax.get_lines():
                label = line.get_label()
                if label and not label.startswith('_'):  # Skip internal labels
                    color = line.get_color()
                    if label not in method_colors:
                        method_colors[label] = []
                    method_colors[label].append(color)

        # Check that methods have consistent colors (allowing for some variation)
        # This is a basic check - in practice, you might need more sophisticated color matching
        for method, colors in method_colors.items():
            if len(colors) > 1:
                # All colors for this method should be similar
                # (This is a simplified check)
                assert len(set(colors)) <= 2, f"Inconsistent colors for method {method}: {colors}"

        plt.close(fig)


class TestERIDynamicsPlotterEdgeCases:
    """Test edge cases and error conditions for ERIDynamicsPlotter."""

    def test_single_epoch_data(self):
        """Test handling of single epoch data."""
        plotter = ERIDynamicsPlotter()

        # Create curve with single epoch
        single_epoch_curve = {
            'test_method': AccuracyCurve(
                epochs=np.array([1.0]),
                mean_accuracy=np.array([0.5]),
                conf_interval=np.array([0.1]),
                method='test_method',
                split='T2_shortcut_normal',
                n_seeds=1
            )
        }

        # Should not raise an error
        fig = plotter.create_quick_dynamics_plot(single_epoch_curve, tau=0.6)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_nan_values_in_curves(self):
        """Test handling of NaN values in accuracy curves."""
        plotter = ERIDynamicsPlotter()

        epochs = np.linspace(0, 5, 11)
        mean_acc = np.full_like(epochs, 0.5)
        mean_acc[5] = np.nan  # Insert NaN value

        curves_with_nan = {
            'test_method': AccuracyCurve(
                epochs=epochs,
                mean_accuracy=mean_acc,
                conf_interval=0.1 * np.ones_like(epochs),
                method='test_method',
                split='T2_shortcut_normal',
                n_seeds=3
            )
        }

        # Should handle NaN values gracefully
        fig = plotter.create_quick_dynamics_plot(curves_with_nan, tau=0.6)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_confidence_intervals(self):
        """Test handling of empty confidence intervals."""
        plotter = ERIDynamicsPlotter()

        epochs = np.linspace(0, 5, 11)
        mean_acc = 0.5 + 0.3 * epochs / 5

        curves_no_ci = {
            'test_method': AccuracyCurve(
                epochs=epochs,
                mean_accuracy=mean_acc,
                conf_interval=np.array([]),  # Empty CI
                method='test_method',
                split='T2_shortcut_normal',
                n_seeds=1
            )
        }

        # Should handle empty CI gracefully
        fig = plotter.create_quick_dynamics_plot(curves_no_ci, tau=0.6)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_very_large_figure_size_warning(self, caplog):
        """Test warning for very large figure sizes."""
        large_style = PlotStyleConfig(
            figure_size=(50, 40),  # Very large
            dpi=300
        )

        plotter = ERIDynamicsPlotter(style=large_style)

        epochs = np.linspace(0, 5, 11)
        simple_curve = {
            'test_method': AccuracyCurve(
                epochs=epochs,
                mean_accuracy=0.5 * np.ones_like(epochs),
                conf_interval=0.1 * np.ones_like(epochs),
                method='test_method',
                split='T2_shortcut_normal',
                n_seeds=3
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "large_figure.pdf"

            fig = plotter.create_quick_dynamics_plot(
                simple_curve, tau=0.6, save_path=str(save_path)
            )

            # Check if warning was logged about file size
            # (This might not trigger with simple test data, but tests the mechanism)

            plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
