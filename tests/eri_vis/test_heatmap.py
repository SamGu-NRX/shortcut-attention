"""
Tests for ERIHeatmapPlotter - Robustness Analysis Visualization.

This module tests the ERIHeatmapPlotter class functionality including:
- Tau sensitivity computation
- Heatmap generation with synthetic data
- NaN handling for censored runs
- Cell annotations and visual styling
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
import tempfile
import os

from eri_vis.plot_heatmap import ERIHeatmapPlotter, TauSensitivityResult
from eri_vis.processing import AccuracyCurve
from eri_vis.styles import PlotStyleConfig


class TestTauSensitivityResult:
    """Test TauSensitivityResult dataclass."""

    def test_initialization(self):
        """Test basic initialization of TauSensitivityResult."""
        methods = ['sgd', 'ewc_on']
        taus = np.array([0.5, 0.6, 0.7])
        ad_matrix = np.array([[1.0, 2.0, np.nan], [-0.5, 0.0, 1.5]])

        result = TauSensitivityResult(
            methods=methods,
            taus=taus,
            ad_matrix=ad_matrix
        )

        assert result.methods == methods
        np.testing.assert_array_equal(result.taus, taus)
        # Test array equality with NaN handling
        assert result.ad_matrix.shape == ad_matrix.shape
        for i in range(ad_matrix.shape[0]):
            for j in range(ad_matrix.shape[1]):
                if np.isnan(ad_matrix[i, j]):
                    assert np.isnan(result.ad_matrix[i, j])
                else:
                    assert result.ad_matrix[i, j] == ad_matrix[i, j]
        assert result.baseline_method == "Scratch_T2"
        assert result.n_censored is None

    def test_with_censored_counts(self):
        """Test initialization with censored counts."""
        methods = ['sgd', 'ewc_on']
        taus = np.array([0.5, 0.6])
        ad_matrix = np.array([[1.0, np.nan], [0.5, 1.0]])
        n_censored = {'sgd': 1, 'ewc_on': 0}

        result = TauSensitivityResult(
            methods=methods,
            taus=taus,
            ad_matrix=ad_matrix,
            n_censored=n_censored
        )

        assert result.n_censored == n_censored


class TestERIHeatmapPlotter:
    """Test ERIHeatmapPlotter class."""

    @pytest.fixture
    def plotter(self):
        """Create ERIHeatmapPlotter instance."""
        return ERIHeatmapPlotter()

    @pytest.fixture
    def custom_style_plotter(self):
        """Create ERIHeatmapPlotter with custom style."""
        style = PlotStyleConfig(
            figure_size=(8, 6),
            dpi=150,
            font_sizes={'annotation': 8}
        )
        return ERIHeatmapPlotter(style=style)

    @pytest.fixture
    def sample_curves(self):
        """Create sample AccuracyCurve objects for testing."""
        epochs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Scratch_T2 baseline - reaches threshold at epoch 2
        scratch_acc = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        scratch_curve = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=scratch_acc,
            conf_interval=np.zeros_like(scratch_acc),
            method="Scratch_T2",
            split="T2_shortcut_normal",
            n_seeds=5
        )

        # SGD method - reaches threshold at epoch 3
        sgd_acc = np.array([0.2, 0.4, 0.55, 0.65, 0.85])
        sgd_curve = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=sgd_acc,
            conf_interval=np.zeros_like(sgd_acc),
            method="sgd",
            split="T2_shortcut_normal",
            n_seeds=5
        )

        # EWC method - never reaches threshold (censored)
        ewc_acc = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
        ewc_curve = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=ewc_acc,
            conf_interval=np.zeros_like(ewc_acc),
            method="ewc_on",
            split="T2_shortcut_normal",
            n_seeds=5
        )

        return {
            "Scratch_T2_T2_shortcut_normal": scratch_curve,
            "sgd_T2_shortcut_normal": sgd_curve,
            "ewc_on_T2_shortcut_normal": ewc_curve
        }

    def test_initialization_default_style(self, plotter):
        """Test plotter initialization with default style."""
        assert plotter.style is not None
        assert hasattr(plotter, 'logger')

    def test_initialization_custom_style(self, custom_style_plotter):
        """Test plotter initialization with custom style."""
        assert custom_style_plotter.style.figure_size == (8, 6)
        assert custom_style_plotter.style.dpi == 150

    def test_find_threshold_crossing_success(self, plotter, sample_curves):
        """Test successful threshold crossing detection."""
        curve = sample_curves["Scratch_T2_T2_shortcut_normal"]
        crossing = plotter._find_threshold_crossing(curve, 0.6)
        assert crossing == 2.0  # Should cross at epoch 2

    def test_find_threshold_crossing_no_crossing(self, plotter, sample_curves):
        """Test threshold crossing when no crossing occurs."""
        curve = sample_curves["ewc_on_T2_shortcut_normal"]
        crossing = plotter._find_threshold_crossing(curve, 0.8)
        assert np.isnan(crossing)  # Should be NaN (censored)

    def test_find_threshold_crossing_empty_curve(self, plotter):
        """Test threshold crossing with empty curve."""
        empty_curve = AccuracyCurve(
            epochs=np.array([]),
            mean_accuracy=np.array([]),
            conf_interval=np.array([]),
            method="empty",
            split="T2_shortcut_normal",
            n_seeds=0
        )
        crossing = plotter._find_threshold_crossing(empty_curve, 0.6)
        assert np.isnan(crossing)

    def test_compute_tau_sensitivity_success(self, plotter, sample_curves):
        """Test successful tau sensitivity computation."""
        taus = [0.5, 0.6, 0.7]
        result = plotter.compute_tau_sensitivity(sample_curves, taus)

        assert result.methods == ['ewc_on', 'sgd']  # Sorted, excluding baseline
        np.testing.assert_array_equal(result.taus, np.array(taus))
        assert result.baseline_method == "Scratch_T2"
        assert result.ad_matrix.shape == (2, 3)

        # Check specific values
        # SGD crosses at epoch 3, Scratch at epoch 2 for tau=0.6
        # So AD should be 3 - 2 = 1.0
        sgd_idx = result.methods.index('sgd')
        tau_06_idx = list(result.taus).index(0.6)
        assert result.ad_matrix[sgd_idx, tau_06_idx] == 1.0

        # EWC never crosses, so should be NaN
        ewc_idx = result.methods.index('ewc_on')
        assert np.isnan(result.ad_matrix[ewc_idx, tau_06_idx])

    def test_compute_tau_sensitivity_no_baseline(self, plotter):
        """Test tau sensitivity computation without baseline method."""
        curves = {
            "sgd_T2_shortcut_normal": AccuracyCurve(
                epochs=np.array([0, 1, 2]),
                mean_accuracy=np.array([0.1, 0.5, 0.8]),
                conf_interval=np.zeros(3),
                method="sgd",
                split="T2_shortcut_normal",
                n_seeds=5
            )
        }

        with pytest.raises(ValueError, match="Baseline method 'Scratch_T2' not found"):
            plotter.compute_tau_sensitivity(curves, [0.6])

    def test_compute_tau_sensitivity_no_cl_methods(self, plotter):
        """Test tau sensitivity computation with only baseline method."""
        curves = {
            "Scratch_T2_T2_shortcut_normal": AccuracyCurve(
                epochs=np.array([0, 1, 2]),
                mean_accuracy=np.array([0.1, 0.5, 0.8]),
                conf_interval=np.zeros(3),
                method="Scratch_T2",
                split="T2_shortcut_normal",
                n_seeds=5
            )
        }

        with pytest.raises(ValueError, match="No continual learning methods found"):
            plotter.compute_tau_sensitivity(curves, [0.6])

    def test_compute_tau_sensitivity_empty_curves(self, plotter):
        """Test tau sensitivity computation with empty curves."""
        with pytest.raises(ValueError, match="No curves provided"):
            plotter.compute_tau_sensitivity({}, [0.6])

    def test_compute_tau_sensitivity_empty_taus(self, plotter, sample_curves):
        """Test tau sensitivity computation with empty tau list."""
        with pytest.raises(ValueError, match="No tau values provided"):
            plotter.compute_tau_sensitivity(sample_curves, [])

    def test_create_tau_sensitivity_heatmap_success(self, plotter):
        """Test successful heatmap creation."""
        # Create synthetic sensitivity result
        methods = ['sgd', 'ewc_on']
        taus = np.array([0.5, 0.6, 0.7])
        ad_matrix = np.array([
            [1.0, 1.5, 2.0],    # sgd
            [np.nan, -0.5, 0.0] # ewc_on
        ])
        n_censored = {'sgd': 0, 'ewc_on': 1}

        result = TauSensitivityResult(
            methods=methods,
            taus=taus,
            ad_matrix=ad_matrix,
            n_censored=n_censored
        )

        fig = plotter.create_tau_sensitivity_heatmap(result)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1  # At least one axis (heatmap)

        # Check that colorbar was added
        assert len(fig.axes) == 2  # Heatmap + colorbar

        plt.close(fig)

    def test_create_tau_sensitivity_heatmap_all_nan(self, plotter):
        """Test heatmap creation with all NaN values."""
        methods = ['sgd', 'ewc_on']
        taus = np.array([0.5, 0.6])
        ad_matrix = np.full((2, 2), np.nan)

        result = TauSensitivityResult(
            methods=methods,
            taus=taus,
            ad_matrix=ad_matrix
        )

        # Should not raise error, but log warning
        with patch.object(plotter.logger, 'warning') as mock_warning:
            fig = plotter.create_tau_sensitivity_heatmap(result)
            mock_warning.assert_called_once()

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_tau_sensitivity_heatmap_empty_matrix(self, plotter):
        """Test heatmap creation with empty matrix."""
        result = TauSensitivityResult(
            methods=[],
            taus=np.array([]),
            ad_matrix=np.array([]).reshape(0, 0)
        )

        with pytest.raises(ValueError, match="Empty sensitivity matrix"):
            plotter.create_tau_sensitivity_heatmap(result)

    def test_create_tau_sensitivity_heatmap_custom_title(self, plotter):
        """Test heatmap creation with custom title."""
        methods = ['sgd']
        taus = np.array([0.6])
        ad_matrix = np.array([[1.0]])

        result = TauSensitivityResult(
            methods=methods,
            taus=taus,
            ad_matrix=ad_matrix
        )

        custom_title = "Custom Heatmap Title"
        fig = plotter.create_tau_sensitivity_heatmap(result, title=custom_title)

        # Check title was set
        assert fig.axes[0].get_title() == custom_title
        plt.close(fig)

    def test_save_heatmap(self, plotter):
        """Test heatmap saving functionality."""
        # Create simple figure
        fig, ax = plt.subplots()
        ax.imshow([[1, 2], [3, 4]])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_heatmap.pdf")
            plotter.save_heatmap(fig, filepath)

            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

        plt.close(fig)

    def test_create_method_comparison_heatmap(self, plotter, sample_curves):
        """Test convenience method for creating complete heatmap."""
        fig = plotter.create_method_comparison_heatmap(
            sample_curves,
            tau_range=(0.5, 0.7),
            tau_step=0.1,
            title="Test Comparison"
        )

        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_title() == "Test Comparison"
        plt.close(fig)

    def test_create_method_comparison_heatmap_custom_params(self, plotter, sample_curves):
        """Test convenience method with custom parameters."""
        fig = plotter.create_method_comparison_heatmap(
            sample_curves,
            tau_range=(0.6, 0.6),  # Single tau value
            tau_step=0.1,
            baseline_method="Scratch_T2"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_add_cell_annotations(self, plotter):
        """Test cell annotation functionality."""
        fig, ax = plt.subplots()
        ad_matrix = np.array([[1.5, np.nan], [-0.8, 2.1]])
        methods = ['sgd', 'ewc_on']
        taus = np.array([0.5, 0.6])

        plotter._add_cell_annotations(ax, ad_matrix, methods, taus)

        # Check that text objects were added (excluding NaN cells)
        texts = [child for child in ax.get_children() if hasattr(child, 'get_text')]
        assert len(texts) >= 3  # Should have annotations for non-NaN cells

        plt.close(fig)

    def test_add_nan_hatching(self, plotter):
        """Test NaN cell hatching functionality."""
        fig, ax = plt.subplots()
        ad_matrix = np.array([[1.0, np.nan], [np.nan, 2.0]])

        plotter._add_nan_hatching(ax, ad_matrix)

        # Check that patches were added for NaN cells
        patches_added = [child for child in ax.get_children()
                        if hasattr(child, 'get_hatch') and child.get_hatch() is not None]
        assert len(patches_added) == 2  # Two NaN cells

        plt.close(fig)

    def test_add_nan_legend(self, plotter):
        """Test NaN legend functionality."""
        fig, ax = plt.subplots()

        result = TauSensitivityResult(
            methods=['sgd', 'ewc_on'],
            taus=np.array([0.5, 0.6]),
            ad_matrix=np.array([[1.0, np.nan], [np.nan, 2.0]]),
            n_censored={'sgd': 1, 'ewc_on': 1}
        )

        plotter._add_nan_legend(ax, result)

        # Check that text was added
        texts = [child for child in ax.get_children() if hasattr(child, 'get_text')]
        assert len(texts) >= 1

        # Check legend text content
        legend_text = texts[0].get_text()
        assert "No threshold crossing" in legend_text
        assert "2/4" in legend_text  # 2 censored out of 4 total

        plt.close(fig)

    @pytest.mark.integration
    def test_end_to_end_heatmap_generation(self, plotter, sample_curves):
        """Integration test for complete heatmap generation pipeline."""
        # Test the complete pipeline from curves to saved heatmap
        taus = [0.5, 0.6, 0.7]

        # Compute sensitivity
        result = plotter.compute_tau_sensitivity(sample_curves, taus)

        # Create heatmap
        fig = plotter.create_tau_sensitivity_heatmap(result)

        # Save heatmap
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "integration_test_heatmap.pdf")
            plotter.save_heatmap(fig, filepath)

            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 1000  # Should be substantial file

        plt.close(fig)

    def test_heatmap_visual_elements(self, plotter):
        """Test that heatmap contains expected visual elements."""
        # Create synthetic data with known properties
        methods = ['method1', 'method2']
        taus = np.array([0.5, 0.6, 0.7])
        ad_matrix = np.array([
            [1.0, 1.5, np.nan],
            [-0.5, 0.0, 2.0]
        ])

        result = TauSensitivityResult(
            methods=methods,
            taus=taus,
            ad_matrix=ad_matrix,
            n_censored={'method1': 1, 'method2': 0}
        )

        fig = plotter.create_tau_sensitivity_heatmap(result)
        ax = fig.axes[0]

        # Check axis labels
        assert ax.get_xlabel() == 'Threshold τ'
        assert ax.get_ylabel() == 'Method'

        # Check tick labels
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert '0.50' in x_labels
        assert '0.70' in x_labels

        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        assert 'method1' in y_labels
        assert 'method2' in y_labels

        plt.close(fig)
