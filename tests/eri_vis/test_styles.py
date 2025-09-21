"""
Tests for ERI Visualization System Plot Style Configuration

This module tests the PlotStyleConfig class and its functionality,
including default settings, customization, and validation.
"""

import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from eri_vis.styles import PlotStyleConfig, DEFAULT_STYLE


class TestPlotStyleConfig:
    """Test cases for PlotStyleConfig class."""

    def test_default_initialization(self):
        """Test that PlotStyleConfig initializes with sensible defaults."""
        config = PlotStyleConfig()

        # Test figure dimensions
        assert config.figure_size == (12, 10)
        assert config.dpi == 300

        # Test color palette contains expected methods
        expected_methods = ["Scratch_T2", "sgd", "ewc_on", "derpp", "gpm", "Interleaved"]
        for method in expected_methods:
            assert method in config.color_palette
            assert config.color_palette[method].startswith("#")

        # Test font sizes
        assert config.font_sizes["title"] == 14
        assert config.font_sizes["axis_label"] == 12
        assert config.font_sizes["legend"] == 10

        # Test visual parameters
        assert 0 <= config.confidence_alpha <= 1
        assert config.line_width > 0
        assert config.marker_size > 0
        assert 0 <= config.grid_alpha <= 1

    def test_custom_initialization(self):
        """Test PlotStyleConfig with custom parameters."""
        custom_colors = {"method1": "#ff0000", "method2": "#00ff00"}
        custom_fonts = {"title": 16, "axis_label": 14}

        config = PlotStyleConfig(
            figure_size=(8, 6),
            dpi=150,
            color_palette=custom_colors,
            font_sizes=custom_fonts,
            confidence_alpha=0.3
        )

        assert config.figure_size == (8, 6)
        assert config.dpi == 150
        assert config.color_palette == custom_colors
        assert config.font_sizes == custom_fonts
        assert config.confidence_alpha == 0.3

    def test_validation_errors(self):
        """Test that invalid configurations raise appropriate errors."""
        # Test invalid DPI
        with pytest.raises(ValueError, match="DPI must be positive"):
            PlotStyleConfig(dpi=0)

        with pytest.raises(ValueError, match="DPI must be positive"):
            PlotStyleConfig(dpi=-100)

        # Test invalid figure size
        with pytest.raises(ValueError, match="Figure size must be"):
            PlotStyleConfig(figure_size=(0, 5))

        with pytest.raises(ValueError, match="Figure size must be"):
            PlotStyleConfig(figure_size=(5, -1))

        with pytest.raises(ValueError, match="Figure size must be"):
            PlotStyleConfig(figure_size=(5,))  # Wrong length

        # Test invalid confidence alpha
        with pytest.raises(ValueError, match="Confidence alpha must be between 0 and 1"):
            PlotStyleConfig(confidence_alpha=-0.1)

        with pytest.raises(ValueError, match="Confidence alpha must be between 0 and 1"):
            PlotStyleConfig(confidence_alpha=1.5)

        # Test invalid line width
        with pytest.raises(ValueError, match="Line width must be positive"):
            PlotStyleConfig(line_width=0)

        # Test invalid marker size
        with pytest.raises(ValueError, match="Marker size must be positive"):
            PlotStyleConfig(marker_size=-1)

        # Test invalid grid alpha
        with pytest.raises(ValueError, match="Grid alpha must be between 0 and 1"):
            PlotStyleConfig(grid_alpha=2.0)

    def test_get_method_color(self):
        """Test method color retrieval."""
        config = PlotStyleConfig()

        # Test existing method
        color = config.get_method_color("sgd")
        assert color == "#1f77b4"

        # Test unknown method (should generate default color)
        unknown_color = config.get_method_color("unknown_method")
        assert unknown_color.startswith("#")
        assert len(unknown_color) == 7  # #RRGGBB format

        # Test that same unknown method returns same color
        unknown_color2 = config.get_method_color("unknown_method")
        assert unknown_color == unknown_color2

    def test_apply_style(self):
        """Test that apply_style correctly sets matplotlib parameters."""
        config = PlotStyleConfig(
            figure_size=(10, 8),
            dpi=200,
            font_sizes={"title": 16, "axis_label": 14, "legend": 12, "tick_label": 11},
            line_width=2.5,
            marker_size=7.0,
            grid_alpha=0.4
        )

        # Store original values
        original_figsize = plt.rcParams['figure.figsize']
        original_dpi = plt.rcParams['figure.dpi']

        try:
            config.apply_style()

            # Test that parameters were set correctly
            assert list(plt.rcParams['figure.figsize']) == [10, 8]
            assert plt.rcParams['figure.dpi'] == 200
            assert plt.rcParams['axes.titlesize'] == 16
            assert plt.rcParams['axes.labelsize'] == 14
            assert plt.rcParams['legend.fontsize'] == 12
            assert plt.rcParams['font.size'] == 11
            assert plt.rcParams['lines.linewidth'] == 2.5
            assert plt.rcParams['lines.markersize'] == 7.0
            assert plt.rcParams['grid.alpha'] == 0.4
            assert plt.rcParams['axes.grid'] is True

        finally:
            # Restore original values
            plt.rcParams['figure.figsize'] = original_figsize
            plt.rcParams['figure.dpi'] = original_dpi

    def test_create_custom_palette(self):
        """Test custom palette creation for specific methods."""
        config = PlotStyleConfig()
        methods = ["sgd", "ewc_on", "unknown_method"]

        palette = config.create_custom_palette(methods)

        assert len(palette) == 3
        assert palette["sgd"] == config.color_palette["sgd"]
        assert palette["ewc_on"] == config.color_palette["ewc_on"]
        assert "unknown_method" in palette
        assert palette["unknown_method"].startswith("#")

    def test_update_color_palette(self):
        """Test updating the color palette."""
        config = PlotStyleConfig()
        original_sgd_color = config.color_palette["sgd"]

        new_colors = {"sgd": "#ff0000", "new_method": "#00ff00"}
        config.update_color_palette(new_colors)

        assert config.color_palette["sgd"] == "#ff0000"
        assert config.color_palette["new_method"] == "#00ff00"
        # Other colors should remain unchanged
        assert config.color_palette["ewc_on"] == "#2ca02c"

    def test_set_dpi(self):
        """Test DPI setter with validation."""
        config = PlotStyleConfig()

        config.set_dpi(150)
        assert config.dpi == 150

        with pytest.raises(ValueError, match="DPI must be positive"):
            config.set_dpi(0)

        with pytest.raises(ValueError, match="DPI must be positive"):
            config.set_dpi(-50)

    def test_set_figure_size(self):
        """Test figure size setter with validation."""
        config = PlotStyleConfig()

        config.set_figure_size(8, 6)
        assert config.figure_size == (8, 6)

        with pytest.raises(ValueError, match="Figure dimensions must be positive"):
            config.set_figure_size(0, 6)

        with pytest.raises(ValueError, match="Figure dimensions must be positive"):
            config.set_figure_size(8, -1)

    def test_get_diverging_colormap(self):
        """Test diverging colormap selection."""
        config = PlotStyleConfig()

        colormap = config.get_diverging_colormap()
        assert colormap == "RdBu_r"

        # Test with different center value (should still return same colormap)
        colormap_custom = config.get_diverging_colormap(center_value=0.5)
        assert colormap_custom == "RdBu_r"

    def test_save_figure(self):
        """Test figure saving functionality."""
        config = PlotStyleConfig()

        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_figure.pdf")

            config.save_figure(fig, filepath)

            # Check that file was created
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

        plt.close(fig)

    def test_save_figure_with_custom_kwargs(self):
        """Test figure saving with custom keyword arguments."""
        config = PlotStyleConfig()

        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_figure.png")

            config.save_figure(fig, filepath, format="png", facecolor="lightgray")

            # Check that file was created
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

        plt.close(fig)

    def test_create_publication_style(self):
        """Test publication style factory method."""
        pub_style = PlotStyleConfig.create_publication_style()

        assert pub_style.figure_size == (10, 8)
        assert pub_style.dpi == 300
        assert pub_style.font_sizes["title"] == 12
        assert pub_style.font_sizes["axis_label"] == 11
        assert pub_style.line_width == 1.5
        assert pub_style.marker_size == 4.0
        assert pub_style.confidence_alpha == 0.2

    def test_create_presentation_style(self):
        """Test presentation style factory method."""
        pres_style = PlotStyleConfig.create_presentation_style()

        assert pres_style.figure_size == (14, 10)
        assert pres_style.dpi == 150
        assert pres_style.font_sizes["title"] == 18
        assert pres_style.font_sizes["axis_label"] == 16
        assert pres_style.line_width == 3.0
        assert pres_style.marker_size == 8.0
        assert pres_style.confidence_alpha == 0.25

    def test_default_style_instance(self):
        """Test that DEFAULT_STYLE is properly initialized."""
        assert isinstance(DEFAULT_STYLE, PlotStyleConfig)
        assert DEFAULT_STYLE.figure_size == (12, 10)
        assert DEFAULT_STYLE.dpi == 300


class TestStyleIntegration:
    """Integration tests for style configuration."""

    def test_style_application_affects_plots(self):
        """Test that applying style actually affects matplotlib plots."""
        config = PlotStyleConfig(
            figure_size=(6, 4),
            dpi=100,
            line_width=3.0
        )

        # Store original values
        original_figsize = plt.rcParams['figure.figsize']
        original_dpi = plt.rcParams['figure.dpi']
        original_linewidth = plt.rcParams['lines.linewidth']

        try:
            config.apply_style()

            # Create a figure and check that it uses the new settings
            fig, ax = plt.subplots()

            # Check figure properties
            assert fig.get_figwidth() == 6
            assert fig.get_figheight() == 4
            # Note: figure.dpi might be different from rcParams due to backend behavior
            # Just check that it's been set to a reasonable value
            assert fig.dpi >= 100

            # Plot a line and check its properties
            line, = ax.plot([1, 2, 3], [1, 4, 2])
            assert line.get_linewidth() == 3.0

            plt.close(fig)

        finally:
            # Restore original values
            plt.rcParams['figure.figsize'] = original_figsize
            plt.rcParams['figure.dpi'] = original_dpi
            plt.rcParams['lines.linewidth'] = original_linewidth

    def test_color_palette_consistency(self):
        """Test that color palette provides consistent colors."""
        config = PlotStyleConfig()

        # Test that colors are consistent across calls
        color1 = config.get_method_color("sgd")
        color2 = config.get_method_color("sgd")
        assert color1 == color2

        # Test that different methods get different colors
        sgd_color = config.get_method_color("sgd")
        ewc_color = config.get_method_color("ewc_on")
        assert sgd_color != ewc_color

    def test_custom_palette_application(self):
        """Test that custom color palettes work correctly."""
        custom_colors = {
            "method_a": "#ff0000",
            "method_b": "#00ff00",
            "method_c": "#0000ff"
        }

        config = PlotStyleConfig(color_palette=custom_colors)

        assert config.get_method_color("method_a") == "#ff0000"
        assert config.get_method_color("method_b") == "#00ff00"
        assert config.get_method_color("method_c") == "#0000ff"

        # Unknown method should still get a generated color
        unknown_color = config.get_method_color("unknown")
        assert unknown_color.startswith("#")
        assert len(unknown_color) == 7


if __name__ == "__main__":
    pytest.main([__file__])
