"""
Plot Style Configuration for ERI Visualization System

This module provides comprehensive styling configuration for ERI visualizations,
including color palettes, fonts, layout settings, and figure dimensions.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


@dataclass
class PlotStyleConfig:
    """
    Configuration class for ERI visualization styling.

    Provides sensible defaults for colors, fonts, and layout while allowing
    customization for different methods and publication requirements.

    Attributes:
        figure_size: Figure dimensions in inches (width, height)
        dpi: Dots per inch for figure resolution
        color_palette: Method-specific color mapping
        font_sizes: Font size configuration for different elements
        confidence_alpha: Transparency for confidence interval bands
        line_width: Default line width for plots
        marker_size: Default marker size for scatter plots
        grid_alpha: Transparency for grid lines
        legend_frameon: Whether to show legend frame
        tight_layout: Whether to use tight layout
        save_format: Default save format for figures
        save_kwargs: Additional keyword arguments for saving figures
    """

    # Figure dimensions and resolution
    figure_size: Tuple[float, float] = (12, 10)
    dpi: int = 300

    # Color palette for different methods
    color_palette: Dict[str, str] = field(default_factory=lambda: {
        "Scratch_T2": "#333333",      # Dark gray for baseline
        "sgd": "#1f77b4",             # Blue for SGD
        "ewc_on": "#2ca02c",          # Green for EWC
        "derpp": "#d62728",           # Red for DER++
        "gmp": "#8c564b",             # Brown for GMP
        "Interleaved": "#9467bd",     # Purple for interleaved
        "er": "#ff7f0e",              # Orange for Experience Replay
        "lwf": "#e377c2",             # Pink for Learning without Forgetting
        "icarl": "#17becf",           # Cyan for iCaRL
        "gem": "#bcbd22",             # Olive for GEM
        "agem": "#7f7f7f",            # Gray for A-GEM
    })

    # Font size configuration
    font_sizes: Dict[str, int] = field(default_factory=lambda: {
        "title": 14,
        "axis_label": 12,
        "legend": 10,
        "annotation": 9,
        "tick_label": 10,
        "panel_label": 16,  # For panel A, B, C labels
    })

    # Visual styling parameters
    confidence_alpha: float = 0.15
    line_width: float = 2.0
    marker_size: float = 6.0
    grid_alpha: float = 0.3
    legend_frameon: bool = True
    tight_layout: bool = True

    # Save configuration
    save_format: str = "pdf"
    save_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "bbox_inches": "tight",
        "pad_inches": 0.1,
        "facecolor": "white",
        "edgecolor": "none",
    })

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.dpi <= 0:
            raise ValueError(f"DPI must be positive, got {self.dpi}")

        if len(self.figure_size) != 2 or any(s <= 0 for s in self.figure_size):
            raise ValueError(f"Figure size must be (width, height) with positive values, got {self.figure_size}")

        if not (0 <= self.confidence_alpha <= 1):
            raise ValueError(f"Confidence alpha must be between 0 and 1, got {self.confidence_alpha}")

        if self.line_width <= 0:
            raise ValueError(f"Line width must be positive, got {self.line_width}")

        if self.marker_size <= 0:
            raise ValueError(f"Marker size must be positive, got {self.marker_size}")

        if not (0 <= self.grid_alpha <= 1):
            raise ValueError(f"Grid alpha must be between 0 and 1, got {self.grid_alpha}")

    def get_method_color(self, method: str) -> str:
        """
        Get color for a specific method.

        Args:
            method: Method name

        Returns:
            Color string (hex format)

        Raises:
            KeyError: If method not found in color palette
        """
        if method not in self.color_palette:
            # Generate a default color for unknown methods
            import hashlib
            hash_obj = hashlib.md5(method.encode())
            hash_hex = hash_obj.hexdigest()
            # Use first 6 characters as hex color
            default_color = f"#{hash_hex[:6]}"
            return default_color

        return self.color_palette[method]

    def apply_style(self) -> None:
        """Apply the style configuration to matplotlib."""
        # Set figure defaults
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['figure.dpi'] = self.dpi

        # Set font sizes
        plt.rcParams['font.size'] = self.font_sizes.get('tick_label', 10)
        plt.rcParams['axes.titlesize'] = self.font_sizes.get('title', 14)
        plt.rcParams['axes.labelsize'] = self.font_sizes.get('axis_label', 12)
        plt.rcParams['legend.fontsize'] = self.font_sizes.get('legend', 10)

        # Set line and marker defaults
        plt.rcParams['lines.linewidth'] = self.line_width
        plt.rcParams['lines.markersize'] = self.marker_size

        # Set grid defaults
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = self.grid_alpha

        # Set legend defaults
        plt.rcParams['legend.frameon'] = self.legend_frameon

        # Set tight layout
        if self.tight_layout:
            plt.rcParams['figure.autolayout'] = True

    def create_custom_palette(self, methods: list[str]) -> Dict[str, str]:
        """
        Create a custom color palette for a specific set of methods.

        Args:
            methods: List of method names

        Returns:
            Dictionary mapping method names to colors
        """
        palette = {}
        for method in methods:
            palette[method] = self.get_method_color(method)
        return palette

    def update_color_palette(self, new_colors: Dict[str, str]) -> None:
        """
        Update the color palette with new method colors.

        Args:
            new_colors: Dictionary of method name to color mappings
        """
        self.color_palette.update(new_colors)

    def set_dpi(self, dpi: int) -> None:
        """
        Set the DPI for figure resolution.

        Args:
            dpi: Dots per inch

        Raises:
            ValueError: If DPI is not positive
        """
        if dpi <= 0:
            raise ValueError(f"DPI must be positive, got {dpi}")
        self.dpi = dpi

    def set_figure_size(self, width: float, height: float) -> None:
        """
        Set the figure dimensions.

        Args:
            width: Figure width in inches
            height: Figure height in inches

        Raises:
            ValueError: If dimensions are not positive
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Figure dimensions must be positive, got ({width}, {height})")
        self.figure_size = (width, height)

    def get_diverging_colormap(self, center_value: float = 0.0) -> str:
        """
        Get a diverging colormap name suitable for heatmaps.

        Args:
            center_value: Value to center the colormap on

        Returns:
            Colormap name
        """
        return "RdBu_r"  # Red-Blue diverging, reversed (red for positive, blue for negative)

    def save_figure(self, fig: plt.Figure, filepath: str, **kwargs) -> None:
        """
        Save figure with configured settings.

        Args:
            fig: Matplotlib figure to save
            filepath: Output file path
            **kwargs: Additional keyword arguments (override defaults)
        """
        save_kwargs = self.save_kwargs.copy()
        save_kwargs.update(kwargs)

        # Set DPI if not already specified in kwargs
        if 'dpi' not in save_kwargs:
            save_kwargs['dpi'] = self.dpi

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filepath, **save_kwargs)

    @classmethod
    def create_publication_style(cls) -> "PlotStyleConfig":
        """
        Create a style configuration optimized for publication.

        Returns:
            PlotStyleConfig instance with publication-ready settings
        """
        return cls(
            figure_size=(10, 8),
            dpi=300,
            font_sizes={
                "title": 12,
                "axis_label": 11,
                "legend": 9,
                "annotation": 8,
                "tick_label": 9,
                "panel_label": 14,
            },
            line_width=1.5,
            marker_size=4.0,
            confidence_alpha=0.2,
            save_kwargs={
                "bbox_inches": "tight",
                "pad_inches": 0.05,
                "facecolor": "white",
                "edgecolor": "none",
            }
        )

    @classmethod
    def create_presentation_style(cls) -> "PlotStyleConfig":
        """
        Create a style configuration optimized for presentations.

        Returns:
            PlotStyleConfig instance with presentation-ready settings
        """
        return cls(
            figure_size=(14, 10),
            dpi=150,
            font_sizes={
                "title": 18,
                "axis_label": 16,
                "legend": 14,
                "annotation": 12,
                "tick_label": 14,
                "panel_label": 20,
            },
            line_width=3.0,
            marker_size=8.0,
            confidence_alpha=0.25,
        )


# Default style instance
DEFAULT_STYLE = PlotStyleConfig()
