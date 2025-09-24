#!/usr/bin/env python3
"""
ERI Visualization CLI Tool

Command-line interface for generating ERI (Einstellung Rigidity Index) visualizations
from CMammoth experiment results.

This tool provides a comprehensive interface for:
- Loading and validating CSV data
- Generating dynamics plots (3-panel figures)
- Creating robustness heatmaps
- Batch processing multiple datasets
- Configurable visualization parameters

Usage Examples:
    # Single CSV file
    python tools/plot_eri.py --csv logs/eri_metrics.csv --outdir results/

    # Multiple CSV files with glob pattern
    python tools/plot_eri.py --csv "logs/run_*.csv" --outdir results/

    # Custom parameters
    python tools/plot_eri.py --csv data.csv --outdir figs/ --tau 0.65 --smooth 5

    # Robustness analysis with tau grid
    python tools/plot_eri.py --csv data.csv --outdir figs/ --tau-grid 0.5 0.55 0.6 0.65 0.7 0.75 0.8

    # Method filtering
    python tools/plot_eri.py --csv data.csv --outdir figs/ --methods sgd ewc_on derpp
"""

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eri_vis.data_loader import ERIDataLoader, ERIDataValidationError
from eri_vis.dataset import ERITimelineDataset
from eri_vis.processing import ERITimelineProcessor
from eri_vis.plot_dynamics import ERIDynamicsPlotter
from eri_vis.plot_heatmap import ERIHeatmapPlotter
from eri_vis.styles import PlotStyleConfig


class ERICLIError(Exception):
    """Exception raised for CLI-specific errors."""
    pass


class ERICLI:
    """
    Main CLI class for ERI visualization tool.

    Handles argument parsing, data loading, processing, and visualization
    generation with comprehensive error handling and progress logging.
    """

    def __init__(self):
        """Initialize CLI with logging setup."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_loader = ERIDataLoader()
        self.processor = None  # Will be initialized with user parameters
        self.dynamics_plotter = None
        self.heatmap_plotter = None

    def setup_logging(self, level: int = logging.INFO) -> None:
        """
        Set up logging configuration.

        Args:
            level: Logging level (default INFO)
        """
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Reduce matplotlib logging noise
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create and configure argument parser.

        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            description='Generate ERI (Einstellung Rigidity Index) visualizations',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --csv data.csv --outdir results/
  %(prog)s --csv "logs/run_*.csv" --outdir results/ --methods sgd ewc_on
  %(prog)s --csv data.csv --outdir figs/ --tau 0.65 --smooth 5
  %(prog)s --csv data.csv --outdir figs/ --tau-grid 0.5 0.6 0.7 0.8

For more information, see docs/README_eri_vis.md
            """
        )

        # Input/Output arguments
        parser.add_argument(
            '--csv',
            type=str,
            required=True,
            help='CSV file path or glob pattern (e.g., "logs/*.csv")'
        )

        parser.add_argument(
            '--outdir',
            type=str,
            required=True,
            help='Output directory for generated figures'
        )

        # Method filtering
        parser.add_argument(
            '--methods',
            nargs='*',
            help='Methods to include in visualization (default: all methods in data)'
        )

        # Processing parameters
        parser.add_argument(
            '--tau',
            type=float,
            default=0.6,
            help='Threshold value for adaptation delay computation (default: 0.6)'
        )

        parser.add_argument(
            '--smooth',
            type=int,
            default=3,
            help='Smoothing window size (default: 3)'
        )

        # Robustness analysis
        parser.add_argument(
            '--tau-grid',
            nargs='*',
            type=float,
            help='Tau values for robustness heatmap (e.g., 0.5 0.55 0.6 0.65 0.7 0.75 0.8)'
        )

        # Visualization options
        parser.add_argument(
            '--style',
            choices=['default', 'publication', 'presentation', 'custom'],
            default='default',
            help='Plot style configuration (default: default)'
        )

        parser.add_argument(
            '--dpi',
            type=int,
            default=300,
            help='Figure DPI for output (default: 300)'
        )

        parser.add_argument(
            '--format',
            choices=['pdf', 'png', 'svg'],
            default='pdf',
            help='Output format (default: pdf)'
        )

        # Batch processing options
        parser.add_argument(
            '--batch-summary',
            action='store_true',
            help='Generate summary plots when processing multiple CSV files'
        )

        # Logging and debugging
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )

        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress progress messages (errors only)'
        )

        return parser

    def validate_args(self, args: argparse.Namespace) -> None:
        """
        Validate command line arguments.

        Args:
            args: Parsed arguments

        Raises:
            ERICLIError: If validation fails
        """
        # Validate tau value
        if not 0.0 <= args.tau <= 1.0:
            raise ERICLIError(f"Tau value must be between 0.0 and 1.0, got {args.tau}")

        # Validate smoothing window
        if args.smooth < 1:
            raise ERICLIError(f"Smoothing window must be >= 1, got {args.smooth}")

        # Validate tau grid if provided
        if args.tau_grid:
            for tau in args.tau_grid:
                if not 0.0 <= tau <= 1.0:
                    raise ERICLIError(f"All tau grid values must be between 0.0 and 1.0, got {tau}")

            if len(set(args.tau_grid)) != len(args.tau_grid):
                raise ERICLIError("Tau grid values must be unique")

        # Validate DPI
        if args.dpi < 50 or args.dpi > 1200:
            raise ERICLIError(f"DPI must be between 50 and 1200, got {args.dpi}")

        # Check for conflicting quiet/verbose flags
        if args.quiet and args.verbose:
            raise ERICLIError("Cannot specify both --quiet and --verbose")

    def setup_components(self, args: argparse.Namespace) -> None:
        """
        Initialize processing and plotting components with user parameters.

        Args:
            args: Parsed command line arguments
        """
        # Initialize processor with user parameters
        self.processor = ERITimelineProcessor(
            smoothing_window=args.smooth,
            tau=args.tau
        )

        # Create style configuration
        style_config = PlotStyleConfig()
        style_config.dpi = args.dpi

        # Apply style modifications based on user choice
        if args.style == 'publication':
            style_config.figure_size = (12, 10)
            style_config.font_sizes['title'] = 16
            style_config.font_sizes['axis_label'] = 14
        elif args.style == 'presentation':
            style_config.figure_size = (14, 12)
            style_config.font_sizes['title'] = 18
            style_config.font_sizes['axis_label'] = 16
            style_config.line_width = 3

        # Initialize plotters
        self.dynamics_plotter = ERIDynamicsPlotter(style_config)
        self.heatmap_plotter = ERIHeatmapPlotter(style_config)

    def find_csv_files(self, csv_pattern: str) -> List[Path]:
        """
        Find CSV files matching the given pattern.

        Args:
            csv_pattern: File path or glob pattern

        Returns:
            List of Path objects for found CSV files

        Raises:
            ERICLIError: If no files found or pattern invalid
        """
        try:
            # Handle glob patterns
            if '*' in csv_pattern or '?' in csv_pattern:
                files = glob.glob(csv_pattern)
                if not files:
                    raise ERICLIError(f"No CSV files found matching pattern: {csv_pattern}")
                csv_files = [Path(f) for f in files if f.endswith('.csv')]
            else:
                # Single file
                csv_file = Path(csv_pattern)
                if not csv_file.exists():
                    raise ERICLIError(f"CSV file not found: {csv_pattern}")
                if not csv_file.suffix.lower() == '.csv':
                    raise ERICLIError(f"File must have .csv extension: {csv_pattern}")
                csv_files = [csv_file]

            if not csv_files:
                raise ERICLIError(f"No valid CSV files found in: {csv_pattern}")

            return sorted(csv_files)

        except Exception as e:
            if isinstance(e, ERICLIError):
                raise
            raise ERICLIError(f"Error finding CSV files: {e}")

    def load_datasets(self, csv_files: List[Path]) -> List[Tuple[Path, ERITimelineDataset]]:
        """
        Load datasets from CSV files with error handling.

        Args:
            csv_files: List of CSV file paths

        Returns:
            List of (file_path, dataset) tuples for successfully loaded files
        """
        datasets = []

        for csv_file in csv_files:
            try:
                self.logger.info(f"Loading CSV file: {csv_file}")
                dataset = self.data_loader.load_csv(csv_file)
                datasets.append((csv_file, dataset))

                # Log dataset info
                self.logger.info(
                    f"Loaded {len(dataset.data)} rows, "
                    f"{len(dataset.methods)} methods, "
                    f"{len(dataset.seeds)} seeds, "
                    f"epochs {dataset.epoch_range[0]:.1f}-{dataset.epoch_range[1]:.1f}"
                )

            except ERIDataValidationError as e:
                self.logger.error(f"Data validation failed for {csv_file}: {e}")
                continue
            except Exception as e:
                self.logger.error(f"Failed to load {csv_file}: {e}")
                continue

        if not datasets:
            raise ERICLIError("No valid datasets could be loaded")

        return datasets

    def filter_methods(self, dataset: ERITimelineDataset, methods: Optional[List[str]]) -> ERITimelineDataset:
        """
        Filter dataset to include only specified methods.

        Args:
            dataset: Input dataset
            methods: List of method names to include (None = all methods)

        Returns:
            Filtered dataset

        Raises:
            ERICLIError: If no valid methods found
        """
        if methods is None:
            return dataset

        # Check which methods are available
        available_methods = set(dataset.methods)
        requested_methods = set(methods)

        missing_methods = requested_methods - available_methods
        if missing_methods:
            self.logger.warning(f"Requested methods not found in data: {missing_methods}")

        valid_methods = requested_methods & available_methods
        if not valid_methods:
            raise ERICLIError(
                f"None of the requested methods found in data. "
                f"Available: {sorted(available_methods)}, "
                f"Requested: {sorted(requested_methods)}"
            )

        # Filter data
        filtered_data = dataset.data[dataset.data['method'].isin(valid_methods)].copy()

        if len(filtered_data) == 0:
            raise ERICLIError("No data remaining after method filtering")

        # Create new dataset with filtered data
        filtered_methods = sorted(list(valid_methods))

        return ERITimelineDataset(
            data=filtered_data,
            metadata={**dataset.metadata, 'filtered_methods': methods},
            methods=filtered_methods,
            splits=dataset.splits,
            seeds=sorted(filtered_data['seed'].unique()),
            epoch_range=(filtered_data['epoch_eff'].min(), filtered_data['epoch_eff'].max())
        )

    def generate_dynamics_plot(
        self,
        dataset: ERITimelineDataset,
        output_dir: Path,
        file_prefix: str = "fig_eri_dynamics",
        format: str = "pdf"
    ) -> Optional[Path]:
        """
        Generate dynamics plot for a dataset.

        Args:
            dataset: Input dataset
            output_dir: Output directory
            file_prefix: File name prefix
            format: Output format

        Returns:
            Path to generated file, or None if generation failed
        """
        try:
            self.logger.info("Computing accuracy curves...")
            curves = self.processor.compute_accuracy_curves(dataset)

            # Separate curves by split
            patched_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}
            masked_curves = {k: v for k, v in curves.items() if 'shortcut_masked' in k}

            if not patched_curves:
                self.logger.warning("No shortcut_normal curves found - skipping dynamics plot")
                return None

            self.logger.info("Computing adaptation delays...")
            ad_values = self.processor.compute_adaptation_delays(curves)

            self.logger.info("Computing performance deficits...")
            pd_series = self.processor.compute_performance_deficits(curves)

            self.logger.info("Computing shortcut forgetting rates...")
            sfr_series = self.processor.compute_sfr_relative(curves)

            # Generate plot
            self.logger.info("Generating dynamics figure...")
            fig = self.dynamics_plotter.create_dynamics_figure(
                patched_curves=patched_curves,
                masked_curves=masked_curves,
                pd_series=pd_series,
                sfr_series=sfr_series,
                ad_values=ad_values,
                tau=self.processor.tau
            )

            # Save figure
            output_file = output_dir / f"{file_prefix}.{format}"
            self.dynamics_plotter.save_figure(fig, str(output_file))

            self.logger.info(f"Dynamics plot saved: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Failed to generate dynamics plot: {e}")
            return None

    def generate_accuracy_plot(
        self,
        dataset: ERITimelineDataset,
        output_dir: Path,
        file_prefix: str = "fig_eri_accuracy",
        format: str = "pdf"
    ) -> Optional[Path]:
        """Generate single-panel accuracy trajectories plot."""
        try:
            curves = self.processor.compute_accuracy_curves(dataset)
            patched_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}
            masked_curves = {k: v for k, v in curves.items() if 'shortcut_masked' in k}

            if not patched_curves:
                self.logger.warning("No shortcut_normal curves found - skipping accuracy plot")
                return None

            ad_values = self.processor.compute_adaptation_delays(curves)

            fig = self.dynamics_plotter.create_accuracy_only_figure(
                patched_curves=patched_curves,
                masked_curves=masked_curves,
                ad_values=ad_values,
                tau=self.processor.tau,
                title="Accuracy Trajectories on Shortcut Task"
            )

            output_file = output_dir / f"{file_prefix}.{format}"
            self.dynamics_plotter.save_figure(fig, str(output_file))
            self.logger.info(f"Accuracy plot saved: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Failed to generate accuracy plot: {e}")
            return None

    def generate_heatmap(
        self,
        dataset: ERITimelineDataset,
        output_dir: Path,
        tau_grid: List[float],
        file_prefix: str = "fig_ad_tau_heatmap",
        format: str = "pdf"
    ) -> Optional[Path]:
        """
        Generate robustness heatmap for a dataset.

        Args:
            dataset: Input dataset
            output_dir: Output directory
            tau_grid: List of tau values for sensitivity analysis
            file_prefix: File name prefix
            format: Output format

        Returns:
            Path to generated file, or None if generation failed
        """
        try:
            self.logger.info("Computing accuracy curves for heatmap...")
            curves = self.processor.compute_accuracy_curves(dataset)

            # Check if we have enough data for heatmap
            shortcut_curves = {k: v for k, v in curves.items() if 'shortcut_normal' in k}
            if len(shortcut_curves) < 2:  # Need at least baseline + 1 CL method
                self.logger.warning("Insufficient methods for heatmap - need at least 2 methods")
                return None

            self.logger.info(f"Generating robustness heatmap with tau values: {tau_grid}")
            fig = self.heatmap_plotter.create_method_comparison_heatmap(
                curves=curves,
                tau_range=(min(tau_grid), max(tau_grid)),
                tau_step=tau_grid[1] - tau_grid[0] if len(tau_grid) > 1 else 0.05,
                title="Adaptation Delay Sensitivity Analysis"
            )

            # Save figure
            output_file = output_dir / f"{file_prefix}.{format}"
            self.heatmap_plotter.save_heatmap(fig, str(output_file))

            self.logger.info(f"Heatmap saved: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Failed to generate heatmap: {e}")
            return None

    def process_single_dataset(
        self,
        csv_file: Path,
        dataset: ERITimelineDataset,
        args: argparse.Namespace,
        output_dir: Path
    ) -> Dict[str, Optional[Path]]:
        """
        Process a single dataset and generate visualizations.

        Args:
            csv_file: Source CSV file path
            dataset: Loaded dataset
            args: Command line arguments
            output_dir: Output directory

        Returns:
            Dictionary mapping plot type to generated file path
        """
        results = {}

        # Filter methods if specified
        if args.methods:
            try:
                dataset = self.filter_methods(dataset, args.methods)
            except ERICLIError as e:
                self.logger.error(f"Method filtering failed for {csv_file}: {e}")
                return results

        # Create file prefix based on CSV file name
        file_prefix = csv_file.stem

        # Generate dynamics plot
        dynamics_file = self.generate_dynamics_plot(
            dataset, output_dir, f"{file_prefix}_dynamics", args.format
        )
        results['dynamics'] = dynamics_file

        accuracy_file = self.generate_accuracy_plot(
            dataset, output_dir, f"{file_prefix}_accuracy", args.format
        )
        results['accuracy'] = accuracy_file

        # Generate heatmap if tau grid specified
        if args.tau_grid:
            heatmap_file = self.generate_heatmap(
                dataset, output_dir, args.tau_grid, f"{file_prefix}_heatmap", args.format
            )
            results['heatmap'] = heatmap_file

        return results

    def generate_batch_summary(
        self,
        datasets: List[Tuple[Path, ERITimelineDataset]],
        args: argparse.Namespace,
        output_dir: Path
    ) -> None:
        """
        Generate summary visualizations for batch processing.

        Args:
            datasets: List of (file_path, dataset) tuples
            args: Command line arguments
            output_dir: Output directory
        """
        try:
            self.logger.info("Generating batch summary...")

            # Combine all datasets
            all_data = []
            for csv_file, dataset in datasets:
                # Add source file info to metadata
                data_copy = dataset.data.copy()
                data_copy['source_file'] = csv_file.stem
                all_data.append(data_copy)

            if not all_data:
                self.logger.warning("No data available for batch summary")
                return

            # Create combined dataset
            import pandas as pd
            combined_data = pd.concat(all_data, ignore_index=True)

            # Get combined metadata
            all_methods = sorted(combined_data['method'].unique())
            all_splits = sorted(combined_data['split'].unique())
            all_seeds = sorted(combined_data['seed'].unique())
            epoch_range = (combined_data['epoch_eff'].min(), combined_data['epoch_eff'].max())

            combined_dataset = ERITimelineDataset(
                data=combined_data,
                metadata={'source': 'batch_summary', 'n_files': len(datasets)},
                methods=all_methods,
                splits=all_splits,
                seeds=all_seeds,
                epoch_range=epoch_range
            )

            # Generate summary plots
            self.process_single_dataset(
                Path("batch_summary"),
                combined_dataset,
                args,
                output_dir
            )

        except Exception as e:
            self.logger.error(f"Failed to generate batch summary: {e}")

    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Main entry point for CLI execution.

        Args:
            args: Command line arguments (uses sys.argv if None)

        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            # Parse arguments
            parser = self.create_parser()
            parsed_args = parser.parse_args(args)

            # Set up logging level
            if parsed_args.quiet:
                self.setup_logging(logging.ERROR)
            elif parsed_args.verbose:
                self.setup_logging(logging.DEBUG)

            # Validate arguments
            self.validate_args(parsed_args)

            # Set up components
            self.setup_components(parsed_args)

            # Create output directory
            output_dir = Path(parsed_args.outdir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {output_dir}")

            # Find CSV files
            csv_files = self.find_csv_files(parsed_args.csv)
            self.logger.info(f"Found {len(csv_files)} CSV file(s)")

            # Load datasets
            datasets = self.load_datasets(csv_files)
            self.logger.info(f"Successfully loaded {len(datasets)} dataset(s)")

            # Process each dataset
            all_results = {}
            for csv_file, dataset in datasets:
                self.logger.info(f"Processing dataset: {csv_file}")
                results = self.process_single_dataset(csv_file, dataset, parsed_args, output_dir)
                all_results[csv_file] = results

            # Generate batch summary if requested and multiple files
            if parsed_args.batch_summary and len(datasets) > 1:
                self.generate_batch_summary(datasets, parsed_args, output_dir)

            # Report results
            self.report_results(all_results)

            self.logger.info("ERI visualization completed successfully")
            return 0

        except ERICLIError as e:
            self.logger.error(f"CLI Error: {e}")
            return 1
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 1
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            if parsed_args.verbose if 'parsed_args' in locals() else False:
                import traceback
                traceback.print_exc()
            return 1

    def report_results(self, results: Dict[Path, Dict[str, Optional[Path]]]) -> None:
        """
        Report generation results to user.

        Args:
            results: Dictionary mapping CSV files to their generated outputs
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("GENERATION SUMMARY")
        self.logger.info("="*60)

        total_files = 0
        successful_files = 0

        for csv_file, file_results in results.items():
            self.logger.info(f"\nDataset: {csv_file}")

            for plot_type, output_file in file_results.items():
                total_files += 1
                if output_file:
                    successful_files += 1
                    self.logger.info(f"  ✓ {plot_type}: {output_file}")
                else:
                    self.logger.info(f"  ✗ {plot_type}: Failed")

        self.logger.info(f"\nTotal: {successful_files}/{total_files} files generated successfully")

        if successful_files < total_files:
            self.logger.warning("Some visualizations failed - check error messages above")


def main():
    """Main entry point for command line usage."""
    cli = ERICLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
