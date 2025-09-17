"""
MammothERIIntegration - Bridge between Mammoth framework and ERI visualization system.

This module provides seamless integration with the existing Mammoth Einstellung
experiment infrastructure, specifically:
- utils/einstellung_evaluator.py: Plugin-based evaluation system
- datasets/seq_cifar100_einstellung_224.py: ViT-compatible dataset
- Mammoth training pipeline: Hooks, logging, checkpoint management

CRITICAL: This module extends existing infrastructure rather than recreating it.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ..data_loader import ERIDataLoader
from ..dataset import ERITimelineDataset
from ..processing import ERITimelineProcessor
from ..plot_dynamics import ERIDynamicsPlotter
from ..plot_heatmap import ERIHeatmapPlotter
from ..styles import PlotStyleConfig


class MammothERIIntegration:
    """
    Bridge between Mammoth framework and ERI visualization system.

    Integrates with existing EinstellungEvaluator to provide automatic
    visualization generation during and after training.
    """

    def __init__(self,
                 evaluator=None,
                 output_dir: Optional[str] = None,
                 style_config: Optional[PlotStyleConfig] = None):
        """
        Initialize the Mammoth integration.

        Args:
            evaluator: EinstellungEvaluator instance (from utils/einstellung_evaluator.py)
            output_dir: Base directory for outputs
            style_config: Plot styling configuration
        """
        self.evaluator = evaluator
        self.output_dir = Path(output_dir) if output_dir else Path("logs")
        self.style_config = style_config or PlotStyleConfig()

        # Initialize components
        self.data_loader = ERIDataLoader()
        self.processor = ERITimelineProcessor()
        self.dynamics_plotter = ERIDynamicsPlotter(self.style_config)
        self.heatmap_plotter = ERIHeatmapPlotter(self.style_config)

        # State tracking
        self.auto_export_enabled = False
        self.export_frequency = 1
        self.last_export_epoch = -1

        # Logging
        self.logger = logging.getLogger(__name__)

    def setup_auto_export(self,
                         output_dir: str,
                         export_frequency: int = 1) -> None:
        """
        Setup automatic export of visualization data during training.

        Hooks into existing EinstellungEvaluator to automatically export
        CSV data and generate visualizations at specified intervals.

        Args:
            output_dir: Directory for outputs
            export_frequency: Export every N epochs (1 = every epoch)
        """
        self.output_dir = Path(output_dir)
        self.export_frequency = export_frequency
        self.auto_export_enabled = True

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figs").mkdir(exist_ok=True)

        self.logger.info(f"Auto-export enabled: {output_dir}, frequency={export_frequency}")

        if self.evaluator is None:
            self.logger.warning("No evaluator provided - auto-export will not function")

    def export_timeline_for_visualization(self, filepath: str) -> None:
        """
        Export timeline data from EinstellungEvaluator in CSV format for visualization.

        Converts the existing evaluator's timeline data to the CSV schema
        required by the visualization system.

        Args:
            filepath: Path to save CSV file
        """
        if self.evaluator is None:
            raise ValueError("No EinstellungEvaluator provided")

        if not hasattr(self.evaluator, 'timeline_data') or not self.evaluator.timeline_data:
            self.logger.warning("No timeline data available in evaluator")
            return

        # Convert timeline data to CSV format
        csv_rows = []

        for entry in self.evaluator.timeline_data:
            epoch = entry.get('epoch', 0)
            task_id = entry.get('task_id', 0)
            subset_accuracies = entry.get('subset_accuracies', {})

            # Convert effective epoch (Phase 2 normalized)
            # For now, use epoch directly - could be enhanced with proper normalization
            epoch_eff = float(epoch)

            # Extract method name from evaluator or use default
            method = getattr(self.evaluator.args, 'model', 'unknown') if hasattr(self.evaluator, 'args') else 'unknown'
            seed = getattr(self.evaluator.args, 'seed', 42) if hasattr(self.evaluator, 'args') else 42

            # Add row for each subset
            for split, acc in subset_accuracies.items():
                if split in self.data_loader.VALID_SPLITS:
                    csv_rows.append({
                        'method': method,
                        'seed': seed,
                        'epoch_eff': epoch_eff,
                        'split': split,
                        'acc': float(acc)
                    })

        if not csv_rows:
            self.logger.warning("No valid data rows generated for CSV export")
            return

        # Create DataFrame and save
        df = pd.DataFrame(csv_rows)

        # Sort for deterministic output
        df = df.sort_values(['method', 'seed', 'epoch_eff', 'split']).reset_index(drop=True)

        # Save CSV
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)

        # Save metadata sidecar
        metadata = {
            'export_timestamp': pd.Timestamp.now().isoformat(),
            'total_rows': len(df),
            'methods': sorted(df['method'].unique().tolist()),
            'seeds': sorted(df['seed'].unique().tolist()),
            'splits': sorted(df['split'].unique().tolist()),
            'epoch_range': [float(df['epoch_eff'].min()), float(df['epoch_eff'].max())],
            'evaluator_config': {
                'dataset_name': getattr(self.evaluator, 'dataset_name', 'unknown'),
                'adaptation_threshold': getattr(self.evaluator, 'adaptation_threshold', 0.8),
                'extract_attention': getattr(self.evaluator, 'extract_attention', False)
            }
        }

        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Exported {len(csv_rows)} timeline entries to {filepath}")
        self.logger.info(f"Saved metadata to {metadata_path}")

    def generate_visualizations_from_evaluator(self, output_dir: str) -> Dict[str, str]:
        """
        Generate visualizations using data from the existing EinstellungEvaluator.

        Creates both dynamics plots and robustness heatmaps from the evaluator's
        timeline data.

        Args:
            output_dir: Directory to save visualization files

        Returns:
            Dictionary mapping visualization names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        try:
            # First export timeline data to temporary CSV
            csv_path = output_dir / "temp_timeline_data.csv"
            self.export_timeline_for_visualization(str(csv_path))

            if not csv_path.exists():
                self.logger.error("Failed to export timeline data")
                return generated_files

            # Load the CSV data
            dataset = self.data_loader.load_csv(csv_path)

            # Generate dynamics visualization
            try:
                dynamics_fig_path = output_dir / "fig_eri_dynamics.pdf"

                # Process data for dynamics plot - compute all curves from full dataset
                all_curves = self.processor.compute_accuracy_curves(dataset)

                # Filter curves by split type
                patched_curves = {k: v for k, v in all_curves.items() if v.split == 'T2_shortcut_normal'}
                masked_curves = {k: v for k, v in all_curves.items() if v.split == 'T2_shortcut_masked'}

                if patched_curves and masked_curves:
                    # Compute derived metrics
                    ad_values = self.processor.compute_adaptation_delays(patched_curves)
                    pd_series = self.processor.compute_performance_deficits(patched_curves)
                    sfr_series = self.processor.compute_sfr_relative(all_curves)

                    # Create dynamics figure
                    fig = self.dynamics_plotter.create_dynamics_figure(
                        patched_curves=patched_curves,
                        masked_curves=masked_curves,
                        pd_series=pd_series,
                        sfr_series=sfr_series,
                        ad_values=ad_values,
                        tau=self.processor.tau
                    )

                    # Save figure
                    fig.savefig(dynamics_fig_path, dpi=self.style_config.dpi, bbox_inches='tight')
                    generated_files['dynamics'] = str(dynamics_fig_path)
                    self.logger.info(f"Generated dynamics plot: {dynamics_fig_path}")

            except Exception as e:
                self.logger.error(f"Failed to generate dynamics plot: {e}")

            # Generate robustness heatmap
            try:
                heatmap_fig_path = output_dir / "fig_ad_tau_heatmap.pdf"

                # Compute tau sensitivity for heatmap
                tau_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
                sensitivity_result = self.heatmap_plotter.compute_tau_sensitivity(
                    all_curves, tau_range
                )

                if sensitivity_result is not None:
                    # Create heatmap figure
                    fig = self.heatmap_plotter.create_tau_sensitivity_heatmap(sensitivity_result)

                    # Save figure
                    fig.savefig(heatmap_fig_path, dpi=self.style_config.dpi, bbox_inches='tight')
                    generated_files['heatmap'] = str(heatmap_fig_path)
                    self.logger.info(f"Generated heatmap plot: {heatmap_fig_path}")

            except Exception as e:
                self.logger.error(f"Failed to generate heatmap plot: {e}")

            # Clean up temporary CSV
            if csv_path.exists():
                csv_path.unlink()

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")

        return generated_files

    def register_visualization_hooks(self) -> None:
        """
        Register visualization hooks with existing Mammoth training infrastructure.

        Extends existing meta_begin_task(), after_training_epoch(), and meta_end_task()
        hooks to automatically generate visualizations during training.
        """
        if self.evaluator is None:
            self.logger.warning("No evaluator provided - cannot register hooks")
            return

        # Store original hook methods
        original_after_epoch = getattr(self.evaluator, 'after_training_epoch', None)
        original_end_task = getattr(self.evaluator, 'meta_end_task', None)

        def enhanced_after_training_epoch(model, dataset, epoch: int):
            """Enhanced after_training_epoch hook with visualization export."""
            # Call original hook first
            if original_after_epoch:
                original_after_epoch(model, dataset, epoch)

            # Auto-export if enabled and frequency matches
            if (self.auto_export_enabled and
                epoch % self.export_frequency == 0 and
                epoch != self.last_export_epoch):

                try:
                    csv_path = self.output_dir / f"eri_timeline_epoch_{epoch}.csv"
                    self.export_timeline_for_visualization(str(csv_path))
                    self.last_export_epoch = epoch

                except Exception as e:
                    self.logger.error(f"Auto-export failed at epoch {epoch}: {e}")

        def enhanced_meta_end_task(model, dataset):
            """Enhanced meta_end_task hook with final visualization generation."""
            # Call original hook first
            if original_end_task:
                original_end_task(model, dataset)

            # Generate final visualizations
            if self.auto_export_enabled:
                try:
                    # Export final timeline
                    csv_path = self.output_dir / "eri_final_timeline.csv"
                    self.export_timeline_for_visualization(str(csv_path))

                    # Generate visualizations
                    figs_dir = self.output_dir / "figs"
                    generated = self.generate_visualizations_from_evaluator(str(figs_dir))

                    if generated:
                        self.logger.info(f"Generated final visualizations: {list(generated.keys())}")
                    else:
                        self.logger.warning("No visualizations generated at task end")

                except Exception as e:
                    self.logger.error(f"Final visualization generation failed: {e}")

        # Replace hooks
        self.evaluator.after_training_epoch = enhanced_after_training_epoch
        self.evaluator.meta_end_task = enhanced_meta_end_task

        self.logger.info("Registered visualization hooks with EinstellungEvaluator")

    def load_from_mammoth_results(self, results_dir: str) -> Optional[ERITimelineDataset]:
        """
        Load ERI data from Mammoth experiment results directory.

        Scans for JSON export files from EinstellungEvaluator and converts
        them to ERITimelineDataset format.

        Args:
            results_dir: Directory containing Mammoth experiment results

        Returns:
            ERITimelineDataset if data found, None otherwise
        """
        results_dir = Path(results_dir)

        if not results_dir.exists():
            self.logger.error(f"Results directory not found: {results_dir}")
            return None

        # Look for EinstellungEvaluator export files
        json_files = list(results_dir.glob("**/einstellung_results.json"))
        csv_files = list(results_dir.glob("**/eri_*.csv"))

        if csv_files:
            # Prefer CSV files if available
            csv_file = csv_files[0]
            self.logger.info(f"Loading ERI data from CSV: {csv_file}")
            return self.data_loader.load_csv(csv_file)

        elif json_files:
            # Load from JSON export
            json_file = json_files[0]
            self.logger.info(f"Loading ERI data from JSON export: {json_file}")

            try:
                with open(json_file, 'r') as f:
                    export_data = json.load(f)
                return self.data_loader.load_from_evaluator_export(export_data)

            except Exception as e:
                self.logger.error(f"Failed to load JSON export: {e}")
                return None

        else:
            self.logger.warning(f"No ERI data files found in {results_dir}")
            return None

    def create_integration_config(self,
                                config_path: str,
                                methods: List[str],
                                seeds: List[int],
                                tau: float = 0.6,
                                smoothing_window: int = 3) -> None:
        """
        Create configuration file for Mammoth-ERI integration.

        Args:
            config_path: Path to save configuration
            methods: List of continual learning methods to include
            seeds: List of random seeds
            tau: Adaptation threshold for AD calculation
            smoothing_window: Window size for curve smoothing
        """
        config = {
            'integration': {
                'auto_export': True,
                'export_frequency': 1,
                'generate_visualizations': True
            },
            'processing': {
                'tau': tau,
                'smoothing_window': smoothing_window,
                'tau_grid': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            },
            'experiment': {
                'methods': methods,
                'seeds': seeds,
                'evaluation_subsets': [
                    'T1_all',
                    'T2_shortcut_normal',
                    'T2_shortcut_masked',
                    'T2_nonshortcut_normal'
                ]
            },
            'visualization': {
                'figure_size': list(self.style_config.figure_size),
                'dpi': self.style_config.dpi,
                'color_palette': self.style_config.color_palette,
                'confidence_alpha': self.style_config.confidence_alpha
            }
        }

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Created integration config: {config_path}")


def create_mammoth_integration(evaluator,
                             output_dir: str = "logs",
                             auto_setup: bool = True) -> MammothERIIntegration:
    """
    Factory function to create and configure MammothERIIntegration.

    Args:
        evaluator: EinstellungEvaluator instance
        output_dir: Base output directory
        auto_setup: Whether to automatically setup hooks and export

    Returns:
        Configured MammothERIIntegration instance
    """
    integration = MammothERIIntegration(evaluator, output_dir)

    if auto_setup:
        integration.setup_auto_export(output_dir)
        integration.register_visualization_hooks()

    return integration
