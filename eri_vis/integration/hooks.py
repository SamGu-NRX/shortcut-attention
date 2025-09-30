"""
ERIExperimentHooks - Experiment Lifecycle Integration

This module provides structured callbacks for integrating ERI visualization
with the Mammoth experiment lifecycle. It hooks into the existing training
pipeline to collect metrics and generate visualizations at appropriate times.

Key Integration Points:
- on_epoch_end: Collect shortcut patched/masked accuracies after each epoch
- on_task_end: Flush partial exports and cleanup
- on_experiment_end: Final export and visualization generation

CRITICAL: This module works with existing EinstellungEvaluator infrastructure.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ..data_loader import ERIDataLoader
from ..dataset import ERITimelineDataset
from ..processing import ERITimelineProcessor
from ..plot_dynamics import ERIDynamicsPlotter
from ..plot_heatmap import ERIHeatmapPlotter
from ..styles import PlotStyleConfig


class ERIExperimentHooks:
    """
    Structured experiment lifecycle hooks for ERI visualization integration.

    Provides callbacks that integrate with Mammoth's training pipeline to:
    - Collect timeline metrics during training
    - Export data at appropriate intervals
    - Generate visualizations at experiment completion
    """

    def __init__(self,
                 output_dir: str = "logs",
                 style_config: Optional[PlotStyleConfig] = None,
                 export_frequency: int = 1,
                 auto_visualize: bool = True):
        """
        Initialize ERI experiment hooks.

        Args:
            output_dir: Base directory for outputs
            style_config: Plot styling configuration
            export_frequency: Export CSV every N epochs (1 = every epoch)
            auto_visualize: Whether to automatically generate visualizations
        """
        self.output_dir = Path(output_dir)
        self.style_config = style_config or PlotStyleConfig()
        self.export_frequency = export_frequency
        self.auto_visualize = auto_visualize

        # Initialize components
        self.data_loader = ERIDataLoader()
        self.processor = ERITimelineProcessor()
        self.dynamics_plotter = ERIDynamicsPlotter(self.style_config)
        self.heatmap_plotter = ERIHeatmapPlotter(self.style_config)

        # State tracking
        self.timeline_data = []
        self.current_task = 0
        self.current_method = "unknown"
        self.current_seed = 42
        self.last_export_epoch = -1

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figs").mkdir(exist_ok=True)

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ERI hooks: output_dir={output_dir}, freq={export_frequency}")

    def on_epoch_end(self, epoch: int, evaluator) -> None:
        """
        Callback invoked at the end of each training epoch.

        Collects shortcut patched and masked accuracies from the evaluator
        and stores them for timeline analysis.

        Args:
            epoch: Current epoch number
            evaluator: EinstellungEvaluator instance with timeline data
        """
        if evaluator is None:
            self.logger.debug(f"No evaluator provided for epoch {epoch}")
            return

        try:
            # Extract method and seed information
            if hasattr(evaluator, 'args'):
                self.current_method = getattr(evaluator.args, 'model', 'unknown')
                self.current_seed = getattr(evaluator.args, 'seed', 42)

            # Get the latest timeline entry from evaluator
            if hasattr(evaluator, 'timeline_data') and evaluator.timeline_data:
                latest_entry = evaluator.timeline_data[-1]

                # Verify this is the current epoch
                if latest_entry.get('epoch') == epoch:
                    # Extract subset metrics (supports legacy structure)
                    subset_metrics = latest_entry.get('subset_metrics') or latest_entry.get('subset_accuracies', {})
                    subset_losses = latest_entry.get('subset_losses', {})

                    epoch_eff = latest_entry.get('epoch_eff')
                    if epoch_eff is None:
                        # Skip pre-Phase-2 evaluations for ERI visualisation
                        return

                    def _top1(value):
                        if isinstance(value, dict):
                            return float(value.get('top1', value.get('accuracy', 0.0)))
                        return float(value)

                    def _top5(value):
                        if isinstance(value, dict):
                            top5_val = value.get('top5')
                            if top5_val is not None:
                                return float(top5_val)
                        return None

                    subset_accuracies_map = {
                        split: _top1(metrics)
                        for split, metrics in subset_metrics.items()
                    }

                    # Focus on shortcut-related metrics
                    sc_patched = subset_accuracies_map.get('T2_shortcut_normal', 0.0)
                    sc_masked = subset_accuracies_map.get('T2_shortcut_masked', 0.0)

                    # Store timeline entry in our format
                    timeline_entry = {
                        'epoch': epoch,
                        'epoch_eff': float(epoch_eff),
                        'task_id': self.current_task,
                        'method': self.current_method,
                        'seed': self.current_seed,
                        'subset_metrics': subset_metrics,
                        'subset_accuracies': subset_accuracies_map,
                        'subset_losses': subset_losses,
                        'timestamp': time.time()
                    }

                    self.timeline_data.append(timeline_entry)

                    # Log key metrics
                    if sc_patched > 0 or sc_masked > 0:
                        deficit = (sc_patched - sc_masked) / max(sc_patched, 1e-8)
                        self.logger.debug(
                            f"Epoch {epoch}: SC_patched={sc_patched:.4f}, "
                            f"SC_masked={sc_masked:.4f}, deficit={deficit:.4f}"
                        )

                    # Export CSV if frequency matches
                    if (self.export_frequency > 0 and
                        epoch % self.export_frequency == 0 and
                        epoch != self.last_export_epoch):

                        self._export_partial_csv(epoch)
                        self.last_export_epoch = epoch

        except Exception as e:
            self.logger.error(f"Error in on_epoch_end for epoch {epoch}: {e}")

    def on_task_end(self, task_id: int, evaluator) -> None:
        """
        Callback invoked at the end of each task.

        Flushes partial exports and performs cleanup operations.

        Args:
            task_id: ID of the completed task
            evaluator: EinstellungEvaluator instance
        """
        self.current_task = task_id

        try:
            self.logger.info(f"Task {task_id} completed - flushing partial exports")

            # Export current timeline data
            if self.timeline_data:
                csv_path = self.output_dir / f"eri_task_{task_id}_timeline.csv"
                self._export_timeline_csv(csv_path)

                # Log summary statistics
                total_epochs = len(self.timeline_data)
                if total_epochs > 0:
                    latest_entry = self.timeline_data[-1]
                    subset_accs = latest_entry.get('subset_accuracies', {})

                    self.logger.info(
                        f"Task {task_id} summary: {total_epochs} epochs, "
                        f"final T1_all={subset_accs.get('T1_all', 0.0):.4f}, "
                        f"T2_SC_normal={subset_accs.get('T2_shortcut_normal', 0.0):.4f}"
                    )

            # Cleanup: could implement memory management here if needed
            # For now, keep all data for final visualization

        except Exception as e:
            self.logger.error(f"Error in on_task_end for task {task_id}: {e}")

    def on_experiment_end(self, evaluator) -> Dict[str, str]:
        """
        Callback invoked at the end of the entire experiment.

        Performs final export of all timeline data and generates
        comprehensive visualizations.

        Args:
            evaluator: EinstellungEvaluator instance

        Returns:
            Dictionary mapping visualization names to file paths
        """
        generated_files = {}

        try:
            self.logger.info("Experiment completed - generating final exports and visualizations")

            # Export final comprehensive CSV
            method_name = (self.current_method or 'method').replace('/', '_')
            final_csv_path = self.output_dir / f"timeline_{method_name}.csv"
            self._export_timeline_csv(final_csv_path)
            generated_files['csv'] = str(final_csv_path)

            # Generate visualizations if enabled
            if self.auto_visualize and self.timeline_data:
                viz_files = self._generate_final_visualizations()
                generated_files.update(viz_files)

            # Export experiment metadata
            metadata_path = self.output_dir / "eri_experiment_metadata.json"
            self._export_experiment_metadata(metadata_path)
            generated_files['metadata'] = str(metadata_path)

            self.logger.info(f"Final export complete: {list(generated_files.keys())}")

        except Exception as e:
            self.logger.error(f"Error in on_experiment_end: {e}")

        return generated_files

    def _export_partial_csv(self, epoch: int) -> None:
        """Export partial CSV for current epoch."""
        try:
            csv_path = self.output_dir / f"eri_partial_epoch_{epoch}.csv"
            self._export_timeline_csv(csv_path)
            self.logger.debug(f"Exported partial CSV for epoch {epoch}")
        except Exception as e:
            self.logger.error(f"Failed to export partial CSV for epoch {epoch}: {e}")

    def _export_timeline_csv(self, csv_path: Path) -> None:
        """
        Export timeline data to CSV in the required format.

        Converts internal timeline data to the CSV schema expected
        by the visualization system.
        """
        if not self.timeline_data:
            self.logger.warning("No timeline data to export")
            return

        # Convert to CSV rows
        csv_rows = []

        for entry in self.timeline_data:
            epoch_eff = entry.get('epoch_eff', entry.get('epoch', 0))
            method = entry.get('method', self.current_method)
            seed = entry.get('seed', self.current_seed)
            subset_metrics = entry.get('subset_metrics') or entry.get('subset_accuracies', {})
            subset_losses = entry.get('subset_losses', {})

            for split, metrics in subset_metrics.items():
                if split not in self.data_loader.VALID_SPLITS:
                    continue

                if isinstance(metrics, dict):
                    top1 = float(metrics.get('top1', metrics.get('accuracy', 0.0)))
                    top5 = metrics.get('top5')
                    top5 = float(top5) if top5 is not None else top1
                else:
                    top1 = float(metrics)
                    top5 = top1

                loss = subset_losses.get(split)
                loss = float(loss) if loss is not None else max(0.0, 1.0 - top1)

                csv_rows.append({
                    'method': method,
                    'seed': seed,
                    'epoch_eff': float(epoch_eff),
                    'split': split,
                    'acc': top1,
                    'top5': top5,
                    'loss': loss
                })

        if not csv_rows:
            self.logger.warning("No valid CSV rows generated")
            return

        # Create DataFrame and save
        df = pd.DataFrame(csv_rows)

        # Sort for deterministic output
        df = df.sort_values(['method', 'seed', 'epoch_eff', 'split']).reset_index(drop=True)

        # Ensure parent directory exists
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            self.logger.error(f"Cannot create directory {csv_path.parent}: {e}")
            return

        # Save CSV
        try:
            df.to_csv(csv_path, index=False)
        except (OSError, PermissionError) as e:
            self.logger.error(f"Cannot write CSV to {csv_path}: {e}")
            return

        self.logger.info(f"Exported {len(csv_rows)} timeline entries to {csv_path}")

    def _generate_final_visualizations(self) -> Dict[str, str]:
        """Generate final visualizations from collected timeline data."""
        generated_files = {}

        try:
            # Create temporary CSV for visualization processing
            temp_csv = self.output_dir / "temp_final_timeline.csv"
            self._export_timeline_csv(temp_csv)

            if not temp_csv.exists():
                self.logger.error("Failed to create temporary CSV for visualization")
                return generated_files

            # Load data using our data loader
            dataset = self.data_loader.load_csv(temp_csv)

            # Generate dynamics visualization
            dynamics_path = self._generate_dynamics_plot(dataset)
            if dynamics_path:
                generated_files['dynamics'] = dynamics_path

            # Generate robustness heatmap
            heatmap_path = self._generate_heatmap_plot(dataset)
            if heatmap_path:
                generated_files['heatmap'] = heatmap_path

            # Clean up temporary file
            if temp_csv.exists():
                temp_csv.unlink()

        except Exception as e:
            self.logger.error(f"Error generating final visualizations: {e}")

        return generated_files

    def _generate_dynamics_plot(self, dataset: ERITimelineDataset) -> Optional[str]:
        """Generate dynamics plot from dataset."""
        try:
            # Compute accuracy curves
            all_curves = self.processor.compute_accuracy_curves(dataset)

            # Filter curves by split type
            patched_curves = {k: v for k, v in all_curves.items() if v.split == 'T2_shortcut_normal'}
            masked_curves = {k: v for k, v in all_curves.items() if v.split == 'T2_shortcut_masked'}

            if not patched_curves or not masked_curves:
                self.logger.warning("Insufficient data for dynamics plot")
                return None

            # Compute derived metrics
            ad_values = self.processor.compute_adaptation_delays(patched_curves)
            pd_series = self.processor.compute_performance_deficits(patched_curves)
            sfr_series = self.processor.compute_sfr_relative(all_curves)

            # Create figure
            fig = self.dynamics_plotter.create_dynamics_figure(
                patched_curves=patched_curves,
                masked_curves=masked_curves,
                pd_series=pd_series,
                sfr_series=sfr_series,
                ad_values=ad_values,
                tau=self.processor.tau
            )

            # Save figure
            fig_path = self.output_dir / "figs" / "fig_eri_dynamics.pdf"
            fig.savefig(fig_path, dpi=self.style_config.dpi, bbox_inches='tight')

            self.logger.info(f"Generated dynamics plot: {fig_path}")
            return str(fig_path)

        except Exception as e:
            self.logger.error(f"Failed to generate dynamics plot: {e}")
            return None

    def _generate_heatmap_plot(self, dataset: ERITimelineDataset) -> Optional[str]:
        """Generate robustness heatmap from dataset."""
        try:
            # Compute accuracy curves
            all_curves = self.processor.compute_accuracy_curves(dataset)

            # Compute tau sensitivity
            tau_range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
            sensitivity_result = self.heatmap_plotter.compute_tau_sensitivity(
                all_curves, tau_range
            )

            if sensitivity_result is None:
                self.logger.warning("Could not compute tau sensitivity for heatmap")
                return None

            # Create heatmap figure
            fig = self.heatmap_plotter.create_tau_sensitivity_heatmap(sensitivity_result)

            # Save figure
            fig_path = self.output_dir / "figs" / "fig_ad_tau_heatmap.pdf"
            fig.savefig(fig_path, dpi=self.style_config.dpi, bbox_inches='tight')

            self.logger.info(f"Generated heatmap plot: {fig_path}")
            return str(fig_path)

        except Exception as e:
            self.logger.error(f"Failed to generate heatmap plot: {e}")
            return None

    def _export_experiment_metadata(self, metadata_path: Path) -> None:
        """Export experiment metadata."""
        try:
            metadata = {
                'experiment_info': {
                    'total_epochs': len(self.timeline_data),
                    'method': self.current_method,
                    'seed': self.current_seed,
                    'tasks_completed': self.current_task + 1,
                    'export_frequency': self.export_frequency,
                    'auto_visualize': self.auto_visualize
                },
                'processing_config': {
                    'tau': self.processor.tau,
                    'smoothing_window': self.processor.smoothing_window
                },
                'style_config': {
                    'figure_size': list(self.style_config.figure_size),
                    'dpi': self.style_config.dpi,
                    'confidence_alpha': self.style_config.confidence_alpha
                },
                'timeline_summary': {
                    'total_entries': len(self.timeline_data),
                    'epoch_range': [
                        min(entry['epoch'] for entry in self.timeline_data),
                        max(entry['epoch'] for entry in self.timeline_data)
                    ] if self.timeline_data else [0, 0]
                },
                'export_timestamp': pd.Timestamp.now().isoformat()
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Exported experiment metadata to {metadata_path}")

        except Exception as e:
            self.logger.error(f"Failed to export metadata: {e}")

    def get_timeline_summary(self) -> Dict[str, Any]:
        """Get summary of collected timeline data."""
        if not self.timeline_data:
            return {'total_entries': 0, 'epoch_range': [0, 0]}

        epochs = [entry['epoch'] for entry in self.timeline_data]

        return {
            'total_entries': len(self.timeline_data),
            'epoch_range': [min(epochs), max(epochs)],
            'method': self.current_method,
            'seed': self.current_seed,
            'tasks': self.current_task + 1
        }

    def reset(self) -> None:
        """Reset hook state for new experiment."""
        self.timeline_data.clear()
        self.current_task = 0
        self.last_export_epoch = -1
        self.logger.info("Reset ERI experiment hooks")


def create_eri_hooks(output_dir: str = "logs",
                    export_frequency: int = 1,
                    auto_visualize: bool = True,
                    style_config: Optional[PlotStyleConfig] = None) -> ERIExperimentHooks:
    """
    Factory function to create ERIExperimentHooks instance.

    Args:
        output_dir: Base directory for outputs
        export_frequency: Export CSV every N epochs
        auto_visualize: Whether to automatically generate visualizations
        style_config: Plot styling configuration

    Returns:
        Configured ERIExperimentHooks instance
    """
    return ERIExperimentHooks(
        output_dir=output_dir,
        style_config=style_config,
        export_frequency=export_frequency,
        auto_visualize=auto_visualize
    )


def integrate_hooks_with_evaluator(evaluator, hooks: ERIExperimentHooks) -> None:
    """
    Integrate ERIExperimentHooks with an existing EinstellungEvaluator.

    This function patches the evaluator's hook methods to call our
    structured callbacks.

    Args:
        evaluator: EinstellungEvaluator instance
        hooks: ERIExperimentHooks instance
    """
    if evaluator is None:
        raise ValueError("Evaluator cannot be None")

    # Store original methods
    original_after_epoch = getattr(evaluator, 'after_training_epoch', None)
    original_end_task = getattr(evaluator, 'meta_end_task', None)

    def enhanced_after_training_epoch(model, dataset, epoch: int):
        """Enhanced after_training_epoch with hooks integration."""
        # Call original method first
        if original_after_epoch:
            original_after_epoch(model, dataset, epoch)

        # Call our hook
        hooks.on_epoch_end(epoch, evaluator)

    def enhanced_meta_end_task(model, dataset):
        """Enhanced meta_end_task with hooks integration."""
        # Call original method first
        if original_end_task:
            original_end_task(model, dataset)

        # Call our hooks
        task_id = getattr(dataset, 'i', hooks.current_task)
        hooks.on_task_end(task_id, evaluator)

        # If this is the final task, call experiment end
        # This is a heuristic - could be enhanced with better detection
        if hasattr(dataset, 'N_TASKS') and task_id >= dataset.N_TASKS - 1:
            hooks.on_experiment_end(evaluator)

    # Patch the evaluator
    evaluator.after_training_epoch = enhanced_after_training_epoch
    evaluator.meta_end_task = enhanced_meta_end_task

    logging.getLogger(__name__).info("Integrated ERI hooks with EinstellungEvaluator")
