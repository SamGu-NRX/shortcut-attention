# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Einstellung Effect Evaluation Plugin

This module provides a plugin-based evaluator that integrates with Mammoth's
training pipeline to collect Einstellung Rigidity Index (ERI) metrics.

The evaluator hooks into the training process to:
- Evaluate multiple subsets (T1_all, T2_shortcut_normal, T2_shortcut_masked, etc.)
- Track timeline metrics for Adaptation Delay calculation
- Log metrics to Mammoth's standard logging system
- Export comprehensive results for analysis
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.einstellung_metrics import EinstellungMetricsCalculator, EinstellungTimelineData
from utils.einstellung_attention import EinstellungAttentionAnalyzer
from utils.evaluate import evaluate


class EinstellungEvaluator:
    """
    Plugin-based evaluator for Einstellung Effect experiments.

    Integrates with Mammoth's training pipeline through hooks to collect
    comprehensive metrics for analyzing cognitive rigidity in continual learning.
    """

    def __init__(self,
                 args,
                 dataset_name: str = 'seq-cifar100-einstellung',
                 adaptation_threshold: float = 0.8,
                 extract_attention: bool = True):
        """
        Initialize the Einstellung evaluator.

        Args:
            args: Command line arguments from Mammoth
            dataset_name: Name of the dataset to monitor
            adaptation_threshold: Accuracy threshold for Adaptation Delay calculation
            extract_attention: Whether to extract attention maps (ViT only)
        """
        self.args = args
        self.dataset_name = dataset_name
        self.adaptation_threshold = adaptation_threshold
        self.extract_attention = extract_attention

        # Initialize metrics calculator
        self.metrics_calculator = EinstellungMetricsCalculator(adaptation_threshold)

        # Initialize attention analyzer if enabled
        self.attention_analyzer = None
        if extract_attention:
            self.attention_analyzer = None  # Will be initialized with model

        # Timeline storage
        self.timeline_data = []
        self.current_task = 0

        # Logging
        self.logger = logging.getLogger(__name__)

        # Configuration from args
        self.evaluation_subsets = getattr(args, 'einstellung_evaluation_subsets', True)

    def initialize_attention_analyzer(self, model):
        """Initialize attention analyzer with the model (called after model is available)."""
        if self.extract_attention and self.attention_analyzer is None:
            # Double-check model compatibility before initialization
            if not self._is_vision_transformer(model):
                self.logger.warning("⚠️  Model does not support attention extraction (not a Vision Transformer)")
                self.logger.warning("   Disabling attention analysis for this experiment")
                self.extract_attention = False
                return

            try:
                self.attention_analyzer = EinstellungAttentionAnalyzer(model, self.args.device)
                self.logger.info("✅ Initialized Einstellung attention analyzer for ViT model")
            except Exception as e:
                self.logger.warning(f"❌ Could not initialize attention analyzer: {e}")
                self.logger.warning("   Continuing without attention analysis")
                self.attention_analyzer = None
                self.extract_attention = False
        elif not self.extract_attention:
            self.logger.info("ℹ️  Attention extraction disabled - using CNN backbone (ResNet/etc.)")
            self.logger.info("   Attention analysis is only available for Vision Transformer models")

    def meta_begin_task(self, model, dataset):
        """Hook called at the beginning of each task."""
        if dataset.NAME == self.dataset_name:
            self.current_task = dataset.i if hasattr(dataset, 'i') else 0
            self.logger.info(f"Starting Einstellung evaluation for task {self.current_task}")

            # Initialize attention analyzer if not done yet
            if self.extract_attention and self.attention_analyzer is None:
                self.initialize_attention_analyzer(model)

    def meta_end_task(self, model, dataset):
        """Hook called at the end of each task."""
        if dataset.NAME == self.dataset_name:
            self.logger.info(f"Completed Einstellung evaluation for task {self.current_task}")

            # Perform comprehensive evaluation at task end
            self._comprehensive_task_evaluation(model, dataset)

    def after_training_epoch(self, model, dataset, epoch: int):
        """
        Hook called after each training epoch.
        OPTIMIZED: Reduces frequency and computational load to prevent ViT timeouts.

        Args:
            model: The continual learning model
            dataset: The dataset being used for training
            epoch: Current epoch number
        """
        if dataset.NAME != self.dataset_name:
            return

        if not self.evaluation_subsets:
            return

        # OPTIMIZATION: Skip expensive evaluations on most epochs for ALL models
        is_vit_model = self._is_vit_model(model)

        # Use actual training epochs from args, not dataset default
        actual_epochs = getattr(self.args, 'n_epochs', dataset.get_epochs())

        # Determine evaluation frequency based on actual epochs being run
        if actual_epochs <= 5:
            # For very short training (debug mode), evaluate every epoch
            eval_frequency = 1
        elif actual_epochs <= 20:
            # For short training, evaluate every 5th epoch
            eval_frequency = 5
        else:
            # For long training, evaluate every 10th epoch
            eval_frequency = 10

        # Always evaluate on the final epoch
        is_final_epoch = (epoch == actual_epochs - 1)
        should_evaluate = (epoch % eval_frequency == 0) or is_final_epoch

        self.logger.info(f"🔍 Evaluation frequency check: epoch={epoch}, actual_epochs={actual_epochs}, eval_frequency={eval_frequency}, should_evaluate={should_evaluate}")

        if not should_evaluate:
            self.logger.debug(f"Skipping evaluation for epoch {epoch} (performance optimization - evaluating every {eval_frequency} epochs)")
            return

        # Create evaluation subsets
        evaluation_subsets = self._get_evaluation_subsets(dataset)

        if not evaluation_subsets:
            self.logger.warning(f"❌ No evaluation subsets available for epoch {epoch}")
            self.logger.warning(f"   Dataset: {dataset.NAME if hasattr(dataset, 'NAME') else type(dataset).__name__}")
            self.logger.warning(f"   Has get_evaluation_subsets: {hasattr(dataset, 'get_evaluation_subsets')}")
            return

        self.logger.info(f"📊 Found {len(evaluation_subsets)} evaluation subsets: {list(evaluation_subsets.keys())}")

        # Evaluate each subset
        subset_accuracies = {}
        subset_losses = {}
        attention_metrics = {}

        model.net.eval()

        # OPTIMIZATION: Use memory-efficient evaluation
        with torch.no_grad():
            for subset_name, subset_loader in evaluation_subsets.items():
                # Basic accuracy evaluation (always run)
                acc, loss = self._evaluate_subset(model, subset_loader)
                subset_accuracies[subset_name] = acc
                subset_losses[subset_name] = loss

                # OPTIMIZATION: Attention analysis with strict controls
                if self.attention_analyzer and subset_loader:
                    try:
                        self.logger.debug(f"Running attention analysis for {subset_name} (epoch {epoch})")

                        # Get a small batch for attention analysis
                        batch_inputs, _ = next(iter(subset_loader))

                        # OPTIMIZATION: Use limited batch size for ViT
                        if is_vit_model and batch_inputs.shape[0] > 8:
                            batch_inputs = batch_inputs[:8]

                        attn_metrics = self.attention_analyzer.analyze_einstellung_attention_batch(
                            batch_inputs, subset_name, epoch
                        )

                        if attn_metrics:
                            attention_metrics.update({f"{subset_name}_{k}": v for k, v in attn_metrics.items()})
                            self.logger.debug(f"Extracted {len(attn_metrics)} attention metrics for {subset_name}")

                    except Exception as e:
                        self.logger.debug(f"Could not extract attention for {subset_name}: {e}")

        # Store timeline data
        self.metrics_calculator.add_timeline_data(
            epoch=epoch,
            task_id=self.current_task,
            subset_accuracies=subset_accuracies,
            subset_losses=subset_losses,
            timestamp=time.time()
        )

        # Log to Mammoth's logging system
        self._log_metrics_to_mammoth(model, epoch, subset_accuracies, attention_metrics)

        # Store for later analysis
        timeline_entry = {
            'epoch': epoch,
            'task_id': self.current_task,
            'subset_accuracies': subset_accuracies,
            'subset_losses': subset_losses,
            'attention_metrics': attention_metrics,
            'timestamp': time.time()
        }
        self.timeline_data.append(timeline_entry)

        # DEBUG: Log timeline data storage
        self.logger.info(f"📊 Stored timeline entry: task={self.current_task}, epoch={epoch}, subsets={len(subset_accuracies)}")
        self.logger.info(f"📊 Total timeline entries: {len(self.timeline_data)}")

        # Log evaluation completion
        self.logger.info(f"✓ Einstellung evaluation completed for epoch {epoch}")

    def _get_evaluation_subsets(self, dataset) -> Dict[str, DataLoader]:
        """
        Get evaluation subsets for Einstellung analysis.

        Returns:
            Dictionary mapping subset names to DataLoaders
        """
        if not hasattr(dataset, 'get_evaluation_subsets'):
            self.logger.warning("❌ Dataset does not support evaluation subsets")
            self.logger.warning(f"   Dataset type: {type(dataset).__name__}")
            self.logger.warning(f"   Dataset NAME: {getattr(dataset, 'NAME', 'Unknown')}")
            return {}

        try:
            self.logger.debug(f"🔍 Calling dataset.get_evaluation_subsets() on {type(dataset).__name__}")
            subsets = dataset.get_evaluation_subsets()
            self.logger.debug(f"✅ Got {len(subsets)} evaluation subsets: {list(subsets.keys())}")

            # Validate that subsets contain data
            for subset_name, loader in subsets.items():
                if loader is None:
                    self.logger.warning(f"⚠️  Subset '{subset_name}' has None loader")
                else:
                    try:
                        # Try to get the length to validate the loader
                        subset_len = len(loader.dataset) if hasattr(loader, 'dataset') else 'Unknown'
                        self.logger.debug(f"   Subset '{subset_name}': {subset_len} samples")
                    except Exception as e:
                        self.logger.debug(f"   Subset '{subset_name}': Could not determine size ({e})")

            return subsets
        except Exception as e:
            self.logger.error(f"❌ Error getting evaluation subsets: {e}")
            self.logger.exception("Full traceback:")
            return {}

    def _evaluate_subset(self, model, subset_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on a specific subset.

        Args:
            model: The continual learning model
            subset_loader: DataLoader for the subset

        Returns:
            Tuple of (accuracy, average_loss)
        """
        if not subset_loader:
            return 0.0, 0.0

        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        model.net.eval()
        with torch.no_grad():
            for data in subset_loader:
                try:
                    if len(data) == 2:
                        inputs, labels = data
                    elif len(data) == 3:
                        inputs, labels, _ = data
                    else:
                        self.logger.warning(f"Unexpected data format: {len(data)} elements")
                        continue

                    inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)

                    # Forward pass
                    # Use the model's forward to obtain unified logits even for
                    # multi-head continual models (e.g., GPM).
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Handle models that return tuples

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()

                    # Calculate loss
                    loss = model.loss(outputs, labels)
                    total_loss += loss.item() * labels.size(0)

                except Exception as e:
                    self.logger.debug(f"Error in subset evaluation: {e}")
                    continue

        accuracy = total_correct / max(total_samples, 1)
        avg_loss = total_loss / max(total_samples, 1)

        return accuracy, avg_loss

    def _log_metrics_to_mammoth(self,
                              model,
                              epoch: int,
                              subset_accuracies: Dict[str, float],
                              attention_metrics: Dict[str, float]):
        """
        Log metrics to Mammoth's logging system.

        Args:
            model: The continual learning model
            epoch: Current epoch
            subset_accuracies: Dictionary of subset accuracies
            attention_metrics: Dictionary of attention metrics
        """
        # Log to TensorBoard/WandB if available
        if hasattr(model, 'writer') and model.writer is not None:
            for subset_name, accuracy in subset_accuracies.items():
                model.writer.add_scalar(f'einstellung/accuracy/{subset_name}', accuracy, epoch)

            for metric_name, value in attention_metrics.items():
                model.writer.add_scalar(f'einstellung/attention/{metric_name}', value, epoch)

        # Log critical metrics at info level
        if 'T1_all' in subset_accuracies:
            self.logger.info(f"Epoch {epoch}: T1_all accuracy = {subset_accuracies['T1_all']:.4f}")

        if 'T2_shortcut_normal' in subset_accuracies and 'T2_shortcut_masked' in subset_accuracies:
            shortcut_acc = subset_accuracies['T2_shortcut_normal']
            masked_acc = subset_accuracies['T2_shortcut_masked']
            deficit = (shortcut_acc - masked_acc) / max(shortcut_acc, 1e-8)
            self.logger.info(f"Epoch {epoch}: Performance Deficit = {deficit:.4f}")

    def _comprehensive_task_evaluation(self, model, dataset):
        """Perform comprehensive evaluation at the end of a task."""
        self.logger.info(f"Performing comprehensive evaluation for task {self.current_task}")

        # Calculate current ERI metrics
        metrics = self.metrics_calculator.calculate_comprehensive_metrics()

        # Log key metrics
        if metrics.adaptation_delay is not None:
            self.logger.info(f"Adaptation Delay: {metrics.adaptation_delay} epochs")

        if metrics.performance_deficit is not None:
            self.logger.info(f"Performance Deficit: {metrics.performance_deficit:.4f}")

        if metrics.shortcut_feature_reliance is not None:
            self.logger.info(f"Shortcut Feature Reliance: {metrics.shortcut_feature_reliance:.4f}")

        if metrics.eri_score is not None:
            self.logger.info(f"ERI Score: {metrics.eri_score:.4f}")

    def get_final_metrics(self):
        """
        Get final ERI metrics after training completion.

        Returns:
            EinstellungMetrics object with comprehensive results
        """
        return self.metrics_calculator.calculate_comprehensive_metrics()

    def export_results(self, filepath: str):
        """
        Export comprehensive results to file.

        Args:
            filepath: Path to save results
        """
        # Get final metrics
        final_metrics = self.get_final_metrics()

        # Prepare export data
        export_data = {
            'configuration': {
                'dataset_name': self.dataset_name,
                'adaptation_threshold': self.adaptation_threshold,
                'extract_attention': self.extract_attention
            },
            'timeline_data': self.timeline_data,
            'metrics_timeline': self.metrics_calculator.export_timeline_data(),
            'final_metrics': {
                'adaptation_delay': final_metrics.adaptation_delay,
                'performance_deficit': final_metrics.performance_deficit,
                'shortcut_feature_reliance': final_metrics.shortcut_feature_reliance,
                'eri_score': final_metrics.eri_score,
                'task1_accuracy_final': final_metrics.task1_accuracy_final,
                'task2_shortcut_accuracy': final_metrics.task2_shortcut_accuracy,
                'task2_masked_accuracy': final_metrics.task2_masked_accuracy,
                'task2_nonshortcut_accuracy': final_metrics.task2_nonshortcut_accuracy
            }
        }

        # Add attention analysis if available
        if self.attention_analyzer:
            export_data['attention_analysis'] = self.attention_analyzer.get_attention_timeline_summary()

        # Export to JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Exported Einstellung results to {filepath}")

        # ALSO export CSV format for ERI visualization system
        csv_filepath = filepath.replace('.json', '_eri_sc_metrics.csv')
        self.export_csv_for_visualization(csv_filepath)

    def export_csv_for_visualization(self, csv_filepath: str):
        """
        Export timeline data in CSV format for ERI visualization system.

        Converts timeline data to the required CSV schema:
        method,seed,epoch_eff,split,acc

        Args:
            csv_filepath: Path to save CSV file
        """
        # DEBUG: Log timeline data status
        self.logger.info(f"📊 CSV Export Debug - Timeline data entries: {len(self.timeline_data)}")

        if not self.timeline_data:
            self.logger.warning("❌ No timeline data available for CSV export")
            self.logger.warning("   This indicates that Einstellung evaluation hooks are not collecting data properly")
            self.logger.warning("   Check that after_training_epoch is being called and evaluation subsets are working")

            # Create an empty CSV file to prevent synthetic data fallback
            try:
                import pandas as pd
                empty_df = pd.DataFrame(columns=['method', 'seed', 'epoch_eff', 'split', 'acc'])

                # Ensure parent directory exists
                import os
                os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)

                empty_df.to_csv(csv_filepath, index=False)
                self.logger.info(f"📄 Created empty CSV file to prevent synthetic data fallback: {csv_filepath}")
            except Exception as e:
                self.logger.error(f"❌ Could not create empty CSV file: {e}")

            return

        try:
            import pandas as pd
        except ImportError:
            self.logger.error("pandas not available for CSV export")
            return

        # Extract method and seed from args
        method = getattr(self.args, 'model', 'unknown') if hasattr(self, 'args') else 'unknown'
        seed = getattr(self.args, 'seed', 42) if hasattr(self, 'args') else 42

        # Convert timeline data to CSV rows
        csv_rows = []

        self.logger.info(f"📊 Processing {len(self.timeline_data)} timeline entries for CSV export...")

        for i, entry in enumerate(self.timeline_data):
            epoch = entry.get('epoch', 0)
            subset_accuracies = entry.get('subset_accuracies', {})

            # DEBUG: Log each entry
            self.logger.debug(f"   Entry {i}: epoch={epoch}, subsets={list(subset_accuracies.keys())}")

            # Use actual epoch number as epoch_eff (not fractional values)
            epoch_eff = float(epoch)

            # Add row for each valid subset
            for split, acc in subset_accuracies.items():
                # Only include valid splits for ERI visualization
                if split in ['T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal']:
                    csv_rows.append({
                        'method': method,
                        'seed': seed,
                        'epoch_eff': epoch_eff,
                        'split': split,
                        'acc': float(acc)
                    })

        if not csv_rows:
            self.logger.warning("❌ No valid CSV rows generated from timeline data")
            self.logger.warning("   Timeline data exists but contains no valid evaluation subsets")
            self.logger.warning("   Expected subsets: T1_all, T2_shortcut_normal, T2_shortcut_masked, T2_nonshortcut_normal")

            # DEBUG: Show what subsets we actually have
            all_subsets = set()
            for entry in self.timeline_data:
                all_subsets.update(entry.get('subset_accuracies', {}).keys())
            self.logger.warning(f"   Found subsets: {sorted(all_subsets)}")
            return

        # Create DataFrame and save
        df = pd.DataFrame(csv_rows)

        # Sort for deterministic output
        df = df.sort_values(['method', 'seed', 'epoch_eff', 'split']).reset_index(drop=True)

        # Ensure parent directory exists
        import os
        os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)

        # Save CSV
        df.to_csv(csv_filepath, index=False)

        self.logger.info(f"✅ Exported {len(csv_rows)} timeline entries to CSV: {csv_filepath}")

        # DEBUG: Show sample of exported data
        if len(csv_rows) > 0:
            sample_epochs = sorted(df['epoch_eff'].unique())[:5]
            self.logger.info(f"   Sample epochs: {sample_epochs}")
            sample_splits = sorted(df['split'].unique())
            self.logger.info(f"   Splits: {sample_splits}")

    def get_accuracy_curves(self) -> Dict[str, Tuple[List[int], List[float]]]:
        """
        Get accuracy curves for visualization.

        Returns:
            Dictionary mapping subset names to (epochs, accuracies) tuples
        """
        curves = {}

        # Group timeline data by subset
        subset_data = {}
        for entry in self.timeline_data:
            epoch = entry['epoch']
            for subset_name, accuracy in entry['subset_accuracies'].items():
                if subset_name not in subset_data:
                    subset_data[subset_name] = {'epochs': [], 'accuracies': []}
                subset_data[subset_name]['epochs'].append(epoch)
                subset_data[subset_name]['accuracies'].append(accuracy)

        # Convert to required format
        for subset_name, data in subset_data.items():
            curves[subset_name] = (data['epochs'], data['accuracies'])

        return curves

    def _is_vit_model(self, model) -> bool:
        """
        Check if the model uses a Vision Transformer backbone.
        """
        try:
            # Check backbone type through various paths
            backbone = None
            if hasattr(model, 'net') and hasattr(model.net, 'backbone'):
                backbone = model.net.backbone
            elif hasattr(model, 'net'):
                backbone = model.net
            elif hasattr(model, 'backbone'):
                backbone = model.backbone

            if backbone is not None:
                backbone_type = type(backbone).__name__.lower()
                return 'vit' in backbone_type or 'vision' in backbone_type or 'transformer' in backbone_type

            return False
        except Exception as e:
            self.logger.debug(f"Error checking model type: {e}")
            return False


def create_einstellung_evaluator(args) -> Optional[EinstellungEvaluator]:
    """
    Factory function to create an Einstellung evaluator based on args.

    Args:
        args: Command line arguments

    Returns:
        EinstellungEvaluator instance or None if not applicable
    """
    # Check if we're using the Einstellung dataset
    if not hasattr(args, 'dataset') or 'einstellung' not in args.dataset:
        return None

    # Extract configuration from args
    adaptation_threshold = getattr(args, 'einstellung_adaptation_threshold', 0.8)

    # Conditionally enable attention extraction based on backbone type
    # Only enable for ViT models, disable for ResNet and other CNN architectures
    backbone_type = getattr(args, 'backbone', 'resnet18').lower()
    is_vit_backbone = ('vit' in backbone_type or
                      'vision' in backbone_type or
                      'transformer' in backbone_type)

    # Override extract_attention based on backbone compatibility
    user_requested_attention = getattr(args, 'einstellung_extract_attention', True)
    if user_requested_attention and not is_vit_backbone:
        # Log warning that attention extraction is not supported for this backbone
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"⚠️  Attention extraction requested but not supported for backbone '{backbone_type}'")
        logger.warning("   Attention analysis is only available for Vision Transformer (ViT) models")
        logger.warning("   Disabling attention extraction for this experiment")
        extract_attention = False
    else:
        extract_attention = user_requested_attention and is_vit_backbone

    return EinstellungEvaluator(
        args=args,
        dataset_name=args.dataset,
        adaptation_threshold=adaptation_threshold,
        extract_attention=extract_attention
    )
