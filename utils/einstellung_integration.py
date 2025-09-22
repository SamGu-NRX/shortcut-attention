# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Einstellung Effect Integration with Mammoth Training Pipeline

This module provides integration hooks to seamlessly integrate the Einstellung
evaluator with Mammoth's native training pipeline without modifying core files.

The integration works by:
1. Monkey-patching key training functions
2. Injecting evaluator hooks at appropriate points
3. Maintaining compatibility with existing Mammoth functionality
"""

import functools
import logging
from typing import Any, Callable, Optional

from utils.einstellung_evaluator import EinstellungEvaluator, create_einstellung_evaluator


# Global evaluator instance
_einstellung_evaluator: Optional[EinstellungEvaluator] = None

# Store original training function before patching
_original_train: Optional[Callable] = None

# Global epoch tracking for proper timeline data collection
_global_epoch_counter: int = 0
_current_task_id: int = 0
_last_evaluated_epoch: int = -1


def enable_einstellung_integration(args) -> bool:
    """
    Enable Einstellung integration if applicable.

    Args:
        args: Command line arguments

    Returns:
        True if integration was enabled, False otherwise
    """
    global _einstellung_evaluator

    logger = logging.getLogger(__name__)

    # COMPREHENSIVE DEBUGGING: Log all relevant information
    logger.info("=" * 60)
    logger.info("EINSTELLUNG INTEGRATION DEBUG")
    logger.info("=" * 60)

    # Check if args has dataset attribute
    if not hasattr(args, 'dataset'):
        logger.warning("❌ args.dataset attribute missing")
        logger.info("Available args attributes:")
        for attr in sorted(dir(args)):
            if not attr.startswith('_'):
                logger.info(f"  - {attr}: {getattr(args, attr, 'N/A')}")
        return False

    logger.info(f"✓ Dataset: {args.dataset}")

    # Check if dataset name contains 'einstellung'
    dataset_name = str(args.dataset).lower()
    contains_einstellung = 'einstellung' in dataset_name
    logger.info(f"✓ Dataset name check: '{dataset_name}' contains 'einstellung': {contains_einstellung}")

    if not contains_einstellung:
        logger.warning(f"❌ Einstellung integration NOT enabled - dataset '{args.dataset}' does not contain 'einstellung'")
        logger.info("Integration will only activate for datasets with 'einstellung' in the name")
        logger.info("Expected dataset names: 'seq-cifar100-einstellung', 'seq-cifar100-einstellung-224'")
        return False

    logger.info("✓ Dataset name check passed - attempting to create evaluator...")

    # Create evaluator
    try:
        _einstellung_evaluator = create_einstellung_evaluator(args)
        logger.info(f"✓ Evaluator creation: {_einstellung_evaluator is not None}")
    except Exception as e:
        logger.error(f"❌ Evaluator creation failed: {e}")
        logger.exception("Full traceback:")
        return False

    if _einstellung_evaluator is None:
        logger.warning("❌ Evaluator creation returned None")
        return False

    logger.info("✓ Evaluator created successfully")

    # Apply integration patches
    try:
        _patch_training_functions()
        logger.info("✓ Training function patches applied")
    except Exception as e:
        logger.error(f"❌ Failed to patch training functions: {e}")
        logger.exception("Full traceback:")
        return False

    # Reset global epoch counter for new experiment
    global _global_epoch_counter, _current_task_id, _last_evaluated_epoch
    _global_epoch_counter = 0
    _current_task_id = 0
    _last_evaluated_epoch = -1

    logger.info("🧠 EINSTELLUNG EFFECT INTEGRATION ENABLED SUCCESSFULLY")
    logger.info("   - Evaluator: Active")
    logger.info("   - Training hooks: Patched")
    logger.info("   - Subset evaluation: Enabled")
    logger.info("   - Attention extraction: Configured")
    logger.info("   - Global epoch tracking: Enabled")
    logger.info("=" * 60)

    return True


def _patch_training_functions():
    """Apply monkey patches to integrate with Mammoth's training pipeline."""
    global _original_train
    logger = logging.getLogger(__name__)

    logger.info("Applying training function patches...")

    # Store original training function before patching
    import utils.training
    if _original_train is None:
        _original_train = utils.training.train
        logger.info("✓ Stored original training function")

    # Replace the main training function
    utils.training.train = _patched_train
    logger.info("✓ Patched utils.training.train")

    # Patch model lifecycle methods
    from models.utils.continual_model import ContinualModel

    # Store originals if not already stored
    if not hasattr(ContinualModel, '_original_meta_begin_task'):
        ContinualModel._original_meta_begin_task = ContinualModel.meta_begin_task
        ContinualModel._original_meta_end_task = ContinualModel.meta_end_task
        logger.info("✓ Stored original meta_begin_task and meta_end_task")

    # Apply patches
    ContinualModel.meta_begin_task = _patched_meta_begin_task
    ContinualModel.meta_end_task = _patched_meta_end_task
    logger.info("✓ Patched ContinualModel.meta_begin_task and meta_end_task")


def _patched_train(model, dataset, args):
    """Patched version of the main train function with Einstellung integration."""
    global _einstellung_evaluator

    logger = logging.getLogger(__name__)
    logger.info("🚀 PATCHED TRAINING FUNCTION CALLED")
    logger.info(f"   - Model: {type(model).__name__}")
    logger.info(f"   - Dataset: {dataset.NAME if hasattr(dataset, 'NAME') else type(dataset).__name__}")
    logger.info(f"   - Evaluator active: {_einstellung_evaluator is not None}")

    # Call original training function with minimal modifications
    result = _call_original_train_with_hooks(model, dataset, args)

    # Export final results if evaluator is active
    if _einstellung_evaluator is not None:
        try:
            output_path = getattr(args, 'results_path', './einstellung_results')

            # Ensure output directory exists
            import os
            os.makedirs(output_path, exist_ok=True)

            results_file = f"{output_path}/einstellung_final_results.json"
            _einstellung_evaluator.export_results(results_file)
            logger.info(f"✓ Exported Einstellung results to: {results_file}")

            # ALSO export CSV in the expected location for visualization
            csv_file = f"{output_path}/eri_sc_metrics.csv"
            _einstellung_evaluator.export_csv_for_visualization(csv_file)
            logger.info(f"✓ Exported Einstellung CSV for visualization: {csv_file}")
        except Exception as e:
            logger.error(f"❌ Error exporting Einstellung results: {e}")
            logger.exception("Full traceback:")

    logger.info("🏁 PATCHED TRAINING FUNCTION COMPLETED")
    return result


def _call_original_train_with_hooks(model, dataset, args):
    """Call the original training function with Einstellung hooks added."""
    global _einstellung_evaluator, _original_train

    logger = logging.getLogger(__name__)
    logger.info("🔄 CALLING ORIGINAL TRAINING WITH EINSTELLUNG HOOKS")

    if _original_train is None:
        logger.error("❌ Original training function not stored! Cannot call original training.")
        raise RuntimeError("Original training function not available")

    # Store original meta_end_epoch method
    original_meta_end_epoch = model.meta_end_epoch

    def patched_meta_end_epoch(epoch, dataset_arg):
        global _global_epoch_counter, _current_task_id

        # Call original meta_end_epoch
        result = original_meta_end_epoch(epoch, dataset_arg)

        # Add Einstellung evaluation hook after the epoch
        if _einstellung_evaluator is not None:
            try:
                # Check if the model wants to skip the current task entirely
                should_skip_task = False
                if hasattr(model, 'should_skip_current_task'):
                    should_skip_task = model.should_skip_current_task()
                    if should_skip_task:
                        logger.info(f"⏭️  Model {type(model).__name__} is skipping task {_current_task_id}, skipping Einstellung evaluation")

                if not should_skip_task:
                    global _last_evaluated_epoch

                    # Use the actual local epoch number instead of global counter for evaluation
                    # This ensures we get proper epoch numbering (0, 1, 2, ...) instead of fractional values
                    actual_epoch = epoch

                    # Create a unique evaluation key to prevent duplicates
                    eval_key = f"{_current_task_id}_{actual_epoch}"

                    # Prevent duplicate evaluations for the same task/epoch combination
                    if not hasattr(_einstellung_evaluator, '_evaluated_epochs'):
                        _einstellung_evaluator._evaluated_epochs = set()

                    if eval_key not in _einstellung_evaluator._evaluated_epochs:
                        logger.info(f"🔍 Running Einstellung evaluation for task {_current_task_id}, epoch {actual_epoch} (global: {_global_epoch_counter})...")
                        _einstellung_evaluator.after_training_epoch(model, dataset, actual_epoch)
                        logger.info(f"✓ Einstellung evaluation completed for task {_current_task_id}, epoch {actual_epoch}")
                        _einstellung_evaluator._evaluated_epochs.add(eval_key)
                    else:
                        logger.info(f"⏭️  Skipping duplicate evaluation for task {_current_task_id}, epoch {actual_epoch}")

                # Increment global epoch counter for tracking (even for skipped tasks)
                _global_epoch_counter += 1
            except Exception as e:
                logger.error(f"❌ Einstellung evaluation error for task {_current_task_id}, epoch {epoch}: {e}")
                logger.exception("Full traceback:")

        return result

    # Temporarily replace the method
    model.meta_end_epoch = patched_meta_end_epoch

    try:
        # Call the ORIGINAL training function (not the patched version)
        logger.info("🔄 Calling stored original training function...")
        result = _original_train(model, dataset, args)
        logger.info("✓ Original training function completed successfully")
    finally:
        # Restore the original method
        model.meta_end_epoch = original_meta_end_epoch

    logger.info("🔄 ORIGINAL TRAINING WITH HOOKS COMPLETED")
    return result


# Removed _train_with_einstellung_hooks function - it was causing issues and is not needed
# The integration now works by patching the model's meta_end_epoch method in _call_original_train_with_hooks


def _patched_meta_begin_task(self, dataset):
    """Patched meta_begin_task with Einstellung hook."""
    global _einstellung_evaluator, _current_task_id

    # Call original method
    result = self._original_meta_begin_task(dataset)

    # Update current task ID for global epoch tracking
    new_task_id = dataset.i if hasattr(dataset, 'i') else 0

    # Log task transition
    if new_task_id != _current_task_id:
        logger.info(f"🔄 Task transition: {_current_task_id} → {new_task_id}")

    _current_task_id = new_task_id

    # Reset evaluated epochs tracking for new task
    if _einstellung_evaluator is not None:
        if not hasattr(_einstellung_evaluator, '_evaluated_epochs'):
            _einstellung_evaluator._evaluated_epochs = set()
        # Clear previous task's epoch tracking
        _einstellung_evaluator._evaluated_epochs.clear()

    logger = logging.getLogger(__name__)

    # Check if the model wants to skip the current task entirely
    should_skip_task = False
    if hasattr(self, 'should_skip_current_task'):
        should_skip_task = self.should_skip_current_task()
        if should_skip_task:
            logger.info(f"🎯 Starting task {_current_task_id} - Model {type(self).__name__} will skip this task")
        else:
            logger.info(f"🎯 Starting task {_current_task_id} - Einstellung evaluation enabled: {_einstellung_evaluator is not None}")
    else:
        logger.info(f"🎯 Starting task {_current_task_id} - Einstellung evaluation enabled: {_einstellung_evaluator is not None}")

    # Call Einstellung hook only if the model is not skipping this task
    if _einstellung_evaluator is not None and not should_skip_task:
        try:
            _einstellung_evaluator.meta_begin_task(self, dataset)
        except Exception as e:
            logger.debug(f"Einstellung meta_begin_task error: {e}")

    return result


def _patched_meta_end_task(self, dataset):
    """Patched meta_end_task with Einstellung hook."""
    global _einstellung_evaluator

    # Call original method
    result = self._original_meta_end_task(dataset)

    # Check if the model wants to skip the current task entirely
    should_skip_task = False
    if hasattr(self, 'should_skip_current_task'):
        should_skip_task = self.should_skip_current_task()

    # Call Einstellung hook only if the model is not skipping this task
    if _einstellung_evaluator is not None and not should_skip_task:
        try:
            _einstellung_evaluator.meta_end_task(self, dataset)
        except Exception as e:
            logging.getLogger(__name__).debug(f"Einstellung meta_end_task error: {e}")

    return result


def get_einstellung_evaluator() -> Optional[EinstellungEvaluator]:
    """Get the current Einstellung evaluator instance."""
    global _einstellung_evaluator
    return _einstellung_evaluator


def disable_einstellung_integration():
    """Disable Einstellung integration and restore original functions."""
    global _einstellung_evaluator, _original_train, _global_epoch_counter, _current_task_id, _last_evaluated_epoch

    # Restore original functions
    import utils.training
    if _original_train is not None:
        utils.training.train = _original_train
        _original_train = None

    from models.utils.continual_model import ContinualModel
    if hasattr(ContinualModel, '_original_meta_begin_task'):
        ContinualModel.meta_begin_task = ContinualModel._original_meta_begin_task
        ContinualModel.meta_end_task = ContinualModel._original_meta_end_task

    # Clear evaluator and reset counters
    _einstellung_evaluator = None
    _global_epoch_counter = 0
    _current_task_id = 0
    _last_evaluated_epoch = -1

    logging.getLogger(__name__).info("Disabled Einstellung Effect integration")
