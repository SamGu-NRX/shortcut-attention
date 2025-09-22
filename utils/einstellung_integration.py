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
from utils.training import train as original_train


# Global evaluator instance
_einstellung_evaluator: Optional[EinstellungEvaluator] = None


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

    logger.info("🧠 EINSTELLUNG EFFECT INTEGRATION ENABLED SUCCESSFULLY")
    logger.info("   - Evaluator: Active")
    logger.info("   - Training hooks: Patched")
    logger.info("   - Subset evaluation: Enabled")
    logger.info("   - Attention extraction: Configured")
    logger.info("=" * 60)

    return True


def _patch_training_functions():
    """Apply monkey patches to integrate with Mammoth's training pipeline."""
    logger = logging.getLogger(__name__)

    logger.info("Applying training function patches...")

    # Replace the main training function
    import utils.training
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

    # Call original training function with epoch hooks
    result = _train_with_einstellung_hooks(model, dataset, args)

    # Export final results if evaluator is active
    if _einstellung_evaluator is not None:
        try:
            output_path = getattr(args, 'output_dir', './einstellung_results')
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


def _train_with_einstellung_hooks(model, dataset, args):
    """
    Enhanced training function with comprehensive Einstellung evaluation hooks.
    Based on Mammoth's original training.py but with epoch-by-epoch evaluation.
    """
    global _einstellung_evaluator

    logger = logging.getLogger(__name__)
    logger.info("🔄 EINSTELLUNG-ENHANCED TRAINING STARTED")
    logger.info(f"   - Dataset: {dataset.NAME}")
    logger.info(f"   - Model: {type(model).__name__}")
    logger.info(f"   - Backbone: {getattr(args, 'backbone', 'unknown')}")
    logger.info(f"   - N_TASKS: {dataset.N_TASKS}")
    logger.info(f"   - Evaluator active: {_einstellung_evaluator is not None}")

    # Import required modules
    from utils.loggers import Logger, FakeLogger
    from utils.status import ProgressBar
    from models.utils.future_model import FutureModel
    from backbone import warn_once
    from datasets import get_dataset
    from tqdm import tqdm
    import time

    # Initialize random results
    random_results_class, random_results_task = [], []

    # Setup logging
    if not args.disable_log:
        logger_mammoth = Logger(args, dataset.SETTING, dataset.NAME, model.NAME)
        logger.info("✓ Mammoth logger initialized")
    else:
        logger_mammoth = FakeLogger()

    # Move model to device
    import torch
    model.net.to(model.device)
    torch.cuda.empty_cache()
    logger.info(f"✓ Model moved to device: {model.device}")

    # Initialize results storage
    results, results_mask_classes = [], []

    # Load checkpoint if specified
    if args.loadcheck is not None:
        model, past_res = mammoth_load_checkpoint(args, model)
        if not args.disable_log and past_res is not None:
            (results, results_mask_classes, csvdump) = past_res
            logger_mammoth.load(csvdump)
        logger.info('✓ Checkpoint loaded')

    # Setup task range
    start_task = 0 if args.start_from is None else args.start_from
    end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after
    logger.info(f"✓ Task range: {start_task} to {end_task}")

    # Check if we need eval_dataset for future evaluation
    if args.eval_future:
        assert isinstance(model, FutureModel), "Model must be an instance of FutureModel to evaluate on future tasks"
        eval_dataset = get_dataset(args)
        # disable logging for this loop
        import contextlib
        with contextlib.nullcontext():  # Simplified - remove disable_logging for now
            for _ in range(dataset.N_TASKS):
                eval_dataset.get_data_loaders()
                model.change_transform(eval_dataset)
                del eval_dataset.train_loader
        logger.info("✓ Future evaluation dataset prepared")
    else:
        eval_dataset = dataset

    logger.info("🚀 Starting task-by-task training...")

    # MAIN TRAINING LOOP
    for t in range(start_task, end_task):
        logger.info(f"\n📋 TASK {t} START")

        # Get data loaders for current task
        train_loader, test_loader = dataset.get_data_loaders()
        logger.info(f"   - Train samples: {len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 'unknown'}")
        logger.info(f"   - Test samples: {len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 'unknown'}")

        # Begin task
        model.meta_begin_task(dataset)
        if _einstellung_evaluator is not None:
            try:
                _einstellung_evaluator.meta_begin_task(model, dataset)
                logger.info("   ✓ Einstellung meta_begin_task hook called")
            except Exception as e:
                logger.error(f"   ❌ Einstellung meta_begin_task error: {e}")

        # Check if we can skip forward pass computation
        can_compute_fwd_beforetask = True
        try:
            if len(results) > 0:
                random_results_class.append(results[-1])
                random_results_task.append(results_mask_classes[-1])
        except:
            pass

        # Setup progress tracking for this task
        n_epochs = dataset.get_epochs()
        logger.info(f"   - Epochs per task: {n_epochs}")

        epoch_pbar = tqdm(range(n_epochs), desc=f'Task {t}')
        device = model.device

        logger.info(f"   🏋️ Starting epoch-by-epoch training for Task {t}...")

        for epoch in epoch_pbar:
            # EPOCH TRAINING
            epoch_start_time = time.time()
            logger.debug(f"     - Epoch {epoch} starting...")

            # Training epoch
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)

            # Single epoch training
            epoch_loss = 0.0
            batch_count = 0
            for i, data in enumerate(train_pbar):
                # Forward pass and backward pass (following Mammoth's pattern)
                if len(data) == 2:
                    inputs, labels = data
                    not_aug_inputs = inputs
                elif len(data) == 3:
                    inputs, labels, not_aug_inputs = data
                else:
                    raise ValueError(f"Unexpected data format: {len(data)} items")

                inputs, labels = inputs.to(device), labels.to(device)
                not_aug_inputs = not_aug_inputs.to(device)

                # Model observation step
                loss = model.meta_observe(
                    inputs=inputs,
                    labels=labels,
                    not_aug_inputs=not_aug_inputs,
                    epoch=epoch
                )

                epoch_loss += loss
                batch_count += 1

                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss:.4f}'})

                # Log periodically for debugging
                if i % 100 == 0:
                    logger.debug(f"     - Batch {i}: loss = {loss:.4f}")

            train_pbar.close()
            avg_epoch_loss = epoch_loss / max(batch_count, 1)

            epoch_duration = time.time() - epoch_start_time
            logger.debug(f"     - Epoch {epoch} completed in {epoch_duration:.2f}s, avg_loss = {avg_epoch_loss:.4f}")

            # EINSTELLUNG EVALUATION HOOK AFTER EACH EPOCH
            if _einstellung_evaluator is not None:
                try:
                    eval_start_time = time.time()
                    logger.debug(f"     - Running Einstellung evaluation for epoch {epoch}...")
                    _einstellung_evaluator.after_training_epoch(model, dataset, epoch)
                    eval_duration = time.time() - eval_start_time
                    logger.debug(f"     - Einstellung evaluation completed in {eval_duration:.2f}s")
                except Exception as e:
                    logger.error(f"     ❌ Einstellung evaluation error: {e}")
                    logger.exception("Full traceback:")

            # Update epoch progress
            epoch_pbar.set_postfix({'avg_loss': f'{avg_epoch_loss:.4f}'})

        epoch_pbar.close()
        logger.info(f"   ✓ Task {t} training completed ({n_epochs} epochs)")

        # End of task
        model.meta_end_task(dataset)
        if _einstellung_evaluator is not None:
            try:
                _einstellung_evaluator.meta_end_task(model, dataset)
                logger.info("   ✓ Einstellung meta_end_task hook called")
            except Exception as e:
                logger.error(f"   ❌ Einstellung meta_end_task error: {e}")

        # Standard Mammoth evaluation
        logger.info(f"   🧪 Running standard Mammoth evaluation for Task {t}...")
        eval_start_time = time.time()
        accs = eval_dataset.evaluate(model, eval_dataset)
        eval_duration = time.time() - eval_start_time
        logger.info(f"   ✓ Standard evaluation completed in {eval_duration:.2f}s")

        dataset.log(args, logger_mammoth, accs, t, dataset.SETTING)

        # Store results
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        logger.info(f"📋 TASK {t} COMPLETE - Accuracy: {accs[0]:.2f}%")

    logger.info("🏁 ALL TASKS COMPLETED")
    logger.info(f"   - Final accuracies: {results}")
    logger.info("🔄 EINSTELLUNG-ENHANCED TRAINING FINISHED")

    return logger_mammoth


def _patched_meta_begin_task(self, dataset):
    """Patched meta_begin_task with Einstellung hook."""
    global _einstellung_evaluator

    # Call original method
    result = self._original_meta_begin_task(dataset)

    # Call Einstellung hook
    if _einstellung_evaluator is not None:
        try:
            _einstellung_evaluator.meta_begin_task(self, dataset)
        except Exception as e:
            logging.getLogger(__name__).debug(f"Einstellung meta_begin_task error: {e}")

    return result


def _patched_meta_end_task(self, dataset):
    """Patched meta_end_task with Einstellung hook."""
    global _einstellung_evaluator

    # Call original method
    result = self._original_meta_end_task(dataset)

    # Call Einstellung hook
    if _einstellung_evaluator is not None:
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
    global _einstellung_evaluator

    # Restore original functions
    import utils.training
    utils.training.train = original_train

    from models.utils.continual_model import ContinualModel
    if hasattr(ContinualModel, '_original_meta_begin_task'):
        ContinualModel.meta_begin_task = ContinualModel._original_meta_begin_task
        ContinualModel.meta_end_task = ContinualModel._original_meta_end_task

    # Clear evaluator
    _einstellung_evaluator = None

    logging.getLogger(__name__).info("Disabled Einstellung Effect integration")
