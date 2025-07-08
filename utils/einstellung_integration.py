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

    # Check if we should enable Einstellung evaluation
    if not hasattr(args, 'dataset') or 'einstellung' not in args.dataset:
        return False

    # Create evaluator
    _einstellung_evaluator = create_einstellung_evaluator(args)

    if _einstellung_evaluator is None:
        return False

    # Apply integration patches
    _patch_training_functions()

    logging.getLogger(__name__).info("Enabled Einstellung Effect integration")
    return True


def _patch_training_functions():
    """Apply monkey patches to integrate Einstellung evaluation."""

    # Patch the main training function
    import utils.training
    utils.training.train = _patched_train

    # Patch the ContinualModel class methods
    from models.utils.continual_model import ContinualModel

    # Store original methods
    if not hasattr(ContinualModel, '_original_meta_begin_task'):
        ContinualModel._original_meta_begin_task = ContinualModel.meta_begin_task
        ContinualModel._original_meta_end_task = ContinualModel.meta_end_task

    # Replace with patched versions
    ContinualModel.meta_begin_task = _patched_meta_begin_task
    ContinualModel.meta_end_task = _patched_meta_end_task


def _patched_train(model, dataset, args):
    """Patched version of the main train function with Einstellung integration."""
    global _einstellung_evaluator

    # Call original training function with epoch hooks
    result = _train_with_einstellung_hooks(model, dataset, args)

    # Export final results if evaluator is active
    if _einstellung_evaluator is not None:
        try:
            output_path = getattr(args, 'output_dir', './einstellung_results')
            results_file = f"{output_path}/einstellung_final_results.json"
            _einstellung_evaluator.export_results(results_file)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error exporting Einstellung results: {e}")

    return result


def _train_with_einstellung_hooks(model, dataset, args):
    """Modified training loop that includes Einstellung evaluation hooks."""
    # Import here to avoid circular imports
    from utils.training import train_single_epoch
    from utils.evaluate import evaluate
    from tqdm import tqdm
    import torch

    global _einstellung_evaluator

    # Initialize logger and other setup (following original train function)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logger = dataset.get_loggers()

    # Training configuration
    device = args.device
    model.to(device)
    model.net.to(device)
    model.train()

    # Training loop
    for t in range(dataset.N_TASKS):
        model.meta_begin_task(dataset)

        # Get data loaders for current task
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()

        # Train for the specified number of epochs
        epoch_pbar = tqdm(range(model.args.n_epochs), desc=f'Task {t}')

        for epoch in epoch_pbar:
            # Training epoch
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)

            # Single epoch training
            epoch_loss = 0.0
            for i, data in enumerate(train_pbar):
                # Forward pass and backward pass (following Mammoth's pattern)
                if len(data) == 2:
                    inputs, labels = data
                    not_aug_inputs = inputs
                elif len(data) == 3:
                    inputs, labels, not_aug_inputs = data
                else:
                    raise ValueError("Unexpected data format")

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

                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss:.4f}'})

            train_pbar.close()

            # Einstellung evaluation hook after each epoch
            if _einstellung_evaluator is not None:
                try:
                    _einstellung_evaluator.after_training_epoch(model, dataset, epoch)
                except Exception as e:
                    logging.getLogger(__name__).debug(f"Einstellung evaluation error: {e}")

            # Update epoch progress
            epoch_pbar.set_postfix({'avg_loss': f'{epoch_loss/len(train_loader):.4f}'})

        epoch_pbar.close()

        # End of task
        model.meta_end_task(dataset)

        # Standard Mammoth evaluation
        accs = dataset.evaluate(model, dataset)
        dataset.log(args, logger, accs, t, dataset.SETTING)

    return logger


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
