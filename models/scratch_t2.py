"""
This module contains the implementation of the Scratch_T2 baseline method.

The Scratch_T2 model trains only on Task 2 data, providing an optimal baseline
for measuring performance deficits in continual learning methods. It skips Task 1
entirely and trains normally during Task 2 epochs (like SGD).

This baseline is essential for Einstellung Effect analysis as it represents
the optimal performance achievable on Task 2 without any interference from Task 1.
"""

import torch
from models.utils.continual_model import ContinualModel
from tqdm import tqdm

from utils.conf import create_seeded_dataloader
from utils.schedulers import get_scheduler


class ScratchT2(ContinualModel):
    NAME = 'scratch_t2'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(ScratchT2, self).__init__(backbone, loss, args, transform, dataset=dataset)

    def begin_task(self, dataset):
        """
        Prepare for the current task.
        Skip Task 1 entirely, train normally on Task 2.
        """
        if self.current_task == 1:  # Task 2 (0-indexed)
            print(f"🎯 ScratchT2: Starting Task 2 training (current_task={self.current_task})")
            # Restore original number of epochs for Task 2
            if hasattr(self, '_original_n_epochs'):
                self.args.n_epochs = self._original_n_epochs
                print(f"🔄 ScratchT2: Restored n_epochs to {self.args.n_epochs} for Task 2")
        else:
            print(f"🚫 ScratchT2: Skipping Task {self.current_task + 1} (current_task={self.current_task})")
            # Store original n_epochs and set to 1 for skipped tasks to make training faster
            if not hasattr(self, '_original_n_epochs'):
                self._original_n_epochs = self.args.n_epochs
            self.args.n_epochs = 1
            print(f"⚡ ScratchT2: Set n_epochs to 1 for skipped task (original: {self._original_n_epochs})")

    def end_task(self, dataset):
        """
        End of task processing. No special training needed since we train normally during epochs.
        """
        print(f"🔚 ScratchT2: End of task {self.current_task + 1} (current_task={self.current_task})")

    def observe(self, inputs, labels, not_aug_inputs, **kwargs):
        """
        Train normally on Task 2, skip Task 1 entirely.
        """
        if self.should_skip_current_task():
            # Skip Task 1 entirely
            return 0.0
        else:
            # Train normally on Task 2 like SGD
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels.long())
            loss.backward()
            self.opt.step()
            return loss.item()

    def should_skip_current_task(self):
        """
        Check if this model should skip the current task entirely.
        Used by integrations like Einstellung evaluator to respect task skipping.

        Returns:
            bool: True if the current task should be skipped entirely
        """
        # ScratchT2 only participates in Task 2 (current_task == 1)
        return self.current_task != 1
