"""
This module contains the implementation of the Interleaved baseline method.

The Interleaved model trains on mixed Task 1 and Task 2 data simultaneously,
providing a baseline for measuring performance without task boundaries.
- Task 1: Trains normally on Task 1 data during epochs
- Task 2: Trains on combined Task 1 + Task 2 data during epochs

This baseline is essential for Einstellung Effect analysis as it represents
the performance achievable when training on mixed data without continual
learning constraints.
"""

import torch
from torch.utils.data import ConcatDataset
from models.utils.continual_model import ContinualModel

from utils.conf import create_seeded_dataloader


class Interleaved(ContinualModel):
    NAME = 'interleaved'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Interleaved, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.task1_loader = None
        self.task2_loader = None
        self.combined_loader = None
        self.all_data = []
        self._combined_iter = None
        self._base_task_epochs = getattr(self.args, 'n_epochs', None)

    def begin_task(self, dataset):
        """
        Prepare for the current task.
        For Task 1: Store data for later mixing
        For Task 2: Create combined Task 1 + Task 2 loader
        """
        print(f"🎯 Interleaved: Starting Task {self.current_task + 1} (current_task={self.current_task})")

        if self.current_task == 0:  # Task 1
            # Capture Task 1 loader for later joint training
            self.task1_loader = dataset.train_loader
            print("📦 Interleaved: Stored Task 1 loader for later mixing")

            # Minimise time spent on Task 1 by shrinking epochs and skipping training steps
            if self._base_task_epochs is None:
                self._base_task_epochs = getattr(self.args, 'n_epochs', 1)
            self.args.n_epochs = 1

        elif self.current_task == 1:  # Task 2
            self.task2_loader = dataset.train_loader

            loaders = []
            if self.task1_loader is not None and hasattr(self.task1_loader, 'dataset'):
                loaders.append(self.task1_loader.dataset)
            if self.task2_loader is not None and hasattr(self.task2_loader, 'dataset'):
                loaders.append(self.task2_loader.dataset)

            if loaders:
                print("🔄 Interleaved: Creating combined Task 1 + Task 2 dataset")
                combined_dataset = ConcatDataset(loaders) if len(loaders) > 1 else loaders[0]

                target_epochs = (self._base_task_epochs or getattr(self.args, 'n_epochs', 1)) * max(len(loaders), 1)
                self.args.n_epochs = target_epochs

                self.combined_loader = create_seeded_dataloader(
                    self.args,
                    combined_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True
                )
                self._combined_iter = None
                print("✅ Interleaved: Combined loader ready with joint training schedule")
            else:
                print("⚠️ Interleaved: Missing Task 1 loader, falling back to Task 2 data only")
                self.combined_loader = dataset.train_loader
                self._combined_iter = None

    def end_task(self, dataset):
        """
        End of task processing. No special training needed since we train normally during epochs.
        """
        print(f"🔚 Interleaved: End of task {self.current_task + 1} (current_task={self.current_task})")

    def observe(self, inputs, labels, not_aug_inputs, **kwargs):
        """
        Train on current task data (Task 1) or combined data (Task 2).
        """
        if self.current_task == 0:
            # Skip actual training during Task 1 to wait for joint mixing
            return 0.0

        elif self.current_task == 1:
            # Task 2: Train on combined Task 1 + Task 2 data
            if self.combined_loader is not None:
                # Get a batch from the combined loader
                try:
                    if self._combined_iter is None:
                        self._combined_iter = iter(self.combined_loader)

                    batch = next(self._combined_iter)
                except StopIteration:
                    # Reset iterator when exhausted
                    self._combined_iter = iter(self.combined_loader)
                    batch = next(self._combined_iter)

                combined_inputs, combined_labels = self._extract_inputs_and_labels(batch)

                combined_inputs = combined_inputs.to(self.device, non_blocking=True)
                combined_labels = combined_labels.to(self.device, dtype=torch.long, non_blocking=True)

                self.opt.zero_grad()
                outputs = self.net(combined_inputs)
                loss = self.loss(outputs, combined_labels)
                loss.backward()
                self.opt.step()
                return loss.item()
            else:
                # Fallback to current batch if no combined loader
                self.opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss(outputs, labels.long())
            loss.backward()
            self.opt.step()
            return loss.item()

        else:
            # For any other tasks, train normally
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels.long())
            loss.backward()
            self.opt.step()
            return loss.item()

    @staticmethod
    def _extract_inputs_and_labels(batch):
        """Support tuples with auxiliary fields when mixing loaders."""
        if isinstance(batch, dict):
            # Prefer explicit keys when available
            inputs = batch.get('inputs') or batch.get('x')
            labels = batch.get('labels') or batch.get('y')
            if inputs is None or labels is None:
                raise ValueError("Combined batch dictionary missing required keys 'inputs'/'labels'.")
            return inputs, labels

        if not isinstance(batch, (list, tuple)):
            raise ValueError(f"Unexpected batch type from combined loader: {type(batch)!r}")

        if len(batch) < 2:
            raise ValueError("Combined loader must return at least (inputs, labels).")

        return batch[0], batch[1]

    def should_skip_current_task(self):
        """
        Check if this model should skip the current task entirely.
        Used by integrations like Einstellung evaluator to respect task skipping.

        Returns:
            bool: True if the current task should be skipped entirely
        """
        # Interleaved model participates in all tasks
        return False
