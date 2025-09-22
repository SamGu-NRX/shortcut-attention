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
from models.utils.continual_model import ContinualModel
from tqdm import tqdm

from utils.conf import create_seeded_dataloader
from utils.schedulers import get_scheduler


class Interleaved(ContinualModel):
    NAME = 'interleaved'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Interleaved, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.task1_data = None
        self.combined_loader = None

    def begin_task(self, dataset):
        """
        Prepare for the current task.
        For Task 1: Store data for later mixing
        For Task 2: Create combined Task 1 + Task 2 loader
        """
        print(f"🎯 Interleaved: Starting Task {self.current_task + 1} (current_task={self.current_task})")

        if self.current_task == 0:  # Task 1
            # Store Task 1 data for later combination
            self.task1_data = dataset.train_loader
            print("📦 Interleaved: Stored Task 1 data for later mixing")

        elif self.current_task == 1:  # Task 2
            # Create combined Task 1 + Task 2 data loader
            if self.task1_data is not None:
                print("🔄 Interleaved: Creating combined Task 1 + Task 2 data loader")

                # Collect all data from both tasks
                all_inputs = []
                all_labels = []

                # Add Task 1 data
                for x, l, _ in self.task1_data:
                    all_inputs.append(x)
                    all_labels.append(l)

                # Add Task 2 data
                for x, l, _ in dataset.train_loader:
                    all_inputs.append(x)
                    all_labels.append(l)

                if len(all_inputs) > 0:
                    all_inputs = torch.cat(all_inputs)
                    all_labels = torch.cat(all_labels)

                    # Create combined dataset and loader
                    combined_dataset = torch.utils.data.TensorDataset(all_inputs, all_labels)
                    self.combined_loader = create_seeded_dataloader(
                        self.args, combined_dataset,
                        batch_size=self.args.batch_size,
                        shuffle=True
                    )
                    print(f"✅ Interleaved: Combined loader created with {len(all_inputs)} samples")
            else:
                print("⚠️ Interleaved: No Task 1 data found, using Task 2 only")
                self.combined_loader = dataset.train_loader

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
            # Task 1: Train normally on Task 1 data
            self.opt.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels.long())
            loss.backward()
            self.opt.step()
            return loss.item()

        elif self.current_task == 1:
            # Task 2: Train on combined Task 1 + Task 2 data
            if self.combined_loader is not None:
                # Get a batch from the combined loader
                try:
                    if not hasattr(self, '_combined_iter'):
                        self._combined_iter = iter(self.combined_loader)

                    combined_inputs, combined_labels = next(self._combined_iter)
                except StopIteration:
                    # Reset iterator when exhausted
                    self._combined_iter = iter(self.combined_loader)
                    combined_inputs, combined_labels = next(self._combined_iter)

                combined_inputs = combined_inputs.to(self.device)
                combined_labels = combined_labels.to(self.device)

                self.opt.zero_grad()
                outputs = self.net(combined_inputs)
                loss = self.loss(outputs, combined_labels.long())
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

    def should_skip_current_task(self):
        """
        Check if this model should skip the current task entirely.
        Used by integrations like Einstellung evaluator to respect task skipping.

        Returns:
            bool: True if the current task should be skipped entirely
        """
        # Interleaved model participates in all tasks
        return False
