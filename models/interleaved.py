"""
This module contains the implementation of the Interleaved baseline method.

The Interleaved model trains on mixed Task 1 and Task 2 data simultaneously,
providing a baseline for measuring performance without task boundaries.
It collects data from both tasks and trains on the combined dataset.

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
        self.all_data = []
        self.all_labels = []

    def begin_task(self, dataset):
        """
        Prepare for the current task.
        Interleaved participates in all tasks normally.
        """
        print(f"🎯 Interleaved: Starting Task {self.current_task + 1} (current_task={self.current_task})")

    def end_task(self, dataset):
        """
        Train on combined data after each task.
        This implements the Interleaved baseline: train on mixed data from all seen tasks.
        """
        # Collect data from the current task (following Joint model pattern)
        self.all_data.append(dataset.train_loader)

        # Train on all collected data so far
        if len(self.all_data) > 0:
            # Combine all data from all tasks seen so far
            all_inputs = []
            all_labels = []

            for task_loader in self.all_data:
                for x, l, _ in task_loader:
                    all_inputs.append(x)
                    all_labels.append(l)

            if len(all_inputs) == 0:
                return

            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)

            # Create dataset and dataloader for training
            combined_dataset = torch.utils.data.TensorDataset(all_inputs, all_labels)
            dataloader = create_seeded_dataloader(
                self.args, combined_dataset,
                batch_size=self.args.batch_size,
                shuffle=True
            )

            # Get scheduler for training
            scheduler = get_scheduler(self, self.args, reload_optim=True)

            # Train on combined data from all tasks
            with tqdm(total=self.args.n_epochs * len(dataloader)) as pbar:
                for e in range(self.args.n_epochs):
                    pbar.set_description(f"Interleaved - Epoch {e}", refresh=False)
                    for i, (inputs, labels) in enumerate(dataloader):
                        if self.args.debug_mode and i > self.get_debug_iters():
                            break
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.opt.zero_grad()
                        outputs = self.net(inputs)
                        loss = self.loss(outputs, labels.long())
                        loss.backward()
                        self.opt.step()
                        pbar.update(self.args.batch_size)
                        pbar.set_postfix({'loss': loss.item()}, refresh=False)

                if scheduler is not None:
                    scheduler.step()

    def observe(self, *args, **kwargs):
        """
        Skip training during task progression.
        All training happens in end_task with combined data.
        """
        return 0

    def should_skip_current_task(self):
        """
        Check if this model should skip the current task entirely.
        Used by integrations like Einstellung evaluator to respect task skipping.

        Returns:
            bool: True if the current task should be skipped entirely
        """
        # Interleaved model participates in all tasks
        return False
