"""
This module contains the implementation of the Scratch_T2 baseline method.

The Scratch_T2 model trains only on Task 2 data, providing an optimal baseline
for measuring performance deficits in continual learning methods. It skips Task 1
entirely and trains from scratch on Task 2 data at the end of Task 2.

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
        self.task2_data = None
        self.task2_labels = None

    def begin_task(self, dataset):
        """
        Prepare for the current task.
        Skip Task 1 (task 0), only collect Task 2 (task 1) data.
        """
        if self.current_task == 1:  # Task 2 (0-indexed)
            # Store Task 2 data loader for later training
            self.task2_data = dataset.train_loader

    def end_task(self, dataset):
        """
        Train only when Task 2 ends.
        This implements the Scratch_T2 baseline: train from scratch on Task 2 data only.
        """
        # Only train at the end of Task 2
        if self.current_task == 1 and self.task2_data is not None:
            # Collect all Task 2 data
            all_inputs = []
            all_labels = []

            for x, l, _ in self.task2_data:
                all_inputs.append(x)
                all_labels.append(l)

            if len(all_inputs) == 0:
                return

            all_inputs = torch.cat(all_inputs)
            all_labels = torch.cat(all_labels)

            # Create dataset and dataloader for training
            task2_dataset = torch.utils.data.TensorDataset(all_inputs, all_labels)
            dataloader = create_seeded_dataloader(
                self.args, task2_dataset,
                batch_size=self.args.batch_size,
                shuffle=True
            )

            # Get scheduler for training
            scheduler = get_scheduler(self, self.args, reload_optim=True)

            # Train on Task 2 data only
            with tqdm(total=self.args.n_epochs * len(dataloader)) as pbar:
                for e in range(self.args.n_epochs):
                    pbar.set_description(f"Scratch_T2 - Epoch {e}", refresh=False)
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
        All training happens in end_task for Task 2 only.
        """
        return 0
