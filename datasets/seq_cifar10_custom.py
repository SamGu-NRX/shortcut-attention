# datasets/seq_cifar10_custom.py
# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import logging
from typing import Tuple, List 

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10 # For getting all class names

from utils.conf import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders,
                                              fix_class_names_order) # Import fix_class_names_order
from datasets.utils import set_default_from_args


class TCIFAR10Custom(CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR10Custom, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class MyCIFAR10Custom(CIFAR10):
    """Overrides the CIFAR10 dataset to change the getitem function."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR10Custom, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        return img, target, not_aug_img


class SequentialCIFAR10Custom(ContinualDataset):
    NAME = 'seq-cifar10-custom'
    SETTING = 'class-il' # This should align with a valid validation_mode if used
    N_CLASSES_PER_TASK = 2
    N_TASKS = 2
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    # This is our desired final order of ORIGINAL CIFAR-10 indices
    CUSTOM_CLASS_ORDER_INDICES = [0, 1, 2, 9] 
    
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __init__(self, args: Namespace, transform_type: str = 'weak'):
        # Set defaults for args expected by ContinualDataset and its utilities
        if not hasattr(args, 'permute_classes'):
            args.permute_classes = False # We are using custom_class_order, not permuting all
        
        # If custom_class_order is not provided by the user, use this dataset's definition
        if not hasattr(args, 'custom_class_order') or args.custom_class_order is None:
            args.custom_class_order = ','.join(map(str, self.CUSTOM_CLASS_ORDER_INDICES))
            logging.info(
                f"Using default custom_class_order for {self.NAME}: {args.custom_class_order}"
            )
        
        if not hasattr(args, 'custom_task_order'): # Not used by this dataset, but good to have
            args.custom_task_order = None
        
        # Set validation_mode to a known value.
        # If args.validation (percentage) is None, this mode might not be heavily used by store_masked_loaders
        # but other parts of the framework might expect it.
        if not hasattr(args, 'validation_mode'):
            args.validation_mode = self.SETTING # e.g., 'class-il'
            logging.info(f"Setting default args.validation_mode to '{args.validation_mode}' for {self.NAME}")
        elif args.validation_mode not in ['current', 'task-il', 'class-il', 'all']:
            logging.warning(f"Provided args.validation_mode '{args.validation_mode}' is not standard. Defaulting to '{self.SETTING}'.")
            args.validation_mode = self.SETTING


        super().__init__(args) # ContinualDataset.__init__ uses these args

        # After super().__init__(), self.class_order should be populated if fix_class_names_order works as expected
        if hasattr(self, 'class_order') and self.class_order is not None:
            logging.info(f"{self.NAME} initialized. self.class_order: {self.class_order}")
            # Verify our custom order is reflected if no permutation was requested
            if not args.permute_classes and args.custom_class_order:
                expected_order = [int(x) for x in args.custom_class_order.split(',')[:self.N_CLASSES]]
                if self.class_order != expected_order:
                    logging.warning(f"Mismatch! self.class_order: {self.class_order}, expected from args: {expected_order}")
        else:
            # This warning should ideally not appear if fix_class_names_order is correctly used by superclass
            logging.error(f"{self.NAME} __init__: self.class_order is NOT SET or is None after super().__init__!")


        assert transform_type in ['weak', 'strong'], "Transform type must be either 'weak' or 'strong'."
        if transform_type == 'strong':
            logging.info("Using strong augmentation for CIFAR10 Custom")
            self.TRANSFORM = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 transforms.ToTensor(),
                 transforms.Normalize(SequentialCIFAR10Custom.MEAN, SequentialCIFAR10Custom.STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM
        train_dataset = MyCIFAR10Custom(base_path() + 'CIFAR10', train=True,
                                      download=True, transform=transform)
        test_dataset = TCIFAR10Custom(base_path() + 'CIFAR10', train=False,
                                    download=True, transform=self.TEST_TRANSFORM)
        # self.args should have a valid validation_mode now
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        return transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10Custom.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone(): 
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCIFAR10Custom.MEAN, SequentialCIFAR10Custom.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialCIFAR10Custom.MEAN, SequentialCIFAR10Custom.STD)

    @set_default_from_args('n_epochs')
    def get_epochs(self): 
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self): 
        return 32

    def get_class_names(self) -> List[str]:
        """
        Returns the class names for the current task setting.
        It uses fix_class_names_order to respect args.custom_class_order or args.permute_classes.
        """
        if self.class_names is not None:
            return self.class_names
        
        # Get all original CIFAR10 class names
        all_cifar_classes = CIFAR10(base_path() + 'CIFAR10', train=True, download=True).classes
        
        # Let fix_class_names_order handle the ordering based on self.args
        # This utility should correctly use args.custom_class_order if provided,
        # or permute if args.permute_classes is True.
        # It also truncates to self.N_CLASSES.
        ordered_and_selected_names = fix_class_names_order(all_cifar_classes, self.args)
        
        self.class_names = ordered_and_selected_names
        logging.info(f"Resolved class names for {self.NAME} using fix_class_names_order: {self.class_names}")
        if len(self.class_names) != self.N_CLASSES:
            logging.error(f"Mismatch in N_CLASSES ({self.N_CLASSES}) and resolved class names len ({len(self.class_names)})")
        return self.class_names
    
    def get_task_labels(self):
        """
        Returns the task labels for analysis, assuming N_CLASSES_PER_TASK.
        """
        class_names = self.get_class_names() 
        if len(class_names) < self.N_CLASSES:
             logging.error(f"get_task_labels: Not enough class names ({len(class_names)}) resolved for {self.N_CLASSES} classes. Task labels might be incorrect.")
             # Fallback to generic names if class_names isn't fully populated
             return {
                i: [f'task{i}_class{j}' for j in range(self.N_CLASSES_PER_TASK)]
                for i in range(self.N_TASKS)
            }
        
        task_labels = {}
        for task_id in range(self.N_TASKS):
            start_idx = task_id * self.N_CLASSES_PER_TASK
            end_idx = (task_id + 1) * self.N_CLASSES_PER_TASK
            task_labels[task_id] = class_names[start_idx:end_idx]
        
        logging.debug(f"Generated task labels for {self.NAME}: {task_labels}")
        return task_labels