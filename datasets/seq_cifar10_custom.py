# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import logging
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10

from utils.conf import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from datasets.utils import set_default_from_args


class TCIFAR10Custom(CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR10Custom, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class MyCIFAR10Custom(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR10Custom, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
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
    """
    Custom Sequential CIFAR10 Dataset for shortcut feature investigation.
    
    Task 1: airplane (0), automobile (1) - potential shortcut: sky, road
    Task 2: bird (2), truck (9) - potential shortcut: sky, road/wheels
    
    CIFAR-10 class mapping:
    0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 
    5: dog, 6: frog, 7: horse, 8: ship, 9: truck
    
    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'seq-cifar10-custom'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 2
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS  # Total 4 classes
    SIZE = (32, 32)
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    
    # Custom class order: airplane, automobile, bird, truck
    CUSTOM_CLASS_ORDER = [0, 1, 2, 9]  # CIFAR-10 indices
    
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __init__(self, args, transform_type: str = 'weak'):
        super().__init__(args)

        assert transform_type in ['weak', 'strong'], "Transform type must be either 'weak' or 'strong'."

        if transform_type == 'strong':
            logging.info("Using strong augmentation for CIFAR10 Custom")
            self.TRANSFORM = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 transforms.ToTensor(),
                 transforms.Normalize(SequentialCIFAR10Custom.MEAN, SequentialCIFAR10Custom.STD)])

        # Set custom class order for our specific experiment
        if not hasattr(args, 'custom_class_order') or args.custom_class_order is None:
            args.custom_class_order = ','.join(map(str, self.CUSTOM_CLASS_ORDER))
            logging.info(f"Setting custom class order for shortcut investigation: {args.custom_class_order}")

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Class method that returns the train and test loaders."""
        transform = self.TRANSFORM

        train_dataset = MyCIFAR10Custom(base_path() + 'CIFAR10', train=True,
                                      download=True, transform=transform)
        test_dataset = TCIFAR10Custom(base_path() + 'CIFAR10', train=False,
                                    download=True, transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10Custom.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"  # Use Vision Transformer for attention visualization

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR10Custom.MEAN, SequentialCIFAR10Custom.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR10Custom.MEAN, SequentialCIFAR10Custom.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        
        # Original CIFAR-10 class names
        all_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Get only the classes we're using in our custom order
        classes = [all_classes[i] for i in self.CUSTOM_CLASS_ORDER]
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names
    
    def get_task_labels(self):
        """
        Returns the task labels for analysis.
        Task 1: airplane, automobile
        Task 2: bird, truck
        """
        return {
            0: ['airplane', 'automobile'],
            1: ['bird', 'truck']
        }
