# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sequential CIFAR-100 Einstellung Dataset (224x224) for ViT

This dataset implements the Einstellung Effect for Vision Transformers by:
- Resizing CIFAR-100 images to 224x224 for ViT compatibility
- Task 1: 8 superclasses (40 classes) learned normally
- Task 2: 4 superclasses (20 classes) with magenta patch shortcuts
- Multi-subset evaluation for comprehensive ERI metrics
"""

import logging
from argparse import Namespace
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import CIFAR100
from PIL import Image

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args

# CIFAR-100 superclass structure
CIFAR100_SUPERCLASS_TO_FINE = {
    0: [4, 30, 55, 72, 95],      # aquatic_mammals
    1: [1, 32, 67, 73, 91],      # fish
    2: [54, 62, 70, 82, 92],     # flowers
    3: [9, 10, 16, 28, 61],      # food_containers
    4: [0, 51, 53, 57, 83],      # fruit_and_vegetables
    5: [22, 39, 40, 86, 87],     # household_electrical_devices
    6: [5, 20, 25, 84, 94],      # household_furniture
    7: [6, 7, 14, 18, 24],       # insects
    8: [3, 42, 43, 88, 97],      # large_carnivores
    9: [12, 17, 37, 68, 76],     # large_man-made_outdoor_things
    10: [23, 33, 49, 60, 71],    # large_natural_outdoor_scenes
    11: [15, 19, 21, 31, 38],    # large_omnivores_and_herbivores
    12: [34, 63, 64, 66, 75],    # medium_mammals
    13: [26, 45, 77, 79, 99],    # non-insect_invertebrates
    14: [2, 11, 35, 46, 98],     # people
    15: [27, 29, 44, 78, 93],    # reptiles
    16: [36, 50, 65, 74, 80],    # small_mammals
    17: [47, 52, 56, 59, 96],    # trees
    18: [8, 13, 48, 58, 90],     # vehicles_1
    19: [41, 69, 81, 85, 89]     # vehicles_2
}

# Task 1: First 8 superclasses (40 fine classes)
T1_SUPERCLASSES = [0, 1, 2, 3, 4, 5, 6, 7]
T1_FINE_LABELS = []
for sc in T1_SUPERCLASSES:
    T1_FINE_LABELS.extend(CIFAR100_SUPERCLASS_TO_FINE[sc])

# Task 2: Last 4 superclasses (20 fine classes)
T2_SUPERCLASSES = [8, 9, 10, 11]
T2_FINE_LABELS = []
for sc in T2_SUPERCLASSES:
    T2_FINE_LABELS.extend(CIFAR100_SUPERCLASS_TO_FINE[sc])

# Shortcut classes: First 2 superclasses of T2 (10 fine classes)
SHORTCUT_SUPERCLASSES = [8, 9]  # large_carnivores, large_man-made_outdoor_things
SHORTCUT_FINE_LABELS = []
for sc in SHORTCUT_SUPERCLASSES:
    SHORTCUT_FINE_LABELS.extend(CIFAR100_SUPERCLASS_TO_FINE[sc])

# All used fine labels in order
ALL_USED_FINE_LABELS = T1_FINE_LABELS + T2_FINE_LABELS


class TCIFAR100Einstellung224(CIFAR100):
    """CIFAR100 dataset wrapper to avoid download messages."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super().__init__(root, train, transform, target_transform,
                        download=not self._check_integrity())


class MyCIFAR100Einstellung224(CIFAR100):
    """
    CIFAR100 dataset with Einstellung Effect patches and 224x224 resizing.

    Supports:
    - Dynamic patch injection based on configuration
    - Patch masking for evaluation
    - 224x224 resizing for ViT compatibility
    - Proper class filtering and remapping
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, apply_shortcut=False, mask_shortcut=False,
                 patch_size=16, patch_color=(255, 0, 255)) -> None:

        self.apply_shortcut = apply_shortcut
        self.mask_shortcut = mask_shortcut
        self.patch_size = patch_size
        self.patch_color = np.array(patch_color, dtype=np.uint8)

        # Store original transform and create base transform
        self.original_transform = transform
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

        super().__init__(root, train, None, target_transform,
                        download=not self._check_integrity())

        # Filter and remap labels after initialization
        self._filter_and_remap_labels()

    def _filter_and_remap_labels(self):
        """Filter data to only include used classes and remap labels."""
        # Create mapping from original CIFAR-100 labels to new labels
        label_mapping = {old_label: new_label for new_label, old_label
                        in enumerate(ALL_USED_FINE_LABELS)}

        # Filter data and targets
        filtered_data = []
        filtered_targets = []

        for i, target in enumerate(self.targets):
            if target in label_mapping:
                filtered_data.append(self.data[i])
                filtered_targets.append(label_mapping[target])

        self.data = np.array(filtered_data)
        self.targets = filtered_targets

        logging.info(f"Filtered dataset: {len(self.data)} samples from {len(ALL_USED_FINE_LABELS)} classes")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get item with Einstellung Effect processing.

        Args:
            index: Index of the sample

        Returns:
            Tuple of (transformed_image, target, not_augmented_image)
        """
        img, target = self.data[index], self.targets[index]

        # Convert to PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        # Apply Einstellung Effect before other transforms
        img = self._apply_einstellung_effect(img, index)

        # Apply augmentations to the processed image
        not_aug_img = self.not_aug_transform(img)

        if self.original_transform is not None:
            img = self.original_transform(img)
        else:
            img = self.not_aug_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, not_aug_img

    def _apply_einstellung_effect(self, img: Image.Image, index: int) -> Image.Image:
        """
        Apply Einstellung Effect (magenta patch injection or masking).

        Args:
            img: PIL Image to process
            index: Sample index for deterministic placement

        Returns:
            Processed PIL Image
        """
        # Only apply to shortcut classes during training/evaluation
        target = self.targets[index]
        original_label = ALL_USED_FINE_LABELS[target]

        if original_label not in SHORTCUT_FINE_LABELS:
            return img

        # Convert to numpy for processing
        arr = np.array(img)
        h, w = arr.shape[:2]

        # Skip if patch is too large for the image
        if self.patch_size > min(h, w):
            return img

        # Deterministic patch placement based on index
        rng = np.random.RandomState(index)
        x = rng.randint(0, w - self.patch_size + 1)
        y = rng.randint(0, h - self.patch_size + 1)

        if self.mask_shortcut:
            # Mask the shortcut area (set to black)
            arr[y:y+self.patch_size, x:x+self.patch_size] = 0
        elif self.apply_shortcut:
            # Apply magenta shortcut patch
            arr[y:y+self.patch_size, x:x+self.patch_size] = self.patch_color

        return Image.fromarray(arr)


class SequentialCIFAR100Einstellung224(ContinualDataset):
    """
    Sequential CIFAR-100 Einstellung Dataset (224x224) for ViT.

    Tests cognitive rigidity through:
    - Task 1: 8 superclasses (40 classes) learned normally
    - Task 2: 4 superclasses (20 classes) with magenta shortcuts
    - 224x224 image resolution for ViT compatibility
    - Multi-subset evaluation for comprehensive ERI metrics
    """

    NAME = 'seq-cifar100-einstellung-224'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 30  # Average for Mammoth compatibility (40+20)/2
    N_CLASSES_PER_TASK_T1 = len(T1_FINE_LABELS)  # 40 classes
    N_CLASSES_PER_TASK_T2 = len(T2_FINE_LABELS)  # 20 classes
    N_TASKS = 2
    N_CLASSES = N_CLASSES_PER_TASK_T1 + N_CLASSES_PER_TASK_T2  # 60 total
    SIZE = (224, 224)
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet normalization for ViT

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        # Einstellung Effect configuration from args
        self.apply_shortcut = getattr(args, 'einstellung_apply_shortcut', False)
        self.mask_shortcut = getattr(args, 'einstellung_mask_shortcut', False)
        self.patch_size = getattr(args, 'einstellung_patch_size', 16)  # Larger for 224x224
        self.patch_color = getattr(args, 'einstellung_patch_color', [255, 0, 255])

        logging.info(f"Initialized {self.NAME} with Einstellung parameters:")
        logging.info(f"  - Apply shortcut: {self.apply_shortcut}")
        logging.info(f"  - Mask shortcut: {self.mask_shortcut}")
        logging.info(f"  - Patch size: {self.patch_size}")
        logging.info(f"  - Patch color: {self.patch_color}")

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Get train and test data loaders."""
        transform = self.TRANSFORM
        test_transform = self.TEST_TRANSFORM

        train_dataset = MyCIFAR100Einstellung224(
            base_path() + 'CIFAR100', train=True, download=True, transform=transform,
            apply_shortcut=self.apply_shortcut, mask_shortcut=self.mask_shortcut,
            patch_size=self.patch_size, patch_color=self.patch_color
        )

        test_dataset = TCIFAR100Einstellung224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform
        )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_evaluation_subsets(self) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Create evaluation subsets for comprehensive Einstellung metrics.

        Returns:
            Dictionary mapping subset names to DataLoaders:
            - T1_all: All Task 1 data (normal evaluation)
            - T2_shortcut_normal: Task 2 shortcut classes with shortcuts
            - T2_shortcut_masked: Task 2 shortcut classes with shortcuts masked
            - T2_nonshortcut_normal: Task 2 non-shortcut classes
        """
        test_transform = self.TEST_TRANSFORM
        subsets = {}

        # T1_all: All Task 1 data
        t1_dataset = MyCIFAR100Einstellung224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=False, patch_size=self.patch_size
        )
        t1_indices = [i for i, target in enumerate(t1_dataset.targets)
                     if ALL_USED_FINE_LABELS[target] in T1_FINE_LABELS]
        subsets['T1_all'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t1_dataset, t1_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

        # T2_shortcut_normal: Task 2 shortcut classes with shortcuts
        t2_shortcut_normal = MyCIFAR100Einstellung224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=True, mask_shortcut=False,
            patch_size=self.patch_size, patch_color=self.patch_color
        )
        t2_shortcut_indices = [i for i, target in enumerate(t2_shortcut_normal.targets)
                              if ALL_USED_FINE_LABELS[target] in SHORTCUT_FINE_LABELS]
        subsets['T2_shortcut_normal'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t2_shortcut_normal, t2_shortcut_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

        # T2_shortcut_masked: Task 2 shortcut classes with shortcuts masked
        t2_shortcut_masked = MyCIFAR100Einstellung224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=True, patch_size=self.patch_size
        )
        subsets['T2_shortcut_masked'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t2_shortcut_masked, t2_shortcut_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

        # T2_nonshortcut_normal: Non-shortcut classes in Task 2
        t2_nonshortcut_normal = MyCIFAR100Einstellung224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=False, patch_size=self.patch_size
        )
        t2_nonshortcut_indices = [i for i, target in enumerate(t2_nonshortcut_normal.targets)
                                 if (ALL_USED_FINE_LABELS[target] in T2_FINE_LABELS and
                                     ALL_USED_FINE_LABELS[target] not in SHORTCUT_FINE_LABELS)]
        subsets['T2_nonshortcut_normal'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t2_nonshortcut_normal, t2_nonshortcut_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

        logging.info(f"Created evaluation subsets:")
        for name, loader in subsets.items():
            logging.info(f"  - {name}: {len(loader.dataset)} samples")

        return subsets

    @staticmethod
    def get_transform():
        transform = transforms.Compose([
            transforms.ToPILImage(),
            SequentialCIFAR100Einstellung224.TRANSFORM
        ])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(
            SequentialCIFAR100Einstellung224.MEAN,
            SequentialCIFAR100Einstellung224.STD
        )
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(
            SequentialCIFAR100Einstellung224.MEAN,
            SequentialCIFAR100Einstellung224.STD
        )
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 20

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 64  # Larger batch size for ViT

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        # Get original CIFAR-100 class names
        cifar100_classes = CIFAR100(base_path() + 'CIFAR100', train=True, download=True).classes

        # Extract names for our used classes
        used_class_names = [cifar100_classes[i] for i in ALL_USED_FINE_LABELS]

        classes = fix_class_names_order(used_class_names, self.args)
        self.class_names = classes
        return self.class_names
