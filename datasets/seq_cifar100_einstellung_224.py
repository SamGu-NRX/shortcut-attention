# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sequential CIFAR-100 Einstellung Dataset (224x224) for ViT

This module provides Einstellung Effect testing for ViT models using proper inheritance
from the working SequentialCIFAR100224 implementation.

Key Features:
- Proper inheritance from proven ViT implementation
- Task 1: 8 superclasses (40 classes)
- Task 2: 4 superclasses (20 classes) with shortcuts
- 224x224 resolution for ViT compatibility
- Magenta patch injection for cognitive rigidity testing
"""

import logging
import numpy as np
from argparse import Namespace
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import CIFAR100
from PIL import Image

from datasets.seq_cifar100_224 import SequentialCIFAR100224
from datasets.seq_cifar100 import MyCIFAR100, TCIFAR100
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import fix_class_names_order, store_masked_loaders
from utils.conf import base_path
from datasets.utils import set_default_from_args


# CIFAR-100 Superclass to Fine-class mapping
CIFAR100_SUPERCLASS_MAPPING = {
    0: [4, 30, 55, 72, 95],      # aquatic mammals
    1: [1, 32, 67, 73, 91],      # fish
    2: [54, 62, 70, 82, 92],     # flowers
    3: [9, 10, 16, 28, 61],      # food containers
    4: [0, 51, 53, 57, 83],      # fruit and vegetables
    5: [22, 39, 40, 86, 87],     # household electrical devices
    6: [5, 20, 25, 84, 94],      # household furniture
    7: [6, 7, 14, 18, 24],       # insects
    8: [3, 42, 43, 88, 97],      # large carnivores
    9: [12, 17, 37, 68, 76],     # large man-made outdoor things
    10: [23, 33, 49, 60, 71],    # large natural outdoor scenes
    11: [15, 19, 21, 31, 38],    # large omnivores and herbivores
    12: [34, 63, 64, 66, 75],    # medium-sized mammals
    13: [26, 45, 77, 79, 99],    # non-insect invertebrates
    14: [2, 11, 35, 46, 98],     # people
    15: [27, 29, 44, 78, 93],    # reptiles
    16: [36, 50, 65, 74, 80],    # small mammals
    17: [47, 52, 56, 59, 96],    # trees
    18: [8, 13, 48, 58, 90],     # vehicles 1
    19: [41, 69, 81, 85, 89]     # vehicles 2
}

# Einstellung Task Structure
T1_SUPERCLASSES = [0, 1, 2, 3, 4, 5, 6, 7]     # First 8 superclasses for Task 1
T2_SUPERCLASSES = [8, 9, 10, 11]               # Next 4 superclasses for Task 2
SHORTCUT_SUPERCLASSES = [8, 9]                 # First 2 T2 superclasses get shortcuts

# Generate fine-class lists
T1_FINE_LABELS = []
for sc in T1_SUPERCLASSES:
    T1_FINE_LABELS.extend(CIFAR100_SUPERCLASS_MAPPING[sc])

T2_FINE_LABELS = []
for sc in T2_SUPERCLASSES:
    T2_FINE_LABELS.extend(CIFAR100_SUPERCLASS_MAPPING[sc])

SHORTCUT_FINE_LABELS = []
for sc in SHORTCUT_SUPERCLASSES:
    SHORTCUT_FINE_LABELS.extend(CIFAR100_SUPERCLASS_MAPPING[sc])

# All labels used in the experiment
# Maintain contiguous Task 1 and Task 2 blocks. Sorting would mix
# classes across tasks, breaking task-aligned label offsets.
ALL_USED_FINE_LABELS = T1_FINE_LABELS + T2_FINE_LABELS

# Create mapping from original labels to contiguous labels (0-59)
CLASS_MAP_TO_CONTIGUOUS = {}
for i, original_label in enumerate(ALL_USED_FINE_LABELS):
    CLASS_MAP_TO_CONTIGUOUS[original_label] = i

# Reverse mapping for convenience
CONTIGUOUS_TO_ORIGINAL = {v: k for k, v in CLASS_MAP_TO_CONTIGUOUS.items()}


class EinstellungMixin:
    """
    Mixin class providing Einstellung Effect functionality.
    Can be added to any CIFAR-100 dataset class.
    """

    def init_einstellung(self, apply_shortcut=False, mask_shortcut=False,
                        patch_size=16, patch_color=(255, 0, 255)):
        """Initialize Einstellung parameters."""
        self.apply_shortcut = apply_shortcut
        self.mask_shortcut = mask_shortcut
        self.patch_size = patch_size
        self.patch_color = np.array(patch_color, dtype=np.uint8)

    def filter_einstellung_classes(self):
        """Filter dataset to only include Einstellung classes and remap labels."""
        # Create mapping from original CIFAR-100 labels to new labels
        label_mapping = {old_label: new_label for new_label, old_label
                        in enumerate(ALL_USED_FINE_LABELS)}

        # Filter data and targets
        filtered_data = []
        filtered_targets = []
        task_ids = []

        for i, target in enumerate(self.targets):
            if target in label_mapping:
                filtered_data.append(self.data[i])
                new_label = label_mapping[target]
                filtered_targets.append(new_label)
                if target in T1_FINE_LABELS:
                    task_ids.append(0)
                elif target in T2_FINE_LABELS:
                    task_ids.append(1)
                else:
                    task_ids.append(0)

        self.data = np.array(filtered_data)
        self.targets = filtered_targets
        self.task_ids = np.array(task_ids)

        logging.info(f"Filtered Einstellung dataset: {len(self.data)} samples from {len(ALL_USED_FINE_LABELS)} classes")

    def apply_einstellung_effect(self, img: Image.Image, index: int) -> Image.Image:
        """Apply Einstellung Effect (magenta patch injection or masking)."""
        if not hasattr(self, 'apply_shortcut'):
            return img

        # Only apply to shortcut classes
        target = self.targets[index]
        original_label = ALL_USED_FINE_LABELS[target]

        if original_label not in SHORTCUT_FINE_LABELS:
            return img

        # Convert to numpy for processing
        arr = np.array(img)
        h, w = arr.shape[:2]

        # Skip if patch is too large
        if self.patch_size > min(h, w):
            return img

        # Deterministic patch placement based on index
        rng = np.random.RandomState(index + 42)  # Add offset for determinism
        x = rng.randint(0, w - self.patch_size + 1)
        y = rng.randint(0, h - self.patch_size + 1)

        if self.mask_shortcut:
            # Mask the shortcut area (set to black)
            arr[y:y+self.patch_size, x:x+self.patch_size] = 0
        elif self.apply_shortcut:
            # Apply magenta shortcut patch
            arr[y:y+self.patch_size, x:x+self.patch_size] = self.patch_color

        return Image.fromarray(arr)


class TEinstellungCIFAR100_224(TCIFAR100, EinstellungMixin):
    """
    Test CIFAR-100 dataset with Einstellung Effect for 224x224 ViT.
    Inherits from proven TCIFAR100 implementation.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, apply_shortcut=False, mask_shortcut=False,
                 patch_size=16, patch_color=(255, 0, 255)):
        # CRITICAL: Set root before calling super().__init__()
        self.root = root

        # Initialize Einstellung parameters
        self.init_einstellung(apply_shortcut, mask_shortcut, patch_size, patch_color)

        # Call parent constructor with proper download logic
        super().__init__(root, train, transform, target_transform,
                        download=not self._check_integrity())

        # Filter to Einstellung classes after initialization
        self.filter_einstellung_classes()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Retrieve items with Einstellung shortcut processing for evaluation."""
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img, mode='RGB')
        img = self.apply_einstellung_effect(img, index)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MyEinstellungCIFAR100_224(MyCIFAR100, EinstellungMixin):
    """
    Training CIFAR-100 dataset with Einstellung Effect for 224x224 ViT.
    Inherits from proven MyCIFAR100 implementation.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, apply_shortcut=False, mask_shortcut=False,
                 patch_size=16, patch_color=(255, 0, 255)):
        # CRITICAL: Set root before calling super().__init__()
        self.root = root

        # Initialize Einstellung parameters
        self.init_einstellung(apply_shortcut, mask_shortcut, patch_size, patch_color)

        # Call parent constructor with proper download logic
        super().__init__(root, train, transform, target_transform,
                        download=not self._check_integrity())

        # Filter to Einstellung classes after initialization
        self.filter_einstellung_classes()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Get item with Einstellung Effect processing."""
        img, target = self.data[index], self.targets[index]

        # Convert to PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        # Apply Einstellung Effect before other transforms
        img = self.apply_einstellung_effect(img, index)

        # Apply transforms (following parent class pattern)
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, not_aug_img


class SequentialCIFAR100Einstellung224(SequentialCIFAR100224):
    """
    Sequential CIFAR-100 Einstellung Dataset (224x224) for ViT.

    Inherits from the proven SequentialCIFAR100224 implementation and adds:
    - Einstellung task structure (8+4 superclasses)
    - Magenta patch injection for shortcuts
    - Multi-subset evaluation capabilities
    - Cognitive rigidity testing framework
    """

    NAME = 'seq-cifar100-einstellung-224'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = [len(T1_FINE_LABELS), len(T2_FINE_LABELS)]  # [40, 20]
    N_CLASSES_PER_TASK_T1 = len(T1_FINE_LABELS)  # 40 classes
    N_CLASSES_PER_TASK_T2 = len(T2_FINE_LABELS)  # 20 classes
    N_TASKS = 2
    N_CLASSES = len(ALL_USED_FINE_LABELS)  # 60 total classes
    SIZE = (224, 224)

    def __init__(self, args: Namespace) -> None:
        # Initialize parent (this sets up all the ViT-specific parameters)
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
        """Get train and test data loaders using Einstellung datasets."""
        # Use parent class transforms (proven to work with ViT)
        transform = self.TRANSFORM
        test_transform = self.TEST_TRANSFORM

        # Create Einstellung datasets using the working base classes
        train_dataset = MyEinstellungCIFAR100_224(
            base_path() + 'CIFAR100', train=True, download=True, transform=transform,
            apply_shortcut=self.apply_shortcut, mask_shortcut=self.mask_shortcut,
            patch_size=self.patch_size, patch_color=self.patch_color
        )

        test_dataset = TEinstellungCIFAR100_224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=self.apply_shortcut, mask_shortcut=self.mask_shortcut,
            patch_size=self.patch_size, patch_color=self.patch_color
        )

        # Use parent's proven loader creation
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_evaluation_subsets(self) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Create evaluation subsets for comprehensive Einstellung metrics.

        Returns:
            Dictionary mapping subset names to DataLoaders
        """
        test_transform = self.TEST_TRANSFORM
        subsets = {}

        # T1_all: All Task 1 data
        t1_dataset = TEinstellungCIFAR100_224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=False, patch_size=self.patch_size,
            patch_color=self.patch_color
        )
        t1_indices = [i for i, target in enumerate(t1_dataset.targets)
                     if ALL_USED_FINE_LABELS[target] in T1_FINE_LABELS]
        subsets['T1_all'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t1_dataset, t1_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

        # T2_shortcut_normal: Task 2 shortcut classes with shortcuts
        t2_shortcut_normal = TEinstellungCIFAR100_224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=True, mask_shortcut=False, patch_size=self.patch_size,
            patch_color=self.patch_color
        )
        shortcut_indices = [i for i, target in enumerate(t2_shortcut_normal.targets)
                           if ALL_USED_FINE_LABELS[target] in SHORTCUT_FINE_LABELS]
        subsets['T2_shortcut_normal'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t2_shortcut_normal, shortcut_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

        # T2_shortcut_masked: Task 2 shortcut classes with shortcuts masked
        t2_shortcut_masked = TEinstellungCIFAR100_224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=True, patch_size=self.patch_size,
            patch_color=self.patch_color
        )
        subsets['T2_shortcut_masked'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t2_shortcut_masked, shortcut_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

        # T2_nonshortcut_normal: Task 2 non-shortcut classes
        nonshortcut_labels = [label for label in T2_FINE_LABELS
                             if label not in SHORTCUT_FINE_LABELS]
        nonshortcut_indices = [i for i, target in enumerate(t1_dataset.targets)
                              if ALL_USED_FINE_LABELS[target] in nonshortcut_labels]
        subsets['T2_nonshortcut_normal'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t1_dataset, nonshortcut_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

        return subsets

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        cifar100_classes = CIFAR100(base_path() + 'CIFAR100', train=True, download=True).classes
        used_class_names = [cifar100_classes[idx] for idx in ALL_USED_FINE_LABELS]
        classes = fix_class_names_order(used_class_names, self.args)
        self.class_names = classes
        return self.class_names

    # Inherit all the proven ViT-specific methods from parent
    # No need to override: get_transform, get_backbone, get_epochs, get_batch_size, etc.
