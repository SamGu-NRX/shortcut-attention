# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Einstellung Effect Dataset Implementation for CIFAR-100

This dataset tests cognitive rigidity in continual learning through artificial shortcuts.
Task 1: 8 superclasses (40 fine labels) learned normally
Task 2: 4 superclasses (20 fine labels) with magenta patches as shortcuts

The Einstellung Effect manifests as models overly relying on shortcuts from Task 2
when evaluating Task 1 data, even when shortcuts are removed or masked.
"""

from argparse import Namespace
from typing import Tuple, Dict, List, Optional
import numpy as np
import os
import pickle
import hashlib
import logging

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR100

from backbone.ResNetBlock import resnet18
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args


# CIFAR-100 superclass to fine class mapping (based on official CIFAR-100 structure)
SUPERCLASS_MAPPING = {
    'aquatic mammals': [4, 30, 55, 72, 95],      # beaver, dolphin, otter, seal, whale
    'fish': [1, 32, 67, 73, 91],                  # aquarium_fish, flatfish, ray, shark, trout
    'flowers': [54, 62, 70, 82, 92],             # orchid, poppy, rose, sunflower, tulip
    'food containers': [9, 10, 16, 28, 61],      # bottle, bowl, can, cup, plate
    'fruit and vegetables': [0, 51, 53, 57, 83], # apple, mushroom, orange, pear, sweet_pepper
    'household electrical device': [22, 39, 40, 86, 87], # clock, computer_keyboard, lamp, telephone, television
    'household furniture': [5, 20, 25, 84, 94],  # bed, chair, couch, table, wardrobe
    'insects': [6, 7, 14, 18, 24],               # bee, beetle, butterfly, caterpillar, cockroach
    'large carnivores': [3, 42, 43, 88, 97],     # bear, leopard, lion, tiger, wolf
    'large man-made outdoor things': [12, 17, 37, 68, 76], # bridge, castle, house, road, skyscraper
    'large natural outdoor scenes': [23, 33, 49, 60, 71],  # cloud, forest, mountain, plain, sea
    'large omnivores and herbivores': [15, 19, 21, 31, 38], # camel, cattle, chimpanzee, elephant, kangaroo
}

# Task split: 8 superclasses for T1, 4 for T2 (first 4 from remaining)
T1_SUPERCLASSES = list(SUPERCLASS_MAPPING.keys())[:8]
T2_SUPERCLASSES = list(SUPERCLASS_MAPPING.keys())[8:12]

# Extract fine labels for each task
T1_FINE_LABELS = []
for superclass in T1_SUPERCLASSES:
    T1_FINE_LABELS.extend(SUPERCLASS_MAPPING[superclass])

T2_FINE_LABELS = []
for superclass in T2_SUPERCLASSES:
    T2_FINE_LABELS.extend(SUPERCLASS_MAPPING[superclass])

# Shortcut configuration - apply to first superclass of T2
SHORTCUT_SUPERCLASS = T2_SUPERCLASSES[0]  # 'large man-made outdoor things'
SHORTCUT_FINE_LABELS = SUPERCLASS_MAPPING[SHORTCUT_SUPERCLASS]

# Preserve task boundaries: Task 1 classes first, then Task 2 classes.
# Using a sorted list would interleave labels from both tasks and break
# the contiguous ranges expected by Mammoth's task offsets.
ALL_USED_FINE_LABELS = T1_FINE_LABELS + T2_FINE_LABELS

# Create contiguous label mapping
CLASS_MAP_TO_CONTIGUOUS = {
    original_label: i for i, original_label in enumerate(ALL_USED_FINE_LABELS)
}


class TCIFAR100Einstellung(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100Einstellung, self).__init__(root, train, transform, target_transform,
                                                   download=not self._check_integrity())


class MyCIFAR100Einstellung(CIFAR100):
    """
    CIFAR100 dataset with Einstellung Effect modifications:
    - Supports magenta patch injection for shortcut learning
    - Supports patch masking for evaluation
    - Maintains compatibility with Mammoth's getitem structure
    - Includes caching functionality for performance optimization
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, apply_shortcut=False, mask_shortcut=False,
                 patch_size=4, patch_color=(255, 0, 255), enable_cache=True) -> None:

        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root

        # Einstellung Effect parameters
        self.apply_shortcut = apply_shortcut
        self.mask_shortcut = mask_shortcut
        self.patch_size = patch_size
        self.patch_color = np.array(patch_color, dtype=np.uint8)
        self.shortcut_labels = set(SHORTCUT_FINE_LABELS)

        # Caching parameters
        self.enable_cache = enable_cache
        self._cached_data = None
        self._cache_loaded = False

        super(MyCIFAR100Einstellung, self).__init__(root, train, transform, target_transform,
                                                    not self._check_integrity())

        # Filter to only keep used labels and remap
        self._filter_and_remap_labels()

        # Setup cache if enabled
        if self.enable_cache:
            self._setup_cache()

    def _filter_and_remap_labels(self):
        """Filter dataset to only include T1 and T2 classes, and remap labels to be contiguous."""
        used_indices = []
        new_targets = []
        new_data = []
        task_ids = []

        for i, target in enumerate(self.targets):
            if target in ALL_USED_FINE_LABELS:
                used_indices.append(i)
                contiguous_label = CLASS_MAP_TO_CONTIGUOUS[target]
                new_targets.append(contiguous_label)
                new_data.append(self.data[i])
                if target in T1_FINE_LABELS:
                    task_ids.append(0)
                elif target in T2_FINE_LABELS:
                    task_ids.append(1)
                else:
                    task_ids.append(0)

        self.data = np.array(new_data)
        self.targets = new_targets
        self.task_ids = np.array(task_ids)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset with Einstellung modifications.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target, not_aug_image) where target is the remapped class index
        """
        # Return cached data if available, otherwise fallback to original processing
        if self.enable_cache and self._cache_loaded and self._cached_data is not None:
            return self._get_cached_item(index)

        # Original processing (fallback)
        img, target = self.data[index], self.targets[index]

        # Convert to PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        # Apply Einstellung Effect modifications
        original_target_in_cifar100 = ALL_USED_FINE_LABELS[target]
        if original_target_in_cifar100 in self.shortcut_labels:
            img = self._apply_einstellung_effect(img, index)

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img

    def _apply_einstellung_effect(self, img: Image.Image, index: int) -> Image.Image:
        """
        Apply shortcut patches or masking for Einstellung Effect testing.

        Args:
            img: PIL Image to modify
            index: Image index for reproducible patch placement

        Returns:
            Modified PIL Image
        """
        if self.patch_size <= 0:
            return img

        # Convert to numpy for manipulation
        arr = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]

        if self.patch_size > min(h, w):
            return img  # Skip if patch is too large

        # Use fixed random state for reproducible patch placement
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

    def _get_cache_key(self) -> str:
        """
        Generate a secure hash key based on Einstellung parameters for cache identification.

        Returns:
            Secure hash string for cache key
        """
        # Create parameter string for hashing
        params = {
            'apply_shortcut': self.apply_shortcut,
            'mask_shortcut': self.mask_shortcut,
            'patch_size': self.patch_size,
            'patch_color': tuple(self.patch_color),
            'train': self.train
        }

        # Create deterministic string representation
        param_str = str(sorted(params.items()))

        # Generate secure hash
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _get_cache_path(self) -> str:
        """
        Get the cache file path using Mammoth's base_path structure.

        Returns:
            Full path to cache file
        """
        cache_key = self._get_cache_key()
        cache_dir = os.path.join(base_path(), 'CIFAR100', 'einstellung_cache')

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Create cache filename
        split_name = 'train' if self.train else 'test'
        cache_filename = f'{split_name}_{cache_key}.pkl'

        return os.path.join(cache_dir, cache_filename)

    def _setup_cache(self) -> None:
        """
        Setup cache by loading existing cache or building new one if needed.
        """
        try:
            cache_path = self._get_cache_path()

            if os.path.exists(cache_path):
                logging.info(f"Loading Einstellung cache from {cache_path}")
                self._load_cache(cache_path)
            else:
                logging.info(f"Building Einstellung cache at {cache_path}")
                self._build_cache(cache_path)

        except Exception as e:
            logging.warning(f"Cache setup failed: {e}. Falling back to original processing.")
            self._cache_loaded = False
            self._cached_data = None

    def _build_cache(self, cache_path: str) -> None:
        """
        Build cache by preprocessing all images using existing processing methods.

        Args:
            cache_path: Path where cache should be stored
        """
        try:
            logging.info("Preprocessing images for cache...")

            cached_images = []
            cached_targets = []
            cached_not_aug_images = []

            # Process all images using existing methods
            for i in range(len(self.data)):
                img, target = self.data[i], self.targets[i]

                # Convert to PIL Image
                img = Image.fromarray(img, mode='RGB')
                original_img = img.copy()

                # Apply Einstellung Effect modifications
                original_target_in_cifar100 = ALL_USED_FINE_LABELS[target]
                if original_target_in_cifar100 in self.shortcut_labels:
                    img = self._apply_einstellung_effect(img, i)

                # Store processed images as numpy arrays for caching
                cached_images.append(np.array(img))
                cached_targets.append(target)
                cached_not_aug_images.append(np.array(original_img))

                # Progress logging
                if (i + 1) % 1000 == 0:
                    logging.info(f"Processed {i + 1}/{len(self.data)} images")

            # Create cache data structure
            cache_data = {
                'processed_images': np.array(cached_images),
                'targets': np.array(cached_targets),
                'not_aug_images': np.array(cached_not_aug_images),
                'params_hash': self._get_cache_key()
            }

            # Save cache atomically
            temp_path = cache_path + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f)
            os.rename(temp_path, cache_path)

            # Load the cache we just built
            self._cached_data = cache_data
            self._cache_loaded = True

            logging.info(f"Cache built successfully with {len(cached_images)} images")

        except Exception as e:
            logging.error(f"Failed to build cache: {e}")
            # Clean up partial cache file
            if os.path.exists(cache_path + '.tmp'):
                os.remove(cache_path + '.tmp')
            self._cache_loaded = False
            self._cached_data = None

    def _load_cache(self, cache_path: str) -> None:
        """
        Load preprocessed images from cache file.

        Args:
            cache_path: Path to cache file
        """
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache parameters
            if cache_data.get('params_hash') != self._get_cache_key():
                logging.warning("Cache parameter mismatch. Rebuilding cache.")
                self._build_cache(cache_path)
                return

            # Basic integrity check
            expected_size = len(self.data)
            if len(cache_data['processed_images']) != expected_size:
                logging.warning(f"Cache size mismatch: expected {expected_size}, got {len(cache_data['processed_images'])}. Rebuilding cache.")
                self._build_cache(cache_path)
                return

            self._cached_data = cache_data
            self._cache_loaded = True
            logging.info(f"Cache loaded successfully with {len(cache_data['processed_images'])} images")

        except Exception as e:
            logging.error(f"Failed to load cache: {e}. Rebuilding cache.")
            self._build_cache(cache_path)

    def _get_cached_item(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Get cached data item in the same format as original __getitem__.

        Args:
            index: Index of the item to retrieve

        Returns:
            tuple: (image, target, not_aug_image) in the same format as original
        """
        # Get cached data
        img_array = self._cached_data['processed_images'][index]
        target = self._cached_data['targets'][index]
        not_aug_array = self._cached_data['not_aug_images'][index]

        # Convert back to PIL Images
        img = Image.fromarray(img_array, mode='RGB')
        not_aug_img = self.not_aug_transform(Image.fromarray(not_aug_array, mode='RGB'))

        # Apply transforms (same as original)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR100Einstellung(ContinualDataset):
    """
    Sequential CIFAR100 Einstellung Dataset for testing cognitive rigidity in continual learning.

    Implements the Einstellung Effect through:
    - Task 1: 8 superclasses (40 classes) learned normally
    - Task 2: 4 superclasses (20 classes) with artificial magenta shortcuts

    The Einstellung Effect manifests when models rely too heavily on T2 shortcuts
    when evaluating T1 data, demonstrating cognitive rigidity.
    """

    NAME = 'seq-cifar100-einstellung'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = [len(T1_FINE_LABELS), len(T2_FINE_LABELS)]  # [40, 20]
    N_CLASSES_PER_TASK_T1 = len(T1_FINE_LABELS)  # 40 classes
    N_CLASSES_PER_TASK_T2 = len(T2_FINE_LABELS)  # 20 classes
    N_TASKS = 2
    N_CLASSES = N_CLASSES_PER_TASK_T1 + N_CLASSES_PER_TASK_T2  # 60 total
    SIZE = (32, 32)
    MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        # Einstellung Effect configuration from args
        self.apply_shortcut = getattr(args, 'einstellung_apply_shortcut', False)
        self.mask_shortcut = getattr(args, 'einstellung_mask_shortcut', False)
        self.patch_size = getattr(args, 'einstellung_patch_size', 4)
        self.patch_color = getattr(args, 'einstellung_patch_color', [255, 0, 255])
        self.enable_cache = getattr(args, 'einstellung_enable_cache', True)

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            self.get_normalization_transform()
        ])

        train_dataset = MyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=True, download=True, transform=transform,
            apply_shortcut=self.apply_shortcut, mask_shortcut=self.mask_shortcut,
            patch_size=self.patch_size, patch_color=self.patch_color,
            enable_cache=self.enable_cache
        )

        test_dataset = MyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=self.apply_shortcut, mask_shortcut=self.mask_shortcut,
            patch_size=self.patch_size, patch_color=self.patch_color,
            enable_cache=self.enable_cache
        )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_evaluation_subsets(self) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Create evaluation subsets for comprehensive Einstellung metrics:
        - T1_all: All Task 1 data (normal evaluation)
        - T2_shortcut_normal: Task 2 data with shortcuts (normal)
        - T2_shortcut_masked: Task 2 data with shortcuts masked
        - T2_nonshortcut_normal: Non-shortcut classes in Task 2
        """
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            self.get_normalization_transform()
        ])

        subsets = {}

        # T1_all: All Task 1 data
        t1_dataset = MyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=False, patch_size=self.patch_size,
            enable_cache=self.enable_cache
        )
        # Filter to T1 classes only
        t1_indices = [i for i, target in enumerate(t1_dataset.targets)
                     if ALL_USED_FINE_LABELS[target] in T1_FINE_LABELS]
        subsets['T1_all'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t1_dataset, t1_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        # T2_shortcut_normal: Task 2 shortcut classes with shortcuts
        t2_shortcut_normal = MyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=True, mask_shortcut=False, patch_size=self.patch_size, patch_color=self.patch_color,
            enable_cache=self.enable_cache
        )
        t2_shortcut_indices = [i for i, target in enumerate(t2_shortcut_normal.targets)
                              if ALL_USED_FINE_LABELS[target] in SHORTCUT_FINE_LABELS]
        subsets['T2_shortcut_normal'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t2_shortcut_normal, t2_shortcut_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        # T2_shortcut_masked: Task 2 shortcut classes with shortcuts masked
        t2_shortcut_masked = MyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=True, patch_size=self.patch_size,
            enable_cache=self.enable_cache
        )
        subsets['T2_shortcut_masked'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t2_shortcut_masked, t2_shortcut_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        # T2_nonshortcut_normal: Non-shortcut classes in Task 2
        t2_nonshortcut_normal = MyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=False, patch_size=self.patch_size,
            enable_cache=self.enable_cache
        )
        t2_nonshortcut_indices = [i for i, target in enumerate(t2_nonshortcut_normal.targets)
                                 if (ALL_USED_FINE_LABELS[target] in T2_FINE_LABELS and
                                     ALL_USED_FINE_LABELS[target] not in SHORTCUT_FINE_LABELS)]
        subsets['T2_nonshortcut_normal'] = torch.utils.data.DataLoader(
            torch.utils.data.Subset(t2_nonshortcut_normal, t2_nonshortcut_indices),
            batch_size=self.args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        return subsets

    @staticmethod
    def get_transform():
        transform = transforms.Compose([
            transforms.ToPILImage(),
            SequentialCIFAR100Einstellung.TRANSFORM
        ])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR100Einstellung.MEAN,
                                        SequentialCIFAR100Einstellung.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR100Einstellung.MEAN,
                               SequentialCIFAR100Einstellung.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    @set_default_from_args('lr_scheduler')
    def get_scheduler_name(self):
        return 'multisteplr'

    @set_default_from_args('lr_milestones')
    def get_scheduler_name(self):
        return [35, 45]

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
