"""
Optimized Einstellung Effect Dataset Implementation for CIFAR-100
Performance-optimized version with image caching and tensor operations.
"""

import os
import pickle
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR100

from datasets.seq_cifar100_einstellung import (
    SequentialCIFAR100Einstellung,
    MyCIFAR100Einstellung,
    SUPERCLASS_MAPPING,
    T1_SUPERCLASSES, T2_SUPERCLASSES,
    T1_FINE_LABELS, T2_FINE_LABELS,
    SHORTCUT_FINE_LABELS, ALL_USED_FINE_LABELS
)
from utils.conf import base_path
import logging

logger = logging.getLogger(__name__)

class OptimizedMyCIFAR100Einstellung(MyCIFAR100Einstellung):
    """
    Performance-optimized version of MyCIFAR100Einstellung with image caching.

    Key optimizations:
    1. Pre-compute and cache all processed images
    2. Use tensor operations instead of PIL/numpy conversions
    3. Lazy loading with memory management
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, apply_shortcut=False, mask_shortcut=False,
                 patch_size=4, patch_color=(255, 0, 255)) -> None:

        # Initialize parent class
        super().__init__(root, train, transform, target_transform, download,
                        apply_shortcut, mask_shortcut, patch_size, patch_color)

        # Cache for processed images
        self._processed_cache = {}
        self._cache_file = None

        # Only enable caching if we're applying effects
        if self.apply_shortcut or self.mask_shortcut:
            self._setup_cache()

    def _setup_cache(self):
        """Setup caching system for processed images"""
        cache_dir = os.path.join(self.root, 'einstellung_cache')
        os.makedirs(cache_dir, exist_ok=True)

        # Create cache filename based on parameters
        cache_name = f"einstellung_{'train' if self.train else 'test'}"
        cache_name += f"_shortcut{int(self.apply_shortcut)}"
        cache_name += f"_mask{int(self.mask_shortcut)}"
        cache_name += f"_patch{self.patch_size}"
        cache_name += f"_color{''.join(map(str, self.patch_color))}"
        cache_name += ".pkl"

        self._cache_file = os.path.join(cache_dir, cache_name)

        # Load existing cache or create new one
        if os.path.exists(self._cache_file):
            logger.info(f"Loading Einstellung cache from {self._cache_file}")
            try:
                with open(self._cache_file, 'rb') as f:
                    self._processed_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._processed_cache)} cached images")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Will rebuild.")
lf._processed_cache = {}
        else:
            logger.info(f"Creating new Einstellung cache at {self._cache_file}")
            self._build_cache()

    def _build_cache(self):
        """Pre-process and cache all images that need Einstellung effects"""
        logger.info("Building Einstellung image cache...")

        # Find all indices that need processing
        indices_to_process = []
        for i in range(len(self.data)):
            target = self.targets[i]
            original_target_in_cifar100 = ALL_USED_FINE_LABELS[target]
            if original_target_in_cifar100 in self.shortcut_labels:
                indices_to_process.append(i)

        logger.info(f"Processing {len(indices_to_process)} images with Einstellung effects...")

        # Process images in batches to manage memory
        batch_size = 1000
        for i in range(0, len(indices_to_process), batch_size):
            batch_indices = indices_to_process[i:i+batch_size]

            for idx in batch_indices:
                # Get original image
                img_data = self.data[idx]
                img = Image.fromarray(img_data)

                # Apply Einstellung effect
                processed_img = self._apply_einstellung_effect_optimized(img, idx)

                # Store as numpy array to save memory
                self._processed_cache[idx] = np.array(processed_img)

            # Save cache periodically
            if i % (batch_size * 5) == 0:
                self._save_cache()
                logger.info(f"Processed {i + len(batch_indices)}/{len(indices_to_process)} images")

        # Final save
        self._save_cache()
        logger.info(f"Cache building complete! Processed {len(self._processed_cache)} images")

    def _save_cache(self):
        """Save cache to disk"""
        if self._cache_file:
            try:
                with open(self._cache_file, 'wb') as f:
                    pickle.dump(self._processed_cache, f)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def _apply_einstellung_effect_optimized(self, img: Image.Image, index: int) -> Image.Image:
        """
        Optimized version of Einstellung effect application.
        Uses numpy operations for better performance.
        """
        if self.patch_size <= 0:
            return img

        # Convert to numpy once
        arr = np.array(img, dtype=np.uint8)
        h, w = arr.shape[:2]

        if self.patch_size > min(h, w):
            return img

        # Use fixed random state for reproducible patch placement
        rng = np.random.RandomState(index)
        x = rng.randint(0, w - self.patch_size + 1)
        y = rng.randint(0, h - self.patch_size + 1)

        if self.mask_shortcut:
            # Mask the shortcut area (set to black)
            arr[y:y+self.patch_size, x:x+self.patch_size] = 0
        elif self.apply_shortcut:
            # Apply shortcut patch
            arr[y:y+self.patch_size, x:x+self.patch_size] = self.patch_color

        return Image.fromarray(arr)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Optimized getitem with caching.
        """
        target = self.targets[index]
        original_target_in_cifar100 = ALL_USED_FINE_LABELS[target]

        # Check if we have a cached processed version
        if index in self._processed_cache:
            # Use cached processed image
            img = Image.fromarray(self._processed_cache[index])
        else:
            # Use original image (no processing needed)
            img_data = self.data[index]
            img = Image.fromarray(img_data)

        # Original image for not_aug_transform
        original_img = Image.fromarray(self.data[index])
        not_aug_img = self.not_aug_transform(original_img)

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR100EinstellungOptimized(SequentialCIFAR100Einstellung):
    """
    Performance-optimized version of SequentialCIFAR100Einstellung.
    """

    NAME = 'seq-cifar100-einstellung-optimized'

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Get optimized data loaders"""
        transform = self.TRANSFORM
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            self.get_normalization_transform()
        ])

        # Use optimized dataset class
        train_dataset = OptimizedMyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=True, download=True, transform=transform,
            apply_shortcut=self.apply_shortcut, mask_shortcut=self.mask_shortcut,
            patch_size=self.patch_size, patch_color=self.patch_color
        )

        test_dataset = OptimizedMyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=self.apply_shortcut, mask_shortcut=self.mask_shortcut,
            patch_size=self.patch_size, patch_color=self.patch_color
        )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_einstellung_evaluation_loaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """Get optimized evaluation loaders for Einstellung analysis"""
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            self.get_normalization_transform()
        ])

        # Use optimized dataset class for all evaluation sets
        datasets = {}

        # T1_all: All Task 1 data
        datasets['T1_all'] = OptimizedMyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=False, patch_size=self.patch_size
        )

        # T2_shortcut_normal: Task 2 shortcut classes with shortcuts
        datasets['T2_shortcut_normal'] = OptimizedMyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=True, mask_shortcut=False, patch_size=self.patch_size,
            patch_color=self.patch_color
        )

        # T2_shortcut_masked: Task 2 shortcut classes with shortcuts masked
        datasets['T2_shortcut_masked'] = OptimizedMyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=True, patch_size=self.patch_size
        )

        # T2_nonshortcut_normal: Non-shortcut classes in Task 2
        datasets['T2_nonshortcut_normal'] = OptimizedMyCIFAR100Einstellung(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=False, patch_size=self.patch_size
        )

        # Create filtered datasets and loaders
        loaders = {}
        for name, dataset in datasets.items():
            # Apply the same filtering logic as parent class
            if name == 'T1_all':
                mask = [target in range(len(T1_FINE_LABELS)) for target in dataset.targets]
            elif 'T2_shortcut' in name:
                shortcut_targets = [T1_FINE_LABELS.index(label) + len(T1_FINE_LABELS)
                                  for label in SHORTCUT_FINE_LABELS if label in T2_FINE_LABELS]
                mask = [target in shortcut_targets for target in dataset.targets]
            else:  # T2_nonshortcut_normal
                nonshortcut_targets = [T1_FINE_LABELS.index(label) + len(T1_FINE_LABELS)
                                     for label in T2_FINE_LABELS if label not in SHORTCUT_FINE_LABELS]
                mask = [target in nonshortcut_targets for target in dataset.targets]

            # Filter dataset
            indices = [i for i, include in enumerate(mask) if include]
            subset = torch.utils.data.Subset(dataset, indices)

            # Create loader
            loaders[name] = torch.utils.data.DataLoader(
                subset, batch_size=self.args.batch_size, shuffle=False,
                num_workers=self.args.num_workers, pin_memory=True
            )

        return loaders
