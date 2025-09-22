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

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR100
import errno
import shutil
import random

from backbone.ResNetBlock import resnet18
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset, fix_class_names_order,
                                              store_masked_loaders)
from utils.conf import base_path
from datasets.utils import set_default_from_args


class RobustEinstellungCache:
    """
    Robust caching system that stores only raw data and applies all processing
    deterministically during retrieval to ensure cross-method consistency.
    """

    def __init__(self, dataset_name: str, train: bool, resolution: str = "32x32"):
        self.dataset_name = dataset_name
        self.train = train
        self.resolution = resolution
        self._cache_loaded = False
        self._raw_cache_data = None

        # Generate cache key based only on raw data parameters
        self._cache_key = self._generate_cache_key()
        self._cache_path = self._get_cache_path()

        self.logger = logging.getLogger(__name__)

    def _generate_cache_key(self) -> str:
        """Generate cache key based only on raw data parameters."""
        params = {
            'dataset_name': self.dataset_name,
            'train': self.train,
            'resolution': self.resolution
        }
        param_str = str(sorted(params.items()))
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _get_cache_path(self) -> str:
        """Get cache file path."""
        cache_dir = os.path.join(base_path(), 'CIFAR100', 'robust_einstellung_cache')
        os.makedirs(cache_dir, exist_ok=True)

        split_name = 'train' if self.train else 'test'
        cache_filename = f'{split_name}_{self.resolution}_{self._cache_key}.pkl'

        return os.path.join(cache_dir, cache_filename)

    def load_or_build_cache(self, raw_data: np.ndarray, raw_targets: list) -> bool:
        """Load existing cache or build new one from raw data."""
        try:
            if os.path.exists(self._cache_path):
                return self._load_cache()
            else:
                return self._build_cache(raw_data, raw_targets)
        except Exception as e:
            self.logger.error(f"Robust cache operation failed: {e}")
            return False

    def _load_cache(self) -> bool:
        """Load cache from disk."""
        try:
            with open(self._cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache structure
            required_keys = ['raw_images', 'raw_targets', 'cache_key', 'version']
            for key in required_keys:
                if key not in cache_data:
                    self.logger.warning(f"Cache missing key {key}, rebuilding...")
                    return False

            # Validate cache key matches
            if cache_data['cache_key'] != self._cache_key:
                self.logger.info("Cache key mismatch, rebuilding...")
                return False

            self._raw_cache_data = cache_data
            self._cache_loaded = True

            self.logger.info(f"Loaded robust cache with {len(cache_data['raw_images'])} samples")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load robust cache: {e}")
            return False

    def _build_cache(self, raw_data: np.ndarray, raw_targets: list) -> bool:
        """Build cache from raw data."""
        try:
            self.logger.info(f"Building robust cache with {len(raw_data)} samples...")

            # Store only raw data - no processing applied
            cache_data = {
                'raw_images': raw_data.copy(),
                'raw_targets': list(raw_targets),
                'cache_key': self._cache_key,
                'version': '2.0',
                'dataset_name': self.dataset_name,
                'train': self.train,
                'resolution': self.resolution,
                'sample_count': len(raw_data)
            }

            # Save cache atomically
            temp_path = self._cache_path + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            os.rename(temp_path, self._cache_path)

            self._raw_cache_data = cache_data
            self._cache_loaded = True

            self.logger.info(f"Built robust cache successfully with {len(raw_data)} samples")
            return True

        except Exception as e:
            self.logger.error(f"Failed to build robust cache: {e}")
            return False

    def get_processed_item(self, index: int, apply_shortcut: bool, mask_shortcut: bool,
                          patch_size: int, patch_color: np.ndarray, shortcut_labels: set,
                          transform=None, target_transform=None, not_aug_transform=None):
        """Get processed item with deterministic Einstellung effects and transforms."""
        if not self._cache_loaded or self._raw_cache_data is None:
            raise RuntimeError("Robust cache not loaded")

        raw_images = self._raw_cache_data['raw_images']
        raw_targets = self._raw_cache_data['raw_targets']

        if index >= len(raw_images):
            raise IndexError(f"Index {index} out of bounds for cache size {len(raw_images)}")

        raw_img_array = raw_images[index]
        target = raw_targets[index]

        # Convert to PIL Image
        img = Image.fromarray(raw_img_array, mode='RGB')
        original_img = img.copy()

        # Apply Einstellung effects deterministically
        if target in shortcut_labels:
            img = self._apply_deterministic_einstellung_effect(
                img, index, apply_shortcut, mask_shortcut, patch_size, patch_color
            )

        # Apply transforms deterministically
        processed_img = apply_deterministic_transform(transform, img, index)
        not_aug_img = apply_deterministic_transform(not_aug_transform, original_img, index)

        # Apply target transform
        if target_transform is not None:
            target = target_transform(target)

        return processed_img, target, not_aug_img

    def _apply_deterministic_einstellung_effect(self, img: Image.Image, index: int,
                                              apply_shortcut: bool, mask_shortcut: bool,
                                              patch_size: int, patch_color: np.ndarray) -> Image.Image:
        """Apply Einstellung effect deterministically based on index."""
        if patch_size <= 0:
            return img

        # Convert to numpy for manipulation
        arr = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]

        if patch_size > min(h, w):
            return img

        # Use deterministic random state based on index
        rng = np.random.RandomState(index + 42)  # Fixed seed offset
        x = rng.randint(0, w - patch_size + 1)
        y = rng.randint(0, h - patch_size + 1)

        if mask_shortcut:
            # Mask the shortcut area (set to black)
            arr[y:y+patch_size, x:x+patch_size] = 0
        elif apply_shortcut:
            # Apply magenta shortcut patch
            arr[y:y+patch_size, x:x+patch_size] = patch_color

        return Image.fromarray(arr)

    def is_loaded(self) -> bool:
        """Check if cache is loaded."""
        return self._cache_loaded and self._raw_cache_data is not None


def apply_deterministic_transform(transform, img, index):
    """
    Apply transform with deterministic seed based on index for consistency across method instances.

    This ensures that different continual learning methods get identical transformed images
    when using datasets, which is critical for fair comparative experiments.

    Args:
        transform: Transform function to apply
        img: PIL Image to transform
        index: Sample index used to generate deterministic seed

    Returns:
        Transformed image
    """
    if transform is None:
        return img

    # Save current random states
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        # Create deterministic seed based on index and transform hash
        # This ensures different transforms get different but deterministic seeds
        transform_hash = hash(str(transform)) % (2**16)
        deterministic_seed = (index + transform_hash + 12345) % (2**32)

        torch.manual_seed(deterministic_seed)
        np.random.seed(deterministic_seed)
        random.seed(deterministic_seed)

        # Apply transform with deterministic seed
        transformed_img = transform(img)

        return transformed_img

    finally:
        # Always restore original random states
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(py_state)


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

        # Initialize cache state
        self._cache_loaded = False
        self._cached_data = None
        self._cache_error_count = 0
        self._max_cache_errors = 3

        super(MyCIFAR100Einstellung, self).__init__(root, train, transform, target_transform,
                                                    not self._check_integrity())

        # Filter to only keep used labels and remap
        self._filter_and_remap_labels()

        # Setup cache if enabled (after data is loaded and filtered)
        if self.enable_cache:
            self._setup_cache_with_fallback()

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
        Uses caching when available for performance, falls back to on-the-fly processing.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target, not_aug_image) where target is the remapped class index
        """
        try:
            # Use cached data if available and caching is enabled
            if self.enable_cache and hasattr(self, '_cache_loaded') and self._cache_loaded:
                return self._get_cached_item(index)

            # Fallback to original on-the-fly processing
            img, target = self.data[index], self.targets[index]

            # Convert to PIL Image
            img = Image.fromarray(img, mode='RGB')
            original_img = img.copy()

            # Apply Einstellung Effect modifications deterministically
            original_target_in_cifar100 = ALL_USED_FINE_LABELS[target]
            if original_target_in_cifar100 in self.shortcut_labels:
                img = self._apply_einstellung_effect(img, index)

            # Apply not_aug_transform deterministically
            not_aug_img = apply_deterministic_transform(self.not_aug_transform, original_img, index)

            # Apply main transform deterministically for cross-method consistency
            img = apply_deterministic_transform(self.transform, img, index)

            if self.target_transform is not None:
                target = self.target_transform(target)

            if hasattr(self, 'logits'):
                return img, target, not_aug_img, self.logits[index]

            return img, target, not_aug_img

        except Exception as e:
            logging.error(f"Critical error in __getitem__ for index {index}: {e}")
            raise

    def _apply_einstellung_effect(self, img: Image.Image, index: int) -> Image.Image:
        """
        Apply shortcut patches or masking for Einstellung Effect testing.
        Uses deterministic patch placement for cross-method consistency.

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

        # Use deterministic random state for reproducible patch placement
        # Include dataset parameters in seed to ensure consistency across method instances
        patch_seed = hash((index, self.patch_size, tuple(self.patch_color), self.apply_shortcut, self.mask_shortcut)) % (2**32)
        rng = np.random.RandomState(patch_seed)
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

    def _setup_cache_with_fallback(self) -> None:
        """
        Setup cache with comprehensive error handling and automatic fallback.
        Implements robust error handling for cache corruption, parameter mismatches,
        and disk space issues as required by task 5.
        """
        if self._cache_error_count >= self._max_cache_errors:
            logging.warning(f"Cache disabled due to {self._cache_error_count} consecutive errors. Using original processing.")
            self.enable_cache = False
            return

        try:
            # Check available disk space before cache operations
            cache_path = self._get_cache_path()
            cache_dir = os.path.dirname(cache_path)

            if not self._check_disk_space(cache_dir):
                logging.error("Insufficient disk space for cache operations. Falling back to original processing.")
                self._disable_cache_with_fallback("Insufficient disk space")
                return

            if os.path.exists(cache_path):
                logging.info(f"Loading Einstellung cache from {cache_path}")
                self._load_cache_with_validation(cache_path)
            else:
                logging.info(f"Building Einstellung cache at {cache_path}")
                self._build_cache_with_validation(cache_path)

        except (OSError, IOError) as e:
            # Handle disk space, permission, and I/O errors
            if e.errno == errno.ENOSPC:
                logging.error(f"No space left on device for cache operations: {e}. Falling back to original processing.")
            elif e.errno == errno.EACCES:
                logging.error(f"Permission denied for cache operations: {e}. Falling back to original processing.")
            else:
                logging.error(f"I/O error during cache operations: {e}. Falling back to original processing.")
            self._disable_cache_with_fallback(f"I/O error: {e}")
        except MemoryError as e:
            logging.error(f"Insufficient memory for cache operations: {e}. Falling back to original processing.")
            self._disable_cache_with_fallback(f"Memory error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during cache setup: {e}. Falling back to original processing.")
            self._disable_cache_with_fallback(f"Unexpected error: {e}")

    def _disable_cache_with_fallback(self, reason: str) -> None:
        """
        Disable cache and ensure fallback to original implementation.
        Maintains identical behavior to original implementation as required.

        Args:
            reason: Reason for disabling cache (for logging)
        """
        self._cache_error_count += 1
        self._cache_loaded = False
        self._cached_data = None

        if self._cache_error_count >= self._max_cache_errors:
            logging.warning(f"Disabling cache permanently after {self._cache_error_count} errors. Reason: {reason}")
            self.enable_cache = False
        else:
            logging.warning(f"Cache error #{self._cache_error_count}: {reason}. Will retry on next initialization.")

    def _check_disk_space(self, path: str, required_mb: int = 1024) -> bool:
        """
        Check if sufficient disk space is available for cache operations.

        Args:
            path: Directory path to check
            required_mb: Required space in MB (default 1GB)

        Returns:
            True if sufficient space available, False otherwise
        """
        try:
            # Ensure directory exists for space check
            os.makedirs(path, exist_ok=True)

            # Get disk usage statistics
            statvfs = os.statvfs(path)
            available_bytes = statvfs.f_frsize * statvfs.f_bavail
            available_mb = available_bytes / (1024 * 1024)

            if available_mb < required_mb:
                logging.warning(f"Insufficient disk space: {available_mb:.1f}MB available, {required_mb}MB required")
                return False

            return True

        except (OSError, AttributeError) as e:
            # AttributeError for Windows systems without os.statvfs
            logging.warning(f"Could not check disk space: {e}. Proceeding with cache operations.")
            return True  # Assume sufficient space if we can't check

    def _build_cache_with_validation(self, cache_path: str) -> None:
        """
        Build cache with comprehensive error handling and validation.
        Implements robust cache building with automatic fallback on any error.

        Args:
            cache_path: Path where cache should be stored
        """
        temp_path = cache_path + '.tmp'

        try:
            # Check disk space before starting
            cache_dir = os.path.dirname(cache_path)
            if not self._check_disk_space(cache_dir, required_mb=2048):  # Require 2GB for building
                raise OSError(errno.ENOSPC, "Insufficient disk space for cache building")

            logging.info("Preprocessing images for cache...")

            cached_images = []
            cached_targets = []
            cached_not_aug_images = []

            # Process all images using existing methods with error handling
            for i in range(len(self.data)):
                try:
                    img, target = self.data[i], self.targets[i]

                    # Convert to PIL Image
                    img = Image.fromarray(img, mode='RGB')
                    original_img = img.copy()

                    # Apply Einstellung Effect modifications
                    original_target_in_cifar100 = ALL_USED_FINE_LABELS[target]
                    if original_target_in_cifar100 in self.shortcut_labels:
                        img = self._apply_einstellung_effect(img, i)

                    # Validate processed image
                    img_array = np.array(img)
                    if img_array.shape != (32, 32, 3):
                        raise ValueError(f"Invalid processed image shape: {img_array.shape}")

                    # Store processed images as numpy arrays for caching
                    cached_images.append(img_array)
                    cached_targets.append(target)
                    cached_not_aug_images.append(np.array(original_img))

                    # Progress logging and periodic disk space check
                    if (i + 1) % 1000 == 0:
                        logging.info(f"Processed {i + 1}/{len(self.data)} images")
                        # Check disk space periodically during processing
                        if not self._check_disk_space(cache_dir, required_mb=1024):
                            raise OSError(errno.ENOSPC, "Disk space exhausted during cache building")

                except Exception as e:
                    logging.error(f"Error processing image {i}: {e}")
                    raise  # Re-raise to trigger fallback

            # Validate processed data before saving
            if len(cached_images) != len(self.data):
                raise ValueError(f"Cache size mismatch: processed {len(cached_images)}, expected {len(self.data)}")

            # Create cache data structure
            cache_data = {
                'processed_images': np.array(cached_images),
                'targets': np.array(cached_targets),
                'not_aug_images': np.array(cached_not_aug_images),
                'params_hash': self._get_cache_key(),
                'version': '1.0',  # Cache format version for future compatibility
                'dataset_size': len(self.data)
            }

            # Validate cache data structure
            self._validate_cache_data(cache_data)

            # Save cache atomically with error handling
            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Verify the written file
                if not os.path.exists(temp_path):
                    raise IOError("Temporary cache file was not created")

                file_size = os.path.getsize(temp_path)
                if file_size < 1024:  # Less than 1KB is suspicious
                    raise IOError(f"Cache file too small: {file_size} bytes")

                # Atomic rename
                os.rename(temp_path, cache_path)

            except (OSError, IOError) as e:
                if e.errno == errno.ENOSPC:
                    logging.error("No space left on device while saving cache")
                elif e.errno == errno.EACCES:
                    logging.error("Permission denied while saving cache")
                else:
                    logging.error(f"I/O error while saving cache: {e}")
                raise

            # Load and validate the cache we just built
            self._cached_data = cache_data
            self._cache_loaded = True

            logging.info(f"Cache built successfully with {len(cached_images)} images")

        except (OSError, IOError) as e:
            logging.error(f"I/O error building cache: {e}")
            self._cleanup_failed_cache(temp_path, cache_path)
            self._disable_cache_with_fallback(f"I/O error during cache building: {e}")
        except MemoryError as e:
            logging.error(f"Memory error building cache: {e}")
            self._cleanup_failed_cache(temp_path, cache_path)
            self._disable_cache_with_fallback(f"Memory error during cache building: {e}")
        except Exception as e:
            logging.error(f"Unexpected error building cache: {e}")
            self._cleanup_failed_cache(temp_path, cache_path)
            self._disable_cache_with_fallback(f"Unexpected error during cache building: {e}")

    def _cleanup_failed_cache(self, temp_path: str, cache_path: str) -> None:
        """
        Clean up failed cache files safely.

        Args:
            temp_path: Temporary cache file path
            cache_path: Final cache file path
        """
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logging.info(f"Cleaned up temporary cache file: {temp_path}")
        except OSError as e:
            logging.warning(f"Could not clean up temporary cache file {temp_path}: {e}")

        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logging.info(f"Cleaned up corrupted cache file: {cache_path}")
        except OSError as e:
            logging.warning(f"Could not clean up corrupted cache file {cache_path}: {e}")

    def _validate_cache_data(self, cache_data: dict) -> None:
        """
        Validate cache data structure before saving.

        Args:
            cache_data: Cache data dictionary to validate

        Raises:
            ValueError: If cache data is invalid
        """
        required_keys = ['processed_images', 'targets', 'not_aug_images', 'params_hash']
        for key in required_keys:
            if key not in cache_data:
                raise ValueError(f"Cache data missing required key: {key}")

        processed_images = cache_data['processed_images']
        targets = cache_data['targets']
        not_aug_images = cache_data['not_aug_images']

        if not isinstance(processed_images, np.ndarray):
            raise ValueError(f"Invalid processed_images type: {type(processed_images)}")

        if processed_images.ndim != 4 or processed_images.shape[1:] != (32, 32, 3):
            raise ValueError(f"Invalid processed_images shape: {processed_images.shape}")

        if len(targets) != len(processed_images) or len(not_aug_images) != len(processed_images):
            raise ValueError("Cache data arrays have mismatched lengths")

    def _load_cache_with_validation(self, cache_path: str) -> None:
        """
        Load preprocessed images from cache file with comprehensive validation and error handling.
        Implements robust cache loading with automatic fallback on corruption or parameter mismatches.

        Args:
            cache_path: Path to cache file
        """
        try:
            # Check file exists and has reasonable size
            if not os.path.exists(cache_path):
                logging.warning(f"Cache file does not exist: {cache_path}")
                self._build_cache_with_validation(cache_path)
                return

            file_size = os.path.getsize(cache_path)
            if file_size < 1024:  # Less than 1KB is suspicious
                logging.warning(f"Cache file too small ({file_size} bytes), rebuilding cache.")
                os.remove(cache_path)  # Remove corrupted file
                self._build_cache_with_validation(cache_path)
                return

            # Load cache data with error handling
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
            except (pickle.PickleError, EOFError, UnicodeDecodeError) as e:
                logging.warning(f"Cache file corrupted (pickle error): {e}. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return
            except (OSError, IOError) as e:
                logging.error(f"I/O error reading cache file: {e}. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            # Validate cache structure
            if not isinstance(cache_data, dict):
                logging.warning(f"Invalid cache data type: {type(cache_data)}. Expected dict. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            required_keys = ['processed_images', 'targets', 'not_aug_images', 'params_hash']
            for key in required_keys:
                if key not in cache_data:
                    logging.warning(f"Cache missing required key '{key}'. Rebuilding cache.")
                    self._handle_corrupted_cache(cache_path)
                    return

            # Validate cache parameters using hash comparison (detect parameter mismatches)
            current_hash = self._get_cache_key()
            cached_hash = cache_data.get('params_hash')
            if cached_hash != current_hash:
                logging.info(f"Cache parameter mismatch: cached={cached_hash}, current={current_hash}. Rebuilding cache for new parameters.")
                os.remove(cache_path)  # Remove outdated cache
                self._build_cache_with_validation(cache_path)
                return

            # Comprehensive integrity checking
            expected_size = len(self.data)
            cached_images = cache_data['processed_images']
            cached_targets = cache_data['targets']
            cached_not_aug = cache_data['not_aug_images']

            # Check data consistency
            if len(cached_images) != expected_size:
                logging.warning(f"Cache size mismatch: expected {expected_size}, got {len(cached_images)}. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            if len(cached_targets) != expected_size or len(cached_not_aug) != expected_size:
                logging.warning(f"Cache data inconsistency: images={len(cached_images)}, targets={len(cached_targets)}, not_aug={len(cached_not_aug)}. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            # Validate data types and shapes
            if not isinstance(cached_images, np.ndarray) or cached_images.ndim != 4:
                logging.warning(f"Invalid cached images format: type={type(cached_images)}, shape={getattr(cached_images, 'shape', 'N/A')}. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            if cached_images.shape[1:] != (32, 32, 3):  # CIFAR-100 image shape
                logging.warning(f"Invalid cached image shape: {cached_images.shape}. Expected (N, 32, 32, 3). Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            # Validate data ranges and types
            if cached_images.dtype not in [np.uint8, np.float32, np.float64]:
                logging.warning(f"Invalid cached images dtype: {cached_images.dtype}. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            # Validate targets match expected range
            try:
                target_array = np.array(cached_targets)
                if target_array.min() < 0 or target_array.max() >= len(ALL_USED_FINE_LABELS):
                    logging.warning(f"Invalid target range: min={target_array.min()}, max={target_array.max()}. Expected 0-{len(ALL_USED_FINE_LABELS)-1}. Rebuilding cache.")
                    self._handle_corrupted_cache(cache_path)
                    return
            except (ValueError, TypeError) as e:
                logging.warning(f"Invalid target data: {e}. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            # Additional validation for cache version compatibility
            cache_version = cache_data.get('version', '0.0')
            if cache_version != '1.0':
                logging.info(f"Cache version mismatch: {cache_version} vs 1.0. Rebuilding cache for compatibility.")
                os.remove(cache_path)
                self._build_cache_with_validation(cache_path)
                return

            # All validations passed
            self._cached_data = cache_data
            self._cache_loaded = True
            logging.info(f"Cache loaded successfully with {len(cached_images)} images")

        except MemoryError as e:
            logging.error(f"Memory error loading cache: {e}. Falling back to original processing.")
            self._disable_cache_with_fallback(f"Memory error loading cache: {e}")
        except Exception as e:
            logging.error(f"Unexpected error loading cache: {e}. Rebuilding cache.")
            self._handle_corrupted_cache(cache_path)

    def _handle_corrupted_cache(self, cache_path: str) -> None:
        """
        Handle corrupted cache files by safely removing them and rebuilding.

        Args:
            cache_path: Path to corrupted cache file
        """
        try:
            if os.path.exists(cache_path):
                # Create backup of corrupted cache for debugging
                backup_path = cache_path + '.corrupted'
                if not os.path.exists(backup_path):
                    shutil.copy2(cache_path, backup_path)
                    logging.info(f"Backed up corrupted cache to {backup_path}")

                os.remove(cache_path)
                logging.info(f"Removed corrupted cache file: {cache_path}")
        except OSError as e:
            logging.warning(f"Could not remove corrupted cache file {cache_path}: {e}")

        # Attempt to rebuild cache
        try:
            self._build_cache_with_validation(cache_path)
        except Exception as e:
            logging.error(f"Failed to rebuild cache after corruption: {e}")
            self._disable_cache_with_fallback(f"Cache rebuild failed after corruption: {e}")

    def _get_cached_item(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Get cached data item with comprehensive error handling and automatic fallback.
        Ensures cached data maintains exact same format as original dataset.
        Implements robust item retrieval with fallback to original processing on any error.

        Args:
            index: Index of the item to retrieve

        Returns:
            tuple: (image, target, not_aug_image) in the same format as original
        """
        try:
            # Validate cache is loaded and available
            if not self._cache_loaded or self._cached_data is None:
                raise ValueError("Cache not loaded or unavailable")

            # Validate index bounds
            cache_size = len(self._cached_data['processed_images'])
            if index < 0 or index >= cache_size:
                raise IndexError(f"Index {index} out of bounds for cached data of size {cache_size}")

            # Get cached data with validation
            try:
                img_array = self._cached_data['processed_images'][index]
                target = self._cached_data['targets'][index]
                not_aug_array = self._cached_data['not_aug_images'][index]
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(f"Cache data access error: {e}")

            # Validate data integrity
            if not isinstance(img_array, np.ndarray) or img_array.shape != (32, 32, 3):
                raise ValueError(f"Invalid cached image shape: {img_array.shape}, expected (32, 32, 3)")

            if not isinstance(not_aug_array, np.ndarray) or not_aug_array.shape != (32, 32, 3):
                raise ValueError(f"Invalid cached not_aug image shape: {not_aug_array.shape}, expected (32, 32, 3)")

            # Validate target type and range
            try:
                target = int(target)  # Ensure Python int type
                if target < 0 or target >= len(ALL_USED_FINE_LABELS):
                    raise ValueError(f"Invalid target value: {target}, expected 0-{len(ALL_USED_FINE_LABELS)-1}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid target data: {e}")

            # Validate image data ranges
            if img_array.dtype == np.uint8:
                if img_array.min() < 0 or img_array.max() > 255:
                    raise ValueError(f"Invalid uint8 image range: [{img_array.min()}, {img_array.max()}]")
            elif img_array.dtype in [np.float32, np.float64]:
                if img_array.min() < 0 or img_array.max() > 1:
                    # Convert float to uint8 if needed
                    img_array = (img_array * 255).astype(np.uint8)

            # Convert back to PIL Images (same as original processing)
            try:
                img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
                not_aug_img = self.not_aug_transform(Image.fromarray(not_aug_array.astype(np.uint8), mode='RGB'))
            except Exception as e:
                raise ValueError(f"PIL Image conversion error: {e}")

            # Apply transforms with deterministic seed for consistency across method instances
            try:
                img = apply_deterministic_transform(self.transform, img, index)

                if self.target_transform is not None:
                    target = self.target_transform(target)
            except Exception as e:
                raise ValueError(f"Transform application error: {e}")

            # Handle logits if present (same as original)
            if hasattr(self, 'logits'):
                try:
                    logits = self.logits[index]
                    return img, target, not_aug_img, logits
                except (IndexError, AttributeError) as e:
                    logging.warning(f"Logits access error for index {index}: {e}")
                    return img, target, not_aug_img

            return img, target, not_aug_img

        except (IndexError, ValueError, TypeError) as e:
            logging.warning(f"Cache data error for item {index}: {e}. Falling back to original processing.")
            return self._fallback_to_original_getitem(index)
        except MemoryError as e:
            logging.error(f"Memory error retrieving cached item {index}: {e}. Disabling cache.")
            self._disable_cache_with_fallback(f"Memory error in cached item retrieval: {e}")
            return self._fallback_to_original_getitem(index)
        except Exception as e:
            logging.error(f"Unexpected error retrieving cached item {index}: {e}. Falling back to original processing.")
            return self._fallback_to_original_getitem(index)

    def _fallback_to_original_getitem(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Fallback to original __getitem__ processing when cache fails.
        Ensures identical behavior to original implementation.

        Args:
            index: Index of the item to retrieve

        Returns:
            tuple: (image, target, not_aug_image) using original processing
        """
        try:
            # Temporarily disable cache to prevent recursion
            original_cache_loaded = self._cache_loaded
            self._cache_loaded = False

            # Call original processing logic
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
                img = apply_deterministic_transform(self.transform, img, index)

            if self.target_transform is not None:
                target = self.target_transform(target)

            # Restore cache state
            self._cache_loaded = original_cache_loaded

            if hasattr(self, 'logits'):
                return img, target, not_aug_img, self.logits[index]

            return img, target, not_aug_img

        except Exception as e:
            logging.error(f"Fallback processing also failed for index {index}: {e}")
            # If even fallback fails, disable cache permanently and re-raise
            self._disable_cache_with_fallback(f"Fallback processing failed: {e}")
            raise

    def get_normalization_transform(self):
        """Get normalization transform for CIFAR-100 Einstellung dataset."""
        return transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    def _get_evaluation_subset_cache_key(self) -> str:
        """
        Generate cache key for evaluation subsets based on parameters.

        Returns:
            Secure hash string for evaluation subset cache key
        """
        # Include batch_size and other relevant parameters for evaluation subsets
        params = {
            'patch_size': self.patch_size,
            'patch_color': tuple(self.patch_color),
            'batch_size': getattr(self.args, 'batch_size', 32) if hasattr(self, 'args') else 32,
            'evaluation_subsets': True  # Distinguish from regular cache
        }

        param_str = str(sorted(params.items()))
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _get_evaluation_subset_cache_path(self) -> str:
        """
        Get cache file path for evaluation subsets.

        Returns:
            Full path to evaluation subset cache file
        """
        cache_key = self._get_evaluation_subset_cache_key()
        cache_dir = os.path.join(base_path(), 'CIFAR100', 'einstellung_cache')
        os.makedirs(cache_dir, exist_ok=True)

        cache_filename = f'eval_subsets_{cache_key}.pkl'
        return os.path.join(cache_dir, cache_filename)

    def _cache_evaluation_subsets(self, subsets: Dict[str, torch.utils.data.DataLoader]) -> None:
        """
        Cache evaluation subsets for future use.

        Args:
            subsets: Dictionary of evaluation subset DataLoaders to cache
        """
        try:
            cache_path = self._get_evaluation_subset_cache_path()
            temp_path = cache_path + '.tmp'

            # Extract data from DataLoaders for caching
            cached_subsets = {}

            for subset_name, dataloader in subsets.items():
                logging.info(f"Caching evaluation subset: {subset_name}")

                # Extract all data from the DataLoader
                subset_data = []
                subset_targets = []

                # Temporarily disable transforms to get raw data
                original_transform = dataloader.dataset.dataset.transform if hasattr(dataloader.dataset, 'dataset') else dataloader.dataset.transform
                if hasattr(dataloader.dataset, 'dataset'):
                    dataloader.dataset.dataset.transform = None
                else:
                    dataloader.dataset.transform = None

                try:
                    for batch_idx, batch in enumerate(dataloader):
                        if len(batch) == 3:  # (img, target, not_aug_img)
                            imgs, targets, _ = batch
                        elif len(batch) == 2:  # (img, target)
                            imgs, targets = batch
                        else:
                            logging.warning(f"Unexpected batch format in {subset_name}: {len(batch)} elements")
                            continue

                        # Convert tensors to numpy for storage
                        if isinstance(imgs, torch.Tensor):
                            imgs = imgs.numpy()
                        if isinstance(targets, torch.Tensor):
                            targets = targets.numpy()

                        subset_data.extend(imgs)
                        subset_targets.extend(targets)

                        # Progress logging for large subsets
                        if (batch_idx + 1) % 10 == 0:
                            logging.debug(f"Cached {batch_idx + 1} batches for {subset_name}")

                finally:
                    # Restore original transform
                    if hasattr(dataloader.dataset, 'dataset'):
                        dataloader.dataset.dataset.transform = original_transform
                    else:
                        dataloader.dataset.transform = original_transform

                cached_subsets[subset_name] = {
                    'data': np.array(subset_data),
                    'targets': np.array(subset_targets),
                    'batch_size': dataloader.batch_size,
                    'num_workers': dataloader.num_workers,
                    'pin_memory': getattr(dataloader, 'pin_memory', False)
                }

                logging.info(f"Cached {len(subset_data)} samples for {subset_name}")

            # Create cache data structure
            cache_data = {
                'subsets': cached_subsets,
                'params_hash': self._get_evaluation_subset_cache_key(),
                'version': '1.0',
                'creation_time': os.path.getmtime(cache_path) if os.path.exists(cache_path) else None
            }

            # Save cache atomically
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Verify and rename
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
                os.rename(temp_path, cache_path)
                logging.info(f"Evaluation subsets cached successfully at {cache_path}")
            else:
                raise IOError("Cache file verification failed")

        except Exception as e:
            logging.error(f"Failed to cache evaluation subsets: {e}")
            # Clean up failed cache
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            raise

    def _get_cached_evaluation_subsets(self) -> Optional[Dict[str, torch.utils.data.DataLoader]]:
        """
        Load cached evaluation subsets if available and valid.

        Returns:
            Dictionary of cached evaluation subset DataLoaders, or None if not available
        """
        try:
            cache_path = self._get_evaluation_subset_cache_path()

            if not os.path.exists(cache_path):
                return None

            # Check file size
            if os.path.getsize(cache_path) < 1024:
                logging.warning("Evaluation subset cache file too small, rebuilding")
                os.remove(cache_path)
                return None

            # Load cache data
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache structure
            if not isinstance(cache_data, dict) or 'subsets' not in cache_data:
                logging.warning("Invalid evaluation subset cache structure, rebuilding")
                os.remove(cache_path)
                return None

            # Validate parameters
            current_hash = self._get_evaluation_subset_cache_key()
            cached_hash = cache_data.get('params_hash')
            if cached_hash != current_hash:
                logging.info(f"Evaluation subset cache parameter mismatch: {cached_hash} vs {current_hash}, rebuilding")
                os.remove(cache_path)
                return None

            # Reconstruct DataLoaders from cached data
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                self.get_normalization_transform()
            ])

            subsets = {}
            cached_subsets = cache_data['subsets']

            for subset_name, subset_cache in cached_subsets.items():
                try:
                    # Create dataset from cached data
                    cached_dataset = CachedEvaluationDataset(
                        subset_cache['data'],
                        subset_cache['targets'],
                        transform=test_transform
                    )

                    # Create DataLoader with original parameters
                    subsets[subset_name] = torch.utils.data.DataLoader(
                        cached_dataset,
                        batch_size=subset_cache.get('batch_size', 32),
                        shuffle=False,
                        num_workers=subset_cache.get('num_workers', 4),
                        pin_memory=subset_cache.get('pin_memory', True)
                    )

                    logging.debug(f"Loaded cached evaluation subset: {subset_name} with {len(cached_dataset)} samples")

                except Exception as e:
                    logging.error(f"Failed to reconstruct DataLoader for {subset_name}: {e}")
                    return None

            # Validate that all expected subsets are present
            expected_subsets = {'T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal'}
            if not expected_subsets.issubset(set(subsets.keys())):
                logging.warning(f"Missing evaluation subsets in cache. Expected: {expected_subsets}, Got: {set(subsets.keys())}")
                return None

            return subsets

        except Exception as e:
            logging.error(f"Failed to load cached evaluation subsets: {e}")
            # Clean up corrupted cache
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except OSError:
                pass
            return None


class CachedEvaluationDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for cached evaluation subset data.
    Maintains compatibility with original dataset interface.
    """

    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        """
        Initialize cached evaluation dataset.

        Args:
            data: Cached image data as numpy array
            targets: Cached targets as numpy array
            transform: Transform to apply to images
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from cached evaluation dataset.

        Args:
            index: Index of item to retrieve

        Returns:
            Tuple of (transformed_image, target)
        """
        img = self.data[index]
        target = self.targets[index]

        # Convert numpy array to PIL Image for transform compatibility
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:


            img = apply_deterministic_transform(self.transform, img, index)

        return img, target


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

        Uses cached evaluation subsets when available for improved performance.
        """
        # Try to load cached evaluation subsets first
        if self.enable_cache:
            try:
                cached_subsets = self._get_cached_evaluation_subsets()
                if cached_subsets is not None:
                    logging.info("Using cached evaluation subsets")
                    return cached_subsets
            except Exception as e:
                logging.warning(f"Failed to load cached evaluation subsets: {e}. Creating fresh subsets.")

        # Create fresh evaluation subsets (original implementation)
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

        # Cache the evaluation subsets for future use
        if self.enable_cache:
            try:
                self._cache_evaluation_subsets(subsets)
                logging.info("Cached evaluation subsets for future use")
            except Exception as e:
                logging.warning(f"Failed to cache evaluation subsets: {e}")

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

    def _get_evaluation_subset_cache_key(self) -> str:
        """
        Generate cache key for evaluation subsets based on parameters.

        Returns:
            Secure hash string for evaluation subset cache key
        """
        # Include batch_size and other relevant parameters for evaluation subsets
        params = {
            'patch_size': self.patch_size,
            'patch_color': tuple(self.patch_color),
            'batch_size': getattr(self.args, 'batch_size', 32) if hasattr(self, 'args') else 32,
            'evaluation_subsets': True  # Distinguish from regular cache
        }

        param_str = str(sorted(params.items()))
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _get_evaluation_subset_cache_path(self) -> str:
        """
        Get cache file path for evaluation subsets.

        Returns:
            Full path to evaluation subset cache file
        """
        cache_key = self._get_evaluation_subset_cache_key()
        cache_dir = os.path.join(base_path(), 'CIFAR100', 'einstellung_cache')
        os.makedirs(cache_dir, exist_ok=True)

        cache_filename = f'eval_subsets_{cache_key}.pkl'
        return os.path.join(cache_dir, cache_filename)

    def _cache_evaluation_subsets(self, subsets: Dict[str, torch.utils.data.DataLoader]) -> None:
        """
        Cache evaluation subsets for future use.

        Args:
            subsets: Dictionary of evaluation subset DataLoaders to cache
        """
        try:
            cache_path = self._get_evaluation_subset_cache_path()
            temp_path = cache_path + '.tmp'

            # Extract data from DataLoaders for caching
            cached_subsets = {}

            for subset_name, dataloader in subsets.items():
                logging.info(f"Caching evaluation subset: {subset_name}")

                # Extract all data by directly accessing the underlying dataset
                subset_data = []
                subset_targets = []

                # Get the underlying dataset (handle Subset wrapper)
                dataset = dataloader.dataset
                if hasattr(dataset, 'dataset'):
                    # This is a Subset, get the underlying dataset and indices
                    underlying_dataset = dataset.dataset
                    indices = dataset.indices
                else:
                    underlying_dataset = dataset
                    indices = range(len(dataset))

                # Temporarily disable transforms to get raw data
                original_transform = underlying_dataset.transform
                underlying_dataset.transform = None

                try:
                    # Extract raw data without transforms
                    for idx in indices:
                        try:
                            item = underlying_dataset[idx]
                            if len(item) == 3:  # (img, target, not_aug_img)
                                img, target, _ = item
                            elif len(item) == 2:  # (img, target)
                                img, target = item
                            else:
                                logging.warning(f"Unexpected item format in {subset_name}: {len(item)} elements")
                                continue

                            # Convert PIL Image to numpy array
                            if isinstance(img, Image.Image):
                                img_array = np.array(img)
                            elif isinstance(img, torch.Tensor):
                                img_array = img.numpy()
                                if img_array.ndim == 3 and img_array.shape[0] == 3:  # CHW to HWC
                                    img_array = img_array.transpose(1, 2, 0)
                            else:
                                img_array = np.array(img)

                            subset_data.append(img_array)
                            subset_targets.append(target)

                        except Exception as e:
                            logging.warning(f"Error processing item {idx} in {subset_name}: {e}")
                            continue

                        # Progress logging for large subsets
                        if len(subset_data) % 1000 == 0:
                            logging.debug(f"Cached {len(subset_data)} items for {subset_name}")

                finally:
                    # Restore original transform
                    underlying_dataset.transform = original_transform

                if not subset_data:
                    raise ValueError(f"No data extracted for subset {subset_name}")

                cached_subsets[subset_name] = {
                    'data': np.array(subset_data),
                    'targets': np.array(subset_targets),
                    'batch_size': dataloader.batch_size,
                    'num_workers': dataloader.num_workers,
                    'pin_memory': getattr(dataloader, 'pin_memory', False)
                }

                logging.info(f"Cached {len(subset_data)} samples for {subset_name}")

            # Create cache data structure
            cache_data = {
                'subsets': cached_subsets,
                'params_hash': self._get_evaluation_subset_cache_key(),
                'version': '1.0',
                'creation_time': os.path.getmtime(cache_path) if os.path.exists(cache_path) else None
            }

            # Save cache atomically
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Verify and rename
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
                os.rename(temp_path, cache_path)
                logging.info(f"Evaluation subsets cached successfully at {cache_path}")
            else:
                raise IOError("Cache file verification failed")

        except Exception as e:
            logging.error(f"Failed to cache evaluation subsets: {e}")
            # Clean up failed cache
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            raise

    def _get_cached_evaluation_subsets(self) -> Optional[Dict[str, torch.utils.data.DataLoader]]:
        """
        Load cached evaluation subsets if available and valid.

        Returns:
            Dictionary of cached evaluation subset DataLoaders, or None if not available
        """
        try:
            cache_path = self._get_evaluation_subset_cache_path()

            if not os.path.exists(cache_path):
                return None

            # Check file size
            if os.path.getsize(cache_path) < 1024:
                logging.warning("Evaluation subset cache file too small, rebuilding")
                os.remove(cache_path)
                return None

            # Load cache data
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache structure
            if not isinstance(cache_data, dict) or 'subsets' not in cache_data:
                logging.warning("Invalid evaluation subset cache structure, rebuilding")
                os.remove(cache_path)
                return None

            # Validate parameters
            current_hash = self._get_evaluation_subset_cache_key()
            cached_hash = cache_data.get('params_hash')
            if cached_hash != current_hash:
                logging.info(f"Evaluation subset cache parameter mismatch: {cached_hash} vs {current_hash}, rebuilding")
                os.remove(cache_path)
                return None

            # Reconstruct DataLoaders from cached data
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                self.get_normalization_transform()
            ])

            subsets = {}
            cached_subsets = cache_data['subsets']

            for subset_name, subset_cache in cached_subsets.items():
                try:
                    # Create dataset from cached data
                    cached_dataset = CachedEvaluationDataset(
                        subset_cache['data'],
                        subset_cache['targets'],
                        transform=test_transform
                    )

                    # Create DataLoader with original parameters (single-threaded for cached data)
                    subsets[subset_name] = torch.utils.data.DataLoader(
                        cached_dataset,
                        batch_size=subset_cache.get('batch_size', 32),
                        shuffle=False,
                        num_workers=0,  # Use single-threaded for cached data
                        pin_memory=False  # Disable pin_memory for cached data to avoid issues
                    )

                    logging.debug(f"Loaded cached evaluation subset: {subset_name} with {len(cached_dataset)} samples")

                except Exception as e:
                    logging.error(f"Failed to reconstruct DataLoader for {subset_name}: {e}")
                    return None

            # Validate that all expected subsets are present
            expected_subsets = {'T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal'}
            if not expected_subsets.issubset(set(subsets.keys())):
                logging.warning(f"Missing evaluation subsets in cache. Expected: {expected_subsets}, Got: {set(subsets.keys())}")
                return None

            return subsets

        except Exception as e:
            logging.error(f"Failed to load cached evaluation subsets: {e}")
            # Clean up corrupted cache
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except OSError:
                pass
            return None


class CachedEvaluationDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for cached evaluation subset data.
    Maintains compatibility with original dataset interface.
    """

    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        """
        Initialize cached evaluation dataset.

        Args:
            data: Cached image data as numpy array
            targets: Cached targets as numpy array
            transform: Transform to apply to images
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get item from cached evaluation dataset.

        Args:
            index: Index of item to retrieve

        Returns:
            Tuple of (transformed_image, target)
        """
        img = self.data[index]
        target = self.targets[index]

        # Convert numpy array to PIL Image for transform compatibility
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:


            img = apply_deterministic_transform(self.transform, img, index)

        return img, target
