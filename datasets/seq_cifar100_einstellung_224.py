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
import os
import pickle
import hashlib
import errno
from argparse import Namespace
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
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
    Inherits from proven TCIFAR100 implementation with caching support.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, apply_shortcut=False, mask_shortcut=False,
                 patch_size=16, patch_color=(255, 0, 255), enable_cache=True):
        # CRITICAL: Set root before calling super().__init__()
        self.root = root

        # Initialize Einstellung parameters
        self.init_einstellung(apply_shortcut, mask_shortcut, patch_size, patch_color)

        # Caching parameters
        self.enable_cache = enable_cache
        self._cached_data = None
        self._cache_loaded = False
        self._cache_error_count = 0
        self._max_cache_errors = 3  # Maximum cache errors before disabling cache permanently

        # Call parent constructor with proper download logic
        super().__init__(root, train, transform, target_transform,
                        download=not self._check_integrity())

        # Filter to Einstellung classes after initialization
        self.filter_einstellung_classes()

        # Setup cache if enabled
        if self.enable_cache:
            self._setup_cache_with_fallback()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Retrieve items with Einstellung shortcut processing for evaluation."""
        # Return cached data if available, with comprehensive error handling
        if self.enable_cache and self._cache_loaded and self._cached_data is not None:
            try:
                return self._get_cached_item(index)
            except Exception as e:
                logging.warning(f"Cache retrieval failed for index {index}: {e}. Using original processing.")
                # Continue to original processing below

        # Original processing (fallback)
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img, mode='RGB')
        img = self.apply_einstellung_effect(img, index)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _get_cache_key(self) -> str:
        """Generate a secure hash key based on Einstellung parameters for cache identification."""
        params = {
            'apply_shortcut': self.apply_shortcut,
            'mask_shortcut': self.mask_shortcut,
            'patch_size': self.patch_size,
            'patch_color': tuple(self.patch_color),
            'train': self.train,
            'resolution': '224x224'  # Distinguish from 32x32 cache
        }
        param_str = str(sorted(params.items()))
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _get_cache_path(self) -> str:
        """Get the cache file path using Mammoth's base_path structure."""
        cache_key = self._get_cache_key()
        cache_dir = os.path.join(base_path(), 'CIFAR100', 'einstellung_cache_224')
        os.makedirs(cache_dir, exist_ok=True)
        split_name = 'train' if self.train else 'test'
        cache_filename = f'{split_name}_{cache_key}.pkl'
        return os.path.join(cache_dir, cache_filename)

    def _setup_cache_with_fallback(self) -> None:
        """Setup cache with comprehensive error handling and automatic fallback."""
        if self._cache_error_count >= self._max_cache_errors:
            logging.warning(f"Cache disabled due to {self._cache_error_count} consecutive errors. Using original processing.")
            self.enable_cache = False
            return

        try:
            cache_path = self._get_cache_path()
            cache_dir = os.path.dirname(cache_path)

            if not self._check_disk_space(cache_dir):
                logging.error("Insufficient disk space for cache operations. Falling back to original processing.")
                self._disable_cache_with_fallback("Insufficient disk space")
                return

            if os.path.exists(cache_path):
                logging.info(f"Loading Einstellung 224x224 cache from {cache_path}")
                self._load_cache_with_validation(cache_path)
            else:
                logging.info(f"Building Einstellung 224x224 cache at {cache_path}")
                self._build_cache_with_validation(cache_path)

        except (OSError, IOError) as e:
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
        """Disable cache and ensure fallback to original implementation."""
        self._cache_error_count += 1
        self._cache_loaded = False
        self._cached_data = None

        if self._cache_error_count >= self._max_cache_errors:
            logging.warning(f"Disabling cache permanently after {self._cache_error_count} errors. Reason: {reason}")
            self.enable_cache = False
        else:
            logging.warning(f"Cache error #{self._cache_error_count}: {reason}. Will retry on next initialization.")

    def _check_disk_space(self, path: str, required_mb: int = 2048) -> bool:
        """Check if sufficient disk space is available for cache operations (224x224 needs more space)."""
        try:
            os.makedirs(path, exist_ok=True)
            statvfs = os.statvfs(path)
            available_bytes = statvfs.f_frsize * statvfs.f_bavail
            available_mb = available_bytes / (1024 * 1024)

            if available_mb < required_mb:
                logging.warning(f"Insufficient disk space: {available_mb:.1f}MB available, {required_mb}MB required")
                return False
            return True
        except (OSError, AttributeError) as e:
            logging.warning(f"Could not check disk space: {e}. Proceeding with cache operations.")
            return True

    def _build_cache_with_validation(self, cache_path: str) -> None:
        """Build cache with comprehensive error handling and validation."""
        temp_path = cache_path + '.tmp'

        try:
            cache_dir = os.path.dirname(cache_path)
            if not self._check_disk_space(cache_dir, required_mb=4096):  # 224x224 needs more space
                raise OSError(errno.ENOSPC, "Insufficient disk space for cache building")

            logging.info("Preprocessing 224x224 images for cache...")

            cached_images = []
            cached_targets = []

            # Process all images using existing methods with error handling
            for i in range(len(self.data)):
                try:
                    img, target = self.data[i], self.targets[i]

                    # Convert to PIL Image
                    img = Image.fromarray(img, mode='RGB')

                    # Apply Einstellung Effect modifications
                    img = self.apply_einstellung_effect(img, i)

                    # Validate processed image (224x224 after resize)
                    img_array = np.array(img)
                    if img_array.shape != (32, 32, 3):  # Still CIFAR size before transform
                        raise ValueError(f"Invalid processed image shape: {img_array.shape}")

                    # Store processed images as numpy arrays for caching
                    cached_images.append(img_array)
                    cached_targets.append(target)

                    # Progress logging and periodic disk space check
                    if (i + 1) % 1000 == 0:
                        logging.info(f"Processed {i + 1}/{len(self.data)} images for 224x224 cache")
                        if not self._check_disk_space(cache_dir, required_mb=2048):
                            raise OSError(errno.ENOSPC, "Disk space exhausted during cache building")

                except Exception as e:
                    logging.error(f"Error processing image {i}: {e}")
                    raise

            # Validate processed data before saving
            if len(cached_images) != len(self.data):
                raise ValueError(f"Cache size mismatch: processed {len(cached_images)}, expected {len(self.data)}")

            # Create cache data structure
            cache_data = {
                'processed_images': np.array(cached_images),
                'targets': np.array(cached_targets),
                'params_hash': self._get_cache_key(),
                'version': '1.0',
                'dataset_size': len(self.data),
                'resolution': '224x224'
            }

            # Validate cache data structure
            self._validate_cache_data(cache_data)

            # Save cache atomically with error handling
            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                if not os.path.exists(temp_path):
                    raise IOError("Temporary cache file was not created")

                file_size = os.path.getsize(temp_path)
                if file_size < 1024:
                    raise IOError(f"Cache file too small: {file_size} bytes")

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

            logging.info(f"224x224 cache built successfully with {len(cached_images)} images")

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
        """Clean up failed cache files safely."""
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
        """Validate cache data structure before saving."""
        required_keys = ['processed_images', 'targets', 'params_hash']
        for key in required_keys:
            if key not in cache_data:
                raise ValueError(f"Cache data missing required key: {key}")

        processed_images = cache_data['processed_images']
        targets = cache_data['targets']

        if not isinstance(processed_images, np.ndarray):
            raise ValueError(f"Invalid processed_images type: {type(processed_images)}")

        if processed_images.ndim != 4 or processed_images.shape[1:] != (32, 32, 3):
            raise ValueError(f"Invalid processed_images shape: {processed_images.shape}")

        if len(targets) != len(processed_images):
            raise ValueError("Cache data arrays have mismatched lengths")

    def _load_cache_with_validation(self, cache_path: str) -> None:
        """Load preprocessed images from cache file with comprehensive validation."""
        try:
            if not os.path.exists(cache_path):
                logging.warning(f"Cache file does not exist: {cache_path}")
                self._build_cache_with_validation(cache_path)
                return

            file_size = os.path.getsize(cache_path)
            if file_size < 1024:
                logging.warning(f"Cache file too small ({file_size} bytes), rebuilding cache.")
                os.remove(cache_path)
                self._build_cache_with_validation(cache_path)
                return

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

            required_keys = ['processed_images', 'targets', 'params_hash']
            for key in required_keys:
                if key not in cache_data:
                    logging.warning(f"Cache missing required key '{key}'. Rebuilding cache.")
                    self._handle_corrupted_cache(cache_path)
                    return

            # Validate cache parameters using hash comparison
            current_hash = self._get_cache_key()
            cached_hash = cache_data.get('params_hash')
            if cached_hash != current_hash:
                logging.info(f"Cache parameter mismatch: cached={cached_hash}, current={current_hash}. Rebuilding cache for new parameters.")
                os.remove(cache_path)
                self._build_cache_with_validation(cache_path)
                return

            # Comprehensive integrity checking
            expected_size = len(self.data)
            cached_images = cache_data['processed_images']
            cached_targets = cache_data['targets']

            if len(cached_images) != expected_size:
                logging.warning(f"Cache size mismatch: expected {expected_size}, got {len(cached_images)}. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            if len(cached_targets) != expected_size:
                logging.warning(f"Cache data inconsistency: images={len(cached_images)}, targets={len(cached_targets)}. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            # Validate data types and shapes
            if not isinstance(cached_images, np.ndarray) or cached_images.ndim != 4:
                logging.warning(f"Invalid cached images format: type={type(cached_images)}, shape={getattr(cached_images, 'shape', 'N/A')}. Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            if cached_images.shape[1:] != (32, 32, 3):
                logging.warning(f"Invalid cached image shape: {cached_images.shape}. Expected (N, 32, 32, 3). Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            # All validations passed
            self._cached_data = cache_data
            self._cache_loaded = True
            logging.info(f"224x224 cache loaded successfully with {len(cached_images)} images")

        except MemoryError as e:
            logging.error(f"Memory error loading cache: {e}. Falling back to original processing.")
            self._disable_cache_with_fallback(f"Memory error loading cache: {e}")
        except Exception as e:
            logging.error(f"Unexpected error loading cache: {e}. Rebuilding cache.")
            self._handle_corrupted_cache(cache_path)

    def _handle_corrupted_cache(self, cache_path: str) -> None:
        """Handle corrupted cache by removing it and rebuilding."""
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logging.info(f"Removed corrupted cache file: {cache_path}")
        except OSError as e:
            logging.warning(f"Could not remove corrupted cache file {cache_path}: {e}")

        try:
            self._build_cache_with_validation(cache_path)
        except Exception as e:
            logging.error(f"Failed to rebuild cache after corruption: {e}")
            self._disable_cache_with_fallback(f"Cache rebuild failed: {e}")

    def _get_cached_item(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get cached item with proper error handling."""
        try:
            if self._cached_data is None:
                raise ValueError("Cache data is None")

            cached_images = self._cached_data['processed_images']
            cached_targets = self._cached_data['targets']

            if index >= len(cached_images):
                raise IndexError(f"Index {index} out of range for cached data (size: {len(cached_images)})")

            # Get cached data
            img_array = cached_images[index]
            target = cached_targets[index]

            # Convert to PIL Image
            img = Image.fromarray(img_array, mode='RGB')

            # Apply transforms
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

        except Exception as e:
            logging.error(f"Error retrieving cached item {index}: {e}")
            raise


class MyEinstellungCIFAR100_224(MyCIFAR100, EinstellungMixin):
    """
    Training CIFAR-100 dataset with Einstellung Effect for 224x224 ViT.
    Inherits from proven MyCIFAR100 implementation with caching support.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, apply_shortcut=False, mask_shortcut=False,
                 patch_size=16, patch_color=(255, 0, 255), enable_cache=True):
        # CRITICAL: Set root before calling super().__init__()
        self.root = root

        # Initialize not_aug_transform before other initialization
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])

        # Initialize Einstellung parameters
        self.init_einstellung(apply_shortcut, mask_shortcut, patch_size, patch_color)

        # Caching parameters
        self.enable_cache = enable_cache
        self._cached_data = None
        self._cache_loaded = False
        self._cache_error_count = 0
        self._max_cache_errors = 3  # Maximum cache errors before disabling cache permanently

        # Call parent constructor with proper download logic
        super().__init__(root, train, transform, target_transform,
                        download=not self._check_integrity())

        # Filter to Einstellung classes after initialization
        self.filter_einstellung_classes()

        # Setup cache if enabled
        if self.enable_cache:
            self._setup_cache_with_fallback()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Get item with Einstellung Effect processing."""
        # Return cached data if available, with comprehensive error handling
        if self.enable_cache and self._cache_loaded and self._cached_data is not None:
            try:
                return self._get_cached_item(index)
            except Exception as e:
                logging.warning(f"Cache retrieval failed for index {index}: {e}. Using original processing.")
                # Continue to original processing below

        # Original processing (fallback)
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

    def _get_cache_key(self) -> str:
        """Generate a secure hash key based on Einstellung parameters for cache identification."""
        params = {
            'apply_shortcut': self.apply_shortcut,
            'mask_shortcut': self.mask_shortcut,
            'patch_size': self.patch_size,
            'patch_color': tuple(self.patch_color),
            'train': self.train,
            'resolution': '224x224'  # Distinguish from 32x32 cache
        }
        param_str = str(sorted(params.items()))
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _get_cache_path(self) -> str:
        """Get the cache file path using Mammoth's base_path structure."""
        cache_key = self._get_cache_key()
        cache_dir = os.path.join(base_path(), 'CIFAR100', 'einstellung_cache_224')
        os.makedirs(cache_dir, exist_ok=True)
        split_name = 'train' if self.train else 'test'
        cache_filename = f'{split_name}_{cache_key}.pkl'
        return os.path.join(cache_dir, cache_filename)

    def _setup_cache_with_fallback(self) -> None:
        """Setup cache with comprehensive error handling and automatic fallback."""
        if self._cache_error_count >= self._max_cache_errors:
            logging.warning(f"Cache disabled due to {self._cache_error_count} consecutive errors. Using original processing.")
            self.enable_cache = False
            return

        try:
            cache_path = self._get_cache_path()
            cache_dir = os.path.dirname(cache_path)

            if not self._check_disk_space(cache_dir):
                logging.error("Insufficient disk space for cache operations. Falling back to original processing.")
                self._disable_cache_with_fallback("Insufficient disk space")
                return

            if os.path.exists(cache_path):
                logging.info(f"Loading Einstellung 224x224 cache from {cache_path}")
                self._load_cache_with_validation(cache_path)
            else:
                logging.info(f"Building Einstellung 224x224 cache at {cache_path}")
                self._build_cache_with_validation(cache_path)

        except (OSError, IOError) as e:
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
        """Disable cache and ensure fallback to original implementation."""
        self._cache_error_count += 1
        self._cache_loaded = False
        self._cached_data = None

        if self._cache_error_count >= self._max_cache_errors:
            logging.warning(f"Disabling cache permanently after {self._cache_error_count} errors. Reason: {reason}")
            self.enable_cache = False
        else:
            logging.warning(f"Cache error #{self._cache_error_count}: {reason}. Will retry on next initialization.")

    def _check_disk_space(self, path: str, required_mb: int = 2048) -> bool:
        """Check if sufficient disk space is available for cache operations (224x224 needs more space)."""
        try:
            os.makedirs(path, exist_ok=True)
            statvfs = os.statvfs(path)
            available_bytes = statvfs.f_frsize * statvfs.f_bavail
            available_mb = available_bytes / (1024 * 1024)

            if available_mb < required_mb:
                logging.warning(f"Insufficient disk space: {available_mb:.1f}MB available, {required_mb}MB required")
                return False
            return True
        except (OSError, AttributeError) as e:
            logging.warning(f"Could not check disk space: {e}. Proceeding with cache operations.")
            return True

    def _build_cache_with_validation(self, cache_path: str) -> None:
        """Build cache with comprehensive error handling and validation."""
        temp_path = cache_path + '.tmp'

        try:
            cache_dir = os.path.dirname(cache_path)
            if not self._check_disk_space(cache_dir, required_mb=4096):  # 224x224 needs more space
                raise OSError(errno.ENOSPC, "Insufficient disk space for cache building")

            logging.info("Preprocessing 224x224 images for cache...")

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
                    img = self.apply_einstellung_effect(img, i)

                    # Validate processed image (224x224 after resize)
                    img_array = np.array(img)
                    if img_array.shape != (32, 32, 3):  # Still CIFAR size before transform
                        raise ValueError(f"Invalid processed image shape: {img_array.shape}")

                    # Store processed images as numpy arrays for caching
                    cached_images.append(img_array)
                    cached_targets.append(target)
                    cached_not_aug_images.append(np.array(original_img))

                    # Progress logging and periodic disk space check
                    if (i + 1) % 1000 == 0:
                        logging.info(f"Processed {i + 1}/{len(self.data)} images for 224x224 cache")
                        if not self._check_disk_space(cache_dir, required_mb=2048):
                            raise OSError(errno.ENOSPC, "Disk space exhausted during cache building")

                except Exception as e:
                    logging.error(f"Error processing image {i}: {e}")
                    raise

            # Validate processed data before saving
            if len(cached_images) != len(self.data):
                raise ValueError(f"Cache size mismatch: processed {len(cached_images)}, expected {len(self.data)}")

            # Create cache data structure
            cache_data = {
                'processed_images': np.array(cached_images),
                'targets': np.array(cached_targets),
                'not_aug_images': np.array(cached_not_aug_images),
                'params_hash': self._get_cache_key(),
                'version': '1.0',
                'dataset_size': len(self.data),
                'resolution': '224x224'
            }

            # Validate cache data structure
            self._validate_cache_data(cache_data)

            # Save cache atomically with error handling
            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

                if not os.path.exists(temp_path):
                    raise IOError("Temporary cache file was not created")

                file_size = os.path.getsize(temp_path)
                if file_size < 1024:
                    raise IOError(f"Cache file too small: {file_size} bytes")

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

            logging.info(f"224x224 cache built successfully with {len(cached_images)} images")

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
        """Clean up failed cache files safely."""
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
        """Validate cache data structure before saving."""
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
        """Load preprocessed images from cache file with comprehensive validation."""
        try:
            if not os.path.exists(cache_path):
                logging.warning(f"Cache file does not exist: {cache_path}")
                self._build_cache_with_validation(cache_path)
                return

            file_size = os.path.getsize(cache_path)
            if file_size < 1024:
                logging.warning(f"Cache file too small ({file_size} bytes), rebuilding cache.")
                os.remove(cache_path)
                self._build_cache_with_validation(cache_path)
                return

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

            # Validate cache parameters using hash comparison
            current_hash = self._get_cache_key()
            cached_hash = cache_data.get('params_hash')
            if cached_hash != current_hash:
                logging.info(f"Cache parameter mismatch: cached={cached_hash}, current={current_hash}. Rebuilding cache for new parameters.")
                os.remove(cache_path)
                self._build_cache_with_validation(cache_path)
                return

            # Comprehensive integrity checking
            expected_size = len(self.data)
            cached_images = cache_data['processed_images']
            cached_targets = cache_data['targets']
            cached_not_aug = cache_data['not_aug_images']

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

            if cached_images.shape[1:] != (32, 32, 3):
                logging.warning(f"Invalid cached image shape: {cached_images.shape}. Expected (N, 32, 32, 3). Rebuilding cache.")
                self._handle_corrupted_cache(cache_path)
                return

            # All validations passed
            self._cached_data = cache_data
            self._cache_loaded = True
            logging.info(f"224x224 cache loaded successfully with {len(cached_images)} images")

        except MemoryError as e:
            logging.error(f"Memory error loading cache: {e}. Falling back to original processing.")
            self._disable_cache_with_fallback(f"Memory error loading cache: {e}")
        except Exception as e:
            logging.error(f"Unexpected error loading cache: {e}. Rebuilding cache.")
            self._handle_corrupted_cache(cache_path)

    def _handle_corrupted_cache(self, cache_path: str) -> None:
        """Handle corrupted cache by removing it and rebuilding."""
        try:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logging.info(f"Removed corrupted cache file: {cache_path}")
        except OSError as e:
            logging.warning(f"Could not remove corrupted cache file {cache_path}: {e}")

        try:
            self._build_cache_with_validation(cache_path)
        except Exception as e:
            logging.error(f"Failed to rebuild cache after corruption: {e}")
            self._disable_cache_with_fallback(f"Cache rebuild failed: {e}")

    def _get_cached_item(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Get cached item with proper error handling."""
        try:
            if self._cached_data is None:
                raise ValueError("Cache data is None")

            cached_images = self._cached_data['processed_images']
            cached_targets = self._cached_data['targets']
            cached_not_aug = self._cached_data['not_aug_images']

            if index >= len(cached_images):
                raise IndexError(f"Index {index} out of range for cached data (size: {len(cached_images)})")

            # Get cached data
            img_array = cached_images[index]
            target = cached_targets[index]
            not_aug_array = cached_not_aug[index]

            # Convert to PIL Image
            img = Image.fromarray(img_array, mode='RGB')
            not_aug_img = self.not_aug_transform(Image.fromarray(not_aug_array, mode='RGB'))

            # Apply transforms
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target, not_aug_img

        except Exception as e:
            logging.error(f"Error retrieving cached item {index}: {e}")
            raise


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
        self.enable_cache = getattr(args, 'einstellung_enable_cache', True)

        logging.info(f"Initialized {self.NAME} with Einstellung parameters:")
        logging.info(f"  - Apply shortcut: {self.apply_shortcut}")
        logging.info(f"  - Mask shortcut: {self.mask_shortcut}")
        logging.info(f"  - Patch size: {self.patch_size}")
        logging.info(f"  - Patch color: {self.patch_color}")
        logging.info(f"  - Enable cache: {self.enable_cache}")

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Get train and test data loaders using Einstellung datasets."""
        # Use parent class transforms (proven to work with ViT)
        transform = self.TRANSFORM
        test_transform = self.TEST_TRANSFORM

        # Create Einstellung datasets using the working base classes with caching
        train_dataset = MyEinstellungCIFAR100_224(
            base_path() + 'CIFAR100', train=True, download=True, transform=transform,
            apply_shortcut=self.apply_shortcut, mask_shortcut=self.mask_shortcut,
            patch_size=self.patch_size, patch_color=self.patch_color,
            enable_cache=self.enable_cache
        )

        test_dataset = TEinstellungCIFAR100_224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=self.apply_shortcut, mask_shortcut=self.mask_shortcut,
            patch_size=self.patch_size, patch_color=self.patch_color,
            enable_cache=self.enable_cache
        )

        # Use parent's proven loader creation
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_evaluation_subsets(self) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Create evaluation subsets for comprehensive Einstellung metrics.
        Uses cached evaluation subsets when available for improved performance.

        Returns:
            Dictionary mapping subset names to DataLoaders
        """
        # Try to load cached evaluation subsets first
        if self.enable_cache:
            try:
                cached_subsets = self._get_cached_evaluation_subsets()
                if cached_subsets is not None:
                    logging.info("Using cached evaluation subsets (224x224)")
                    return cached_subsets
            except Exception as e:
                logging.warning(f"Failed to load cached evaluation subsets (224x224): {e}. Creating fresh subsets.")

        # Create fresh evaluation subsets (original implementation)
        test_transform = self.TEST_TRANSFORM
        subsets = {}

        # T1_all: All Task 1 data
        t1_dataset = TEinstellungCIFAR100_224(
            base_path() + 'CIFAR100', train=False, download=True, transform=test_transform,
            apply_shortcut=False, mask_shortcut=False, patch_size=self.patch_size,
            patch_color=self.patch_color, enable_cache=self.enable_cache
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
            patch_color=self.patch_color, enable_cache=self.enable_cache
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
            patch_color=self.patch_color, enable_cache=self.enable_cache
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

        # Cache the evaluation subsets for future use
        if self.enable_cache:
            try:
                self._cache_evaluation_subsets(subsets)
                logging.info("Cached evaluation subsets for future use (224x224)")
            except Exception as e:
                logging.warning(f"Failed to cache evaluation subsets (224x224): {e}")

        return subsets

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        cifar100_classes = CIFAR100(base_path() + 'CIFAR100', train=True, download=True).classes
        used_class_names = [cifar100_classes[idx] for idx in ALL_USED_FINE_LABELS]
        classes = fix_class_names_order(used_class_names, self.args)
        self.class_names = classes
        return self.class_names

    def _get_evaluation_subset_cache_key(self) -> str:
        """
        Generate cache key for evaluation subsets based on parameters (224x224 version).

        Returns:
            Secure hash string for evaluat subset cache key
        """
        # Include batch_size and other relevant parameters for evaluation subsets
        params = {
            'patch_size': self.patch_size,
            'patch_color': tuple(self.patch_color),
            'batch_size': getattr(self.args, 'batch_size', 32) if hasattr(self, 'args') else 32,
            'evaluation_subsets': True,  # Distinguish from regular cache
            'resolution': '224x224'  # Distinguish from 32x32 version
        }

        param_str = str(sorted(params.items()))
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def _get_evaluation_subset_cache_path(self) -> str:
        """
        Get cache file path for evaluation subsets (224x224 version).

        Returns:
            Full path to evaluation subset cache file
        """
        cache_key = self._get_evaluation_subset_cache_key()
        cache_dir = os.path.join(base_path(), 'CIFAR100', 'einstellung_cache')
        os.makedirs(cache_dir, exist_ok=True)

        cache_filename = f'eval_subsets_224_{cache_key}.pkl'
        return os.path.join(cache_dir, cache_filename)

    def _cache_evaluation_subsets(self, subsets: Dict[str, torch.utils.data.DataLoader]) -> None:
        """
        Cache evaluation subsets for future use (224x224 version).

        Args:
            subsets: Dictionary of evaluation subset DataLoaders to cache
        """
        try:
            cache_path = self._get_evaluation_subset_cache_path()
            temp_path = cache_path + '.tmp'

            # Extract data from DataLoaders for caching
            cached_subsets = {}

            for subset_name, dataloader in subsets.items():
                logging.info(f"Caching evaluation subset (224x224): {subset_name}")

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
                            logging.debug(f"Cached {len(subset_data)} items for {subset_name} (224x224)")

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

                logging.info(f"Cached {len(subset_data)} samples for {subset_name} (224x224)")

            # Create cache data structure
            cache_data = {
                'subsets': cached_subsets,
                'params_hash': self._get_evaluation_subset_cache_key(),
                'version': '1.0',
                'resolution': '224x224',
                'creation_time': os.path.getmtime(cache_path) if os.path.exists(cache_path) else None
            }

            # Save cache atomically
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Verify and rename
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1024:
                os.rename(temp_path, cache_path)
                logging.info(f"Evaluation subsets cached successfully at {cache_path} (224x224)")
            else:
                raise IOError("Cache file verification failed")

        except Exception as e:
            logging.error(f"Failed to cache evaluation subsets (224x224): {e}")
            # Clean up failed cache
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            raise

    def _get_cached_evaluation_subsets(self) -> Optional[Dict[str, torch.utils.data.DataLoader]]:
        """
        Load cached evaluation subsets if available and valid (224x224 version).

        Returns:
            Dictionary of cached evaluation subset DataLoaders, or None if not available
        """
        try:
            cache_path = self._get_evaluation_subset_cache_path()

            if not os.path.exists(cache_path):
                return None

            # Check file size
            if os.path.getsize(cache_path) < 1024:
                logging.warning("Evaluation subset cache file too small (224x224), rebuilding")
                os.remove(cache_path)
                return None

            # Load cache data
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache structure
            if not isinstance(cache_data, dict) or 'subsets' not in cache_data:
                logging.warning("Invalid evaluation subset cache structure (224x224), rebuilding")
                os.remove(cache_path)
                return None

            # Validate parameters
            current_hash = self._get_evaluation_subset_cache_key()
            cached_hash = cache_data.get('params_hash')
            if cached_hash != current_hash:
                logging.info(f"Evaluation subset cache parameter mismatch (224x224): {cached_hash} vs {current_hash}, rebuilding")
                os.remove(cache_path)
                return None

            # Reconstruct DataLoaders from cached data
            test_transform = self.TEST_TRANSFORM

            subsets = {}
            cached_subsets = cache_data['subsets']

            for subset_name, subset_cache in cached_subsets.items():
                try:
                    # Create dataset from cached data
                    cached_dataset = CachedEvaluationDataset224(
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

                    logging.debug(f"Loaded cached evaluation subset (224x224): {subset_name} with {len(cached_dataset)} samples")

                except Exception as e:
                    logging.error(f"Failed to reconstruct DataLoader for {subset_name} (224x224): {e}")
                    return None

            # Validate that all expected subsets are present
            expected_subsets = {'T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal'}
            if not expected_subsets.issubset(set(subsets.keys())):
                logging.warning(f"Missing evaluation subsets in cache (224x224). Expected: {expected_subsets}, Got: {set(subsets.keys())}")
                return None

            return subsets

        except Exception as e:
            logging.error(f"Failed to load cached evaluation subsets (224x224): {e}")
            # Clean up corrupted cache
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except OSError:
                pass
            return None

    # Inherit all the proven ViT-specific methods from parent
    # No need to override: get_transform, get_backbone, get_epochs, get_batch_size, etc.


class CachedEvaluationDataset224(torch.utils.data.Dataset):
    """
    Dataset wrapper for cached evaluation subset data (224x224 version).
    Maintains compatibility with original dataset interface.
    """

    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        """
        Initialize cached evaluation dataset for 224x224 images.

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
        Get item from cached evaluation dataset (224x224 version).

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
            # Handle different image shapes (224x224 vs 32x32)
            if img.shape == (224, 224, 3):
                img = Image.fromarray(img, mode='RGB')
            elif img.shape == (32, 32, 3):
                # Resize 32x32 to 224x224 for ViT compatibility
                img = Image.fromarray(img, mode='RGB')
                img = img.resize((224, 224), Image.BILINEAR)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

        if self.transform is not None:
            img = self.transform(img)

        return img, target
