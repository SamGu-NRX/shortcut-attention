#!/usr/bin/env python3
"""
Robust Einstellung Dataset Caching System

This module provides a completely redesigned caching approach that ensures
deterministic, consistent behavior across different continual learning methods.

Key Design Principles:
1. Cache only RAW data (original CIFAR-100 images and targets)
2. Apply ALL processing (Einstellung effects + transforms) deterministically during retrieval
3. Use index-based seeding for complete reproducibility
4. Ensure identical behavior across method instances

This fixes the fundamental issues causing cross-method inconsistencies.
"""

import os
import sys
import hashlib
import pickle
import logging
import numpy as np
import random
from typing import Tuple, Dict, Any, Optional
from PIL import Image

import torch
import torchvision.transforms as transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conf import base_path


class RobustEinstellungCache:
    """
    Robust caching system that ensures deterministic behavior across method instances.

    This cache stores only raw data and applies all processing deterministically
    during retrieval, ensuring identical results across different continual learning methods.
    """

    def __init__(self,
                 dataset_name: str,
                 train: bool,
                 apply_shortcut: bool,
                 mask_shortcut: bool,
                 patch_size: int,
                 patch_color: Tuple[int, int, int],
                 resolution: str = "32x32"):
        """
        Initialize robust cache system.

        Args:
            dataset_name: Name of the dataset (e.g., 'seq-cifar100-einstellung')
            train: Whether this is training data
            apply_shortcut: Whether to apply shortcut patches
            mask_shortcut: Whether to mask shortcut patches
            patch_size: Size of patches
            patch_color: Color of patches
            resolution: Image resolution (32x32 or 224x224)
        """
        self.dataset_name = dataset_name
        self.train = train
        self.apply_shortcut = apply_shortcut
        self.mask_shortcut = mask_shortcut
        self.patch_size = patch_size
        self.patch_color = np.array(patch_color, dtype=np.uint8)
        self.resolution = resolution

        # Cache state
        self._cache_loaded = False
        self._raw_cache_data = None
        self._cache_path = None

        # Generate cache key and path
        self._cache_key = self._generate_cache_key()
        self._cache_path = self._get_cache_path()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _generate_cache_key(self) -> str:
        """Generate cache key based on dataset parameters (excluding processing parameters)."""
        # Only include parameters that affect the RAW data selection
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
        """
        Load existing cache or build new one from raw data.

        Args:
            raw_data: Raw CIFAR-100 image data
            raw_targets: Raw CIFAR-100 targets

        Returns:
            True if cache loaded/built successfully, False otherwise
        """
        try:
            if os.path.exists(self._cache_path):
                return self._load_cache()
            else:
                return self._build_cache(raw_data, raw_targets)
        except Exception as e:
            self.logger.error(f"Cache operation failed: {e}")
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

            # Validate data integrity
            raw_images = cache_data['raw_images']
            raw_targets = cache_data['raw_targets']

            if not isinstance(raw_images, np.ndarray) or raw_images.ndim != 4:
                self.logger.warning("Invalid cache image format, rebuilding...")
                return False

            if len(raw_images) != len(raw_targets):
                self.logger.warning("Cache data length mismatch, rebuilding...")
                return False

            self._raw_cache_data = cache_data
            self._cache_loaded = True

            self.logger.info(f"Loaded cache with {len(raw_images)} samples")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
            return False

    def _build_cache(self, raw_data: np.ndarray, raw_targets: list) -> bool:
        """Build cache from raw data."""
        try:
            self.logger.info(f"Building robust cache with {len(raw_data)} samples...")

            # Store only raw data - no processing applied
            cache_data = {
                'raw_images': raw_data.copy(),  # Raw CIFAR-100 images
                'raw_targets': list(raw_targets),  # Raw targets
                'cache_key': self._cache_key,
                'version': '2.0',  # New robust version
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

            self.logger.info(f"Built cache successfully with {len(raw_data)} samples")
            return True

        except Exception as e:
            self.logger.error(f"Failed to build cache: {e}")
            return False

    def get_processed_item(self,
                          index: int,
                          transform=None,
                          target_transform=None,
                          not_aug_transform=None,
                          shortcut_labels: set = None) -> Tuple[Any, int, Any]:
        """
        Get processed item with deterministic Einstellung effects and transforms.

        This method applies ALL processing deterministically based on the index,
        ensuring identical results across different method instances.

        Args:
            index: Sample index
            transform: Transform to apply to processed image
            target_transform: Transform to apply to target
            not_aug_transform: Transform for not-augmented image
            shortcut_labels: Set of labels that should get shortcut patches

        Returns:
            Tuple of (processed_image, target, not_aug_image)
        """
        if not self._cache_loaded or self._raw_cache_data is None:
            raise RuntimeError("Cache not loaded")

        # Get raw data
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
        if shortcut_labels and target in shortcut_labels:
            img = self._apply_deterministic_einstellung_effect(img, index)

        # Apply transforms deterministically
        processed_img = self._apply_deterministic_transform(transform, img, index)
        not_aug_img = self._apply_deterministic_transform(not_aug_transform, original_img, index)

        # Apply target transform
        if target_transform is not None:
            target = target_transform(target)

        return processed_img, target, not_aug_img

    def _apply_deterministic_einstellung_effect(self, img: Image.Image, index: int) -> Image.Image:
        """Apply Einstellung effect deterministically based on index."""
        if self.patch_size <= 0:
            return img

        # Convert to numpy for manipulation
        arr = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]

        if self.patch_size > min(h, w):
            return img

        # Use deterministic random state based on index
        # This ensures identical patch placement across method instances
        rng = np.random.RandomState(index + 42)  # Fixed seed offset
        x = rng.randint(0, w - self.patch_size + 1)
        y = rng.randint(0, h - self.patch_size + 1)

        if self.mask_shortcut:
            # Mask the shortcut area (set to black)
            arr[y:y+self.patch_size, x:x+self.patch_size] = 0
        elif self.apply_shortcut:
            # Apply magenta shortcut patch
            arr[y:y+self.patch_size, x:x+self.patch_size] = self.patch_color

        return Image.fromarray(arr)

    def _apply_deterministic_transform(self, transform, img, index: int):
        """Apply transform deterministically based on index."""
        if transform is None:
            return img

        # Save current random states
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        py_state = random.getstate()

        try:
            # Set deterministic seed based on index and transform hash
            # This ensures different transforms get different but deterministic seeds
            transform_hash = hash(str(transform)) % (2**16)  # Get transform-specific hash
            deterministic_seed = (index + transform_hash + 12345) % (2**32)

            torch.manual_seed(deterministic_seed)
            np.random.seed(deterministic_seed)
            random.seed(deterministic_seed)

            # Apply transform with deterministic seed
            return transform(img)

        finally:
            # Always restore original random states
            torch.set_rng_state(torch_state)
            np.random.set_state(np_state)
            random.setstate(py_state)

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        if not self._cache_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "cache_key": self._cache_key,
            "cache_path": self._cache_path,
            "sample_count": len(self._raw_cache_data['raw_images']),
            "version": self._raw_cache_data.get('version', 'unknown'),
            "dataset_name": self._raw_cache_data.get('dataset_name'),
            "train": self._raw_cache_data.get('train'),
            "resolution": self._raw_cache_data.get('resolution')
        }

    def is_loaded(self) -> bool:
        """Check if cache is loaded."""
        return self._cache_loaded and self._raw_cache_data is not None


class RobustEinstellungDatasetMixin:
    """
    Mixin class that provides robust caching functionality to Einstellung datasets.

    This mixin replaces the existing caching system with the robust approach
    that ensures cross-method consistency.
    """

    def init_robust_cache(self):
        """Initialize robust cache system."""
        self._robust_cache = RobustEinstellungCache(
            dataset_name=getattr(self, 'dataset_name', 'seq-cifar100-einstellung'),
            train=self.train,
            apply_shortcut=self.apply_shortcut,
            mask_shortcut=self.mask_shortcut,
            patch_size=self.patch_size,
            patch_color=tuple(self.patch_color),
            resolution=getattr(self, 'resolution', '32x32')
        )

    def setup_robust_cache(self, raw_data: np.ndarray, raw_targets: list) -> bool:
        """Setup robust cache with raw data."""
        if not hasattr(self, '_robust_cache'):
            self.init_robust_cache()

        return self._robust_cache.load_or_build_cache(raw_data, raw_targets)

    def get_robust_cached_item(self, index: int):
        """Get item using robust cache."""
        if not hasattr(self, '_robust_cache') or not self._robust_cache.is_loaded():
            raise RuntimeError("Robust cache not initialized or loaded")

        return self._robust_cache.get_processed_item(
            index=index,
            transform=self.transform,
            target_transform=self.target_transform,
            not_aug_transform=getattr(self, 'not_aug_transform', None),
            shortcut_labels=getattr(self, 'shortcut_labels', set())
        )

    def get_robust_cache_info(self) -> Dict[str, Any]:
        """Get robust cache information."""
        if not hasattr(self, '_robust_cache'):
            return {"initialized": False}

        info = self._robust_cache.get_cache_info()
        info["initialized"] = True
        return info


def validate_robust_cache_consistency(dataset_class,
                                    root: str,
                                    sample_size: int = 50,
                                    methods: list = None) -> Dict[str, Any]:
    """
    Validate that the robust cache provides consistent results across method instances.

    Args:
        dataset_class: Dataset class to test
        root: Dataset root directory
        sample_size: Number of samples to validate
        methods: List of method names (for logging)

    Returns:
        Dictionary with validation results
    """
    methods = methods or ['SGD', 'EWC', 'DER++']

    print(f"🔍 Validating robust cache consistency...")
    print(f"   Dataset: {dataset_class.__name__}")
    print(f"   Methods: {methods}")
    print(f"   Sample size: {sample_size}")

    try:
        # Create multiple dataset instances (simulating different methods)
        datasets = []
        for method in methods:
            dataset = dataset_class(
                root=root,
                train=False,
                download=True,
                apply_shortcut=True,
                patch_size=4,
                enable_cache=True
            )
            datasets.append(dataset)

        # Compare data across instances
        inconsistencies = []
        reference_dataset = datasets[0]

        for i in range(min(sample_size, len(reference_dataset))):
            ref_item = reference_dataset[i]
            ref_img_hash = hashlib.sha256(np.array(ref_item[0]).tobytes()).hexdigest()
            ref_target = ref_item[1]

            for j, dataset in enumerate(datasets[1:], 1):
                item = dataset[i]
                img_hash = hashlib.sha256(np.array(item[0]).tobytes()).hexdigest()
                target = item[1]

                if ref_img_hash != img_hash:
                    inconsistencies.append({
                        "index": i,
                        "type": "image_mismatch",
                        "reference_method": methods[0],
                        "comparison_method": methods[j],
                        "ref_hash": ref_img_hash[:16],
                        "comp_hash": img_hash[:16]
                    })

                if ref_target != target:
                    inconsistencies.append({
                        "index": i,
                        "type": "target_mismatch",
                        "reference_method": methods[0],
                        "comparison_method": methods[j],
                        "ref_target": ref_target,
                        "comp_target": target
                    })

        success = len(inconsistencies) == 0

        if success:
            print(f"   ✅ Robust cache validation PASSED")
            print(f"   All {len(methods)} methods get identical data")
        else:
            print(f"   ❌ Robust cache validation FAILED")
            print(f"   Found {len(inconsistencies)} inconsistencies")

        return {
            "success": success,
            "methods_tested": methods,
            "samples_validated": min(sample_size, len(reference_dataset)),
            "inconsistencies": len(inconsistencies),
            "inconsistency_details": inconsistencies[:5]  # First 5 for debugging
        }

    except Exception as e:
        print(f"   ❌ Validation failed with error: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


if __name__ == '__main__':
    # Test the robust cache system
    print("🚀 Testing Robust Einstellung Cache System")
    print("=" * 60)

    # This would be integrated into the actual dataset classes
    print("This module provides the foundation for robust caching.")
    print("Integration with existing dataset classes is needed.")
    print("Run the validation after integration to verify consistency.")
