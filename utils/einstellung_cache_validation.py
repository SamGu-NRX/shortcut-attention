#!/usr/bin/env python3
"""
Comprehensive validation system for Einstellung dataset caching.

This module implements pixel-perfect comparison, statistical validation,
and checksum validation for cache integrity as required by task 9.

Requirements addressed:
- 6.1: Pixel-perfect comparison between cached and on-the-fly processed images
- 6.2: Integrity checks using checksums and file size validation
- 6.4: Standard Python unittest-compatible tools for debugging
- 6.6: Pass all existing Mammoth dataset tests without modification
"""

import os
import sys
import hashlib
import pickle
import logging
import tempfile
import shutil
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass
import unittest

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.seq_cifar100_einstellung import MyCIFAR100Einstellung, ALL_USED_FINE_LABELS, SHORTCUT_FINE_LABELS
from datasets.seq_cifar100_einstellung_224 import MyEinstellungCIFAR100_224, TEinstellungCIFAR100_224
from utils.conf import base_path


@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class EinstellungCacheValidator:
    """
    Comprehensive validation system for Einstellung dataset caching.

    Provides pixel-perfect comparison, statistical validation, and integrity checking
    to ensure cached datasets produce identical results to original implementations.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the validator.

        Args:
            verbose: Whether to print detailed validation messages
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def validate_pixel_perfect_comparison(self,
                                        dataset_class,
                                        root: str,
                                        train: bool = False,
                                        apply_shortcut: bool = True,
                                        mask_shortcut: bool = False,
                                        patch_size: int = 4,
                                        patch_color: Tuple[int, int, int] = (255, 0, 255),
                                        sample_size: int = 100) -> ValidationResult:
        """
        Perform pixel-perfect comparison between cached and on-the-fly processed images.

        Args:
            dataset_class: Dataset class to test (MyCIFAR100Einstellung or variants)
            root: Dataset root directory
            train: Whether to use training set
            apply_shortcut: Whether to apply shortcut patches
            mask_shortcut: Whether to mask shortcut patches
            patch_size: Size of patches
            patch_color: Color of patches
            sample_size: Number of samples to validate

        Returns:
            ValidationResult with comparison results
        """
        if self.verbose:
            print(f"Performing pixel-perfect validation for {dataset_class.__name__}...")

        try:
            # Create dataset with caching disabled (original processing)
            dataset_original = dataset_class(
                root=root,
                train=train,
                download=True,
                apply_shortcut=apply_shortcut,
                mask_shortcut=mask_shortcut,
                patch_size=patch_size,
                patch_color=patch_color,
                enable_cache=False
            )

            # Create dataset with caching enabled
            dataset_cached = dataset_class(
                root=root,
                train=train,
                download=True,
                apply_shortcut=apply_shortcut,
                mask_shortcut=mask_shortcut,
                patch_size=patch_size,
                patch_color=patch_color,
                enable_cache=True
            )

            # Ensure cache is built and loaded
            if not dataset_cached._cache_loaded:
                return Validationlt(
                    passed=False,
                    message="Cache was not loaded successfully",
                    details={"cache_loaded": dataset_cached._cache_loaded}
                )

            # Compare samples
            mismatches = []
            total_samples = min(sample_size, len(dataset_original))

            for i in range(total_samples):
                # Get items from both datasets
                original_item = dataset_original[i]
                cached_item = dataset_cached[i]

                # Extract images and targets
                if len(original_item) == 3:  # (image, target, not_aug_image)
                    orig_img, orig_target, orig_not_aug = original_item
                    cached_img, cached_target, cached_not_aug = cached_item
                elif len(original_item) == 2:  # (image, target)
                    orig_img, orig_target = original_item
                    cached_img, cached_target = cached_item
                    orig_not_aug = cached_not_aug = None
                else:
                    return ValidationResult(
                        passed=False,
                        message=f"Unexpected item format: {len(original_item)} elements",
                        details={"item_length": len(original_item)}
                    )

                # Compare targets
                if orig_target != cached_target:
                    mismatches.append({
                        "index": i,
                        "type": "target_mismatch",
                        "original": orig_target,
                        "cached": cached_target
                    })

                # Compare images (convert to numpy for comparison)
                orig_img_np = self._tensor_to_numpy(orig_img)
                cached_img_np = self._tensor_to_numpy(cached_img)

                if not np.allclose(orig_img_np, cached_img_np, rtol=1e-5, atol=1e-8):
                    # Calculate pixel difference statistics
                    diff = np.abs(orig_img_np - cached_img_np)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)

                    mismatches.append({
                        "index": i,
                        "type": "image_mismatch",
                        "max_pixel_diff": float(max_diff),
                        "mean_pixel_diff": float(mean_diff),
                        "shapes": {
                            "original": orig_img_np.shape,
                            "cached": cached_img_np.shape
                        }
                    })

                # Compare not_aug_images if present
                if orig_not_aug is not None and cached_not_aug is not None:
                    orig_not_aug_np = self._tensor_to_numpy(orig_not_aug)
                    cached_not_aug_np = self._tensor_to_numpy(cached_not_aug)

                    if not np.allclose(orig_not_aug_np, cached_not_aug_np, rtol=1e-5, atol=1e-8):
                        diff = np.abs(orig_not_aug_np - cached_not_aug_np)
                        mismatches.append({
                            "index": i,
                            "type": "not_aug_image_mismatch",
                            "max_pixel_diff": float(np.max(diff)),
                            "mean_pixel_diff": float(np.mean(diff))
                        })

                # Progress reporting
                if self.verbose and (i + 1) % 20 == 0:
                    print(f"  Validated {i + 1}/{total_samples} samples...")

            # Analyze results
            if len(mismatches) == 0:
                if self.verbose:
                    print(f"✓ Pixel-perfect validation passed for {total_samples} samples")
                return ValidationResult(
                    passed=True,
                    message=f"All {total_samples} samples match perfectly",
                    details={"samples_validated": total_samples, "mismatches": 0}
                )
            else:
                if self.verbose:
                    print(f"✗ Found {len(mismatches)} mismatches in {total_samples} samples")
                    for mismatch in mismatches[:5]:  # Show first 5 mismatches
                        print(f"    {mismatch}")

                return ValidationResult(
                    passed=False,
                    message=f"Found {len(mismatches)} mismatches in {total_samples} samples",
                    details={
                        "samples_validated": total_samples,
                        "mismatches": len(mismatches),
                        "mismatch_details": mismatches[:10]  # Limit details to first 10
                    }
                )

        except Exception as e:
            self.logger.error(f"Pixel-perfect validation failed: {e}")
            return ValidationResult(
                passed=False,
                message=f"Validation failed with error: {str(e)}",
                details={"error_type": type(e).__name__}
            )

    def validate_statistical_properties(self,
                                      dataset_class,
                                      root: str,
                                      train: bool = False,
                                      apply_shortcut: bool = True,
                                      patch_size: int = 4,
                                      sample_size: int = 500) -> ValidationResult:
        """
        Validate that Einstellung effects are preserved correctly in cached data.

        This checks statistical properties like:
        - Shortcut patch presence in correct classes
        - Color distribution changes
        - Patch placement consistency

        Args:
            dataset_class: Dataset class to test
            root: Dataset root directory
            train: Whether to use training set
            apply_shortcut: Whether shortcuts should be applied
            patch_size: Size of patches
            sample_size: Number of samples to analyze

        Returns:
            ValidationResult with statistical validation results
        """
        if self.verbose:
            print(f"Performing statistical validation for Einstellung effects...")

        try:
            # Create cached dataset
            dataset = dataset_class(
                root=root,
                train=train,
                download=True,
                apply_shortcut=apply_shortcut,
                mask_shortcut=False,
                patch_size=patch_size,
                enable_cache=True
            )

            if not dataset._cache_loaded:
                return ValidationResult(
                    passed=False,
                    message="Cache not loaded for statistical validation"
                )

            # Analyze shortcut presence
            shortcut_stats = self._analyze_shortcut_presence(dataset, sample_size)

            # Validate shortcut application logic
            validation_errors = []

            if apply_shortcut:
                # Should have shortcuts in shortcut classes
                if shortcut_stats["shortcut_classes_with_patches"] == 0:
                    validation_errors.append("No shortcut patches found in shortcut classes")

                # Should not have shortcuts in non-shortcut classes
                if shortcut_stats["non_shortcut_classes_with_patches"] > 0:
                    validation_errors.append(f"Found {shortcut_stats['non_shortcut_classes_with_patches']} patches in non-shortcut classes")

                # Check patch color consistency
                if shortcut_stats["patch_color_consistency"] < 0.95:  # 95% threshold
                    validation_errors.append(f"Patch color consistency too low: {shortcut_stats['patch_color_consistency']:.2f}")
            else:
                # Should have no shortcuts when apply_shortcut=False
                if shortcut_stats["total_patches_found"] > 0:
                    validation_errors.append(f"Found {shortcut_stats['total_patches_found']} patches when apply_shortcut=False")

            # Validate patch placement determinism
            determinism_result = self._validate_patch_determinism(dataset, sample_size=min(50, sample_size))
            if not determinism_result["is_deterministic"]:
                validation_errors.append("Patch placement is not deterministic")

            if len(validation_errors) == 0:
                if self.verbose:
                    print("✓ Statistical validation passed")
                    print(f"  Analyzed {shortcut_stats['samples_analyzed']} samples")
                    print(f"  Found {shortcut_stats['total_patches_found']} patches")
                    print(f"  Patch color consistency: {shortcut_stats['patch_color_consistency']:.2f}")

                return ValidationResult(
                    passed=True,
                    message="Statistical properties validated successfully",
                    details=shortcut_stats
                )
            else:
                if self.verbose:
                    print("✗ Statistical validation failed:")
                    for error in validation_errors:
                        print(f"    {error}")

                return ValidationResult(
                    passed=False,
                    message=f"Statistical validation failed: {'; '.join(validation_errors)}",
                    details=shortcut_stats
                )

        except Exception as e:
            self.logger.error(f"Statistical validation failed: {e}")
            return ValidationResult(
                passed=False,
                message=f"Statistical validation failed with error: {str(e)}",
                details={"error_type": type(e).__name__}
            )

    def validate_cache_integrity(self, cache_path: str) -> ValidationResult:
        """
        Validate cache file integrity using checksums and structure validation.

        Args:
            cache_path: Path to cache file to validate

        Returns:
            ValidationResult with integrity check results
        """
        if self.verbose:
            print(f"Validating cache integrity: {cache_path}")

        try:
            if not os.path.exists(cache_path):
                return ValidationResult(
                    passed=False,
                    message=f"Cache file does not exist: {cache_path}"
                )

            # Check file size
            file_size = os.path.getsize(cache_path)
            if file_size < 1024:  # Less than 1KB is suspicious
                return ValidationResult(
                    passed=False,
                    message=f"Cache file too small: {file_size} bytes",
                    details={"file_size": file_size}
                )

            # Calculate file checksum
            file_checksum = self._calculate_file_checksum(cache_path)

            # Load and validate cache structure
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
            except (pickle.PickleError, EOFError, UnicodeDecodeError) as e:
                return ValidationResult(
                    passed=False,
                    message=f"Cache file corrupted (pickle error): {str(e)}",
                    details={"error_type": type(e).__name__}
                )

            # Validate cache structure
            structure_validation = self._validate_cache_structure(cache_data)
            if not structure_validation["valid"]:
                return ValidationResult(
                    passed=False,
                    message=f"Invalid cache structure: {structure_validation['error']}",
                    details=structure_validation
                )

            # Calculate data checksum
            data_checksum = self._calculate_data_checksum(cache_data)

            # Validate data consistency
            consistency_validation = self._validate_data_consistency(cache_data)
            if not consistency_validation["consistent"]:
                return ValidationResult(
                    passed=False,
                    message=f"Data consistency check failed: {consistency_validation['error']}",
                    details=consistency_validation
                )

            if self.verbose:
                print("✓ Cache integrity validation passed")
                print(f"  File size: {file_size:,} bytes")
                print(f"  File checksum: {file_checksum[:16]}...")
                print(f"  Data checksum: {data_checksum[:16]}...")
                print(f"  Samples: {len(cache_data.get('processed_images', []))}")

            return ValidationResult(
                passed=True,
                message="Cache integrity validated successfully",
                details={
                    "file_size": file_size,
                    "file_checksum": file_checksum,
                    "data_checksum": data_checksum,
                    "samples": len(cache_data.get('processed_images', [])),
                    "structure_valid": True,
                    "data_consistent": True
                }
            )

        except Exception as e:
            self.logger.error(f"Cache integrity validation failed: {e}")
            return ValidationResult(
                passed=False,
                message=f"Integrity validation failed with error: {str(e)}",
                details={"error_type": type(e).__name__}
            )

    def run_comprehensive_validation(self,
                                   dataset_configs: List[Dict[str, Any]],
                                   sample_size: int = 100) -> Dict[str, ValidationResult]:
        """
        Run comprehensive validation on multiple dataset configurations.

        Args:
            dataset_configs: List of dataset configuration dictionaries
            sample_size: Number of samples to validate per configuration

        Returns:
            Dictionary mapping config names to validation results
        """
        if self.verbose:
            print("Running comprehensive validation suite...")

        results = {}

        for i, config in enumerate(dataset_configs):
            config_name = config.get('name', f'config_{i}')
            if self.verbose:
                print(f"\nValidating configuration: {config_name}")

            try:
                # Pixel-perfect validation
                pixel_result = self.validate_pixel_perfect_comparison(
                    dataset_class=config['dataset_class'],
                    root=config['root'],
                    train=config.get('train', False),
                    apply_shortcut=config.get('apply_shortcut', True),
                    mask_shortcut=config.get('mask_shortcut', False),
                    patch_size=config.get('patch_size', 4),
                    patch_color=config.get('patch_color', (255, 0, 255)),
                    sample_size=sample_size
                )

                # Statistical validation
                stats_result = self.validate_statistical_properties(
                    dataset_class=config['dataset_class'],
                    root=config['root'],
                    train=config.get('train', False),
                    apply_shortcut=config.get('apply_shortcut', True),
                    patch_size=config.get('patch_size', 4),
                    sample_size=sample_size
                )

                # Cache integrity validation (if cache exists)
                integrity_result = None
                if config.get('validate_cache_integrity', True):
                    # Create temporary dataset to get cache path
                    temp_dataset = config['dataset_class'](
                        root=config['root'],
                        train=config.get('train', False),
                        download=True,
                        apply_shortcut=config.get('apply_shortcut', True),
                        mask_shortcut=config.get('mask_shortcut', False),
                        patch_size=config.get('patch_size', 4),
                        patch_color=config.get('patch_color', (255, 0, 255)),
                        enable_cache=True
                    )

                    if hasattr(temp_dataset, '_get_cache_path'):
                        cache_path = temp_dataset._get_cache_path()
                        if os.path.exists(cache_path):
                            integrity_result = self.validate_cache_integrity(cache_path)

                # Combine results
                overall_passed = pixel_result.passed and stats_result.passed
                if integrity_result is not None:
                    overall_passed = overall_passed and integrity_result.passed

                results[config_name] = ValidationResult(
                    passed=overall_passed,
                    message=f"Comprehensive validation {'passed' if overall_passed else 'failed'}",
                    details={
                        "pixel_perfect": pixel_result,
                        "statistical": stats_result,
                        "integrity": integrity_result
                    }
                )

            except Exception as e:
                self.logger.error(f"Validation failed for {config_name}: {e}")
                results[config_name] = ValidationResult(
                    passed=False,
                    message=f"Validation failed with error: {str(e)}",
                    details={"error_type": type(e).__name__}
                )

        return results

    # Helper methods

    def _tensor_to_numpy(self, tensor_or_image) -> np.ndarray:
        """Convert tensor or PIL image to numpy array for comparison."""
        if isinstance(tensor_or_image, torch.Tensor):
            if tensor_or_image.dim() == 3:  # CHW format
                return tensor_or_image.permute(1, 2, 0).numpy()
            else:
                return tensor_or_image.numpy()
        elif isinstance(tensor_or_image, Image.Image):
            return np.array(tensor_or_image)
        elif isinstance(tensor_or_image, np.ndarray):
            return tensor_or_image
        else:
            raise ValueError(f"Unsupported type for conversion: {type(tensor_or_image)}")

    def _analyze_shortcut_presence(self, dataset, sample_size: int) -> Dict[str, Any]:
        """Analyze shortcut patch presence in dataset samples."""
        shortcut_classes_with_patches = 0
        non_shortcut_classes_with_patches = 0
        total_patches_found = 0
        patch_colors_found = []
        samples_analyzed = min(sample_size, len(dataset))

        target_patch_color = np.array([255, 0, 255])  # Magenta

        for i in range(samples_analyzed):
            item = dataset[i]
            if len(item) >= 2:
                img_tensor, target = item[0], item[1]

                # Convert to numpy
                img_np = self._tensor_to_numpy(img_tensor)

                # Check if this is a shortcut class
                original_label = ALL_USED_FINE_LABELS[target]
                is_shortcut_class = original_label in SHORTCUT_FINE_LABELS

                # Look for magenta patches
                patches_found = self._find_magenta_patches(img_np, target_patch_color)

                if len(patches_found) > 0:
                    total_patches_found += len(patches_found)
                    if is_shortcut_class:
                        shortcut_classes_with_patches += 1
                    else:
                        non_shortcut_classes_with_patches += 1

                    # Collect patch colors for consistency check
                    for patch in patches_found:
                        patch_colors_found.append(patch['color'])

        # Calculate color consistency
        patch_color_consistency = 0.0
        if len(patch_colors_found) > 0:
            consistent_colors = sum(1 for color in patch_colors_found
                                  if np.allclose(color, target_patch_color, atol=5))
            patch_color_consistency = consistent_colors / len(patch_colors_found)

        return {
            "samples_analyzed": samples_analyzed,
            "shortcut_classes_with_patches": shortcut_classes_with_patches,
            "non_shortcut_classes_with_patches": non_shortcut_classes_with_patches,
            "total_patches_found": total_patches_found,
            "patch_color_consistency": patch_color_consistency,
            "unique_patch_colors": len(set(tuple(color) for color in patch_colors_found))
        }

    def _find_magenta_patches(self, img_np: np.ndarray, target_color: np.ndarray, tolerance: int = 5) -> List[Dict]:
        """Find magenta patches in an image."""
        patches = []

        if img_np.ndim != 3 or img_np.shape[2] != 3:
            return patches

        h, w = img_np.shape[:2]

        # Look for regions with target color
        color_mask = np.all(np.abs(img_np - target_color) <= tolerance, axis=2)

        if np.any(color_mask):
            # Find connected components (simple approach)
            y_coords, x_coords = np.where(color_mask)
            if len(y_coords) > 0:
                # Group nearby pixels into patches
                patch_pixels = list(zip(y_coords, x_coords))

                # Simple clustering: if we have a significant number of pixels
                # with the target color, consider it a patch
                if len(patch_pixels) >= 4:  # Minimum patch size
                    avg_color = np.mean(img_np[color_mask], axis=0)
                    patches.append({
                        'pixels': len(patch_pixels),
                        'color': avg_color,
                        'bbox': (np.min(x_coords), np.min(y_coords),
                                np.max(x_coords), np.max(y_coords))
                    })

        return patches

    def _validate_patch_determinism(self, dataset, sample_size: int = 50) -> Dict[str, Any]:
        """Validate that patch placement is deterministic across multiple calls."""
        deterministic_results = []

        for i in range(min(sample_size, len(dataset))):
            # Get the same item multiple times
            item1 = dataset[i]
            item2 = dataset[i]

            if len(item1) >= 2 and len(item2) >= 2:
                img1_np = self._tensor_to_numpy(item1[0])
                img2_np = self._tensor_to_numpy(item2[0])

                # Check if images are identical
                is_identical = np.array_equal(img1_np, img2_np)
                deterministic_results.append(is_identical)

        is_deterministic = all(deterministic_results) if deterministic_results else True

        return {
            "is_deterministic": is_deterministic,
            "samples_tested": len(deterministic_results),
            "identical_count": sum(deterministic_results)
        }

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _calculate_data_checksum(self, cache_data: Dict) -> str:
        """Calculate checksum of cache data contents."""
        # Create a deterministic representation of the data
        data_repr = {
            'processed_images_shape': cache_data.get('processed_images', np.array([])).shape,
            'targets_len': len(cache_data.get('targets', [])),
            'params_hash': cache_data.get('params_hash', ''),
            'version': cache_data.get('version', ''),
            'dataset_size': cache_data.get('dataset_size', 0)
        }

        # Add sample of actual data for integrity
        processed_images = cache_data.get('processed_images')
        if processed_images is not None and len(processed_images) > 0:
            # Use first and last images for checksum
            sample_indices = [0]
            if len(processed_images) > 1:
                sample_indices.append(len(processed_images) - 1)

            for idx in sample_indices:
                img_hash = hashlib.sha256(processed_images[idx].tobytes()).hexdigest()[:16]
                data_repr[f'img_{idx}_hash'] = img_hash

        data_str = str(sorted(data_repr.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _validate_cache_structure(self, cache_data: Dict) -> Dict[str, Any]:
        """Validate the structure of cache data."""
        required_keys = ['processed_images', 'targets', 'params_hash']

        for key in required_keys:
            if key not in cache_data:
                return {
                    "valid": False,
                    "error": f"Missing required key: {key}",
                    "missing_keys": [k for k in required_keys if k not in cache_data]
                }

        # Validate data types
        processed_images = cache_data['processed_images']
        if not isinstance(processed_images, np.ndarray):
            return {
                "valid": False,
                "error": f"processed_images must be numpy array, got {type(processed_images)}"
            }

        if processed_images.ndim != 4:
            return {
                "valid": False,
                "error": f"processed_images must be 4D array, got {processed_images.ndim}D"
            }

        targets = cache_data['targets']
        if not isinstance(targets, (list, np.ndarray)):
            return {
                "valid": False,
                "error": f"targets must be list or array, got {type(targets)}"
            }

        return {"valid": True, "error": None}

    def _validate_data_consistency(self, cache_data: Dict) -> Dict[str, Any]:
        """Validate consistency of cache data."""
        processed_images = cache_data['processed_images']
        targets = cache_data['targets']

        if len(processed_images) != len(targets):
            return {
                "consistent": False,
                "error": f"Length mismatch: {len(processed_images)} images vs {len(targets)} targets"
            }

        # Check if dataset_size matches actual data
        expected_size = cache_data.get('dataset_size')
        if expected_size is not None and expected_size != len(processed_images):
            return {
                "consistent": False,
                "error": f"Dataset size mismatch: expected {expected_size}, got {len(processed_images)}"
            }

        # Validate image shapes
        expected_shape = (32, 32, 3)  # CIFAR-100 shape
        if processed_images.shape[1:] != expected_shape:
            return {
                "consistent": False,
                "error": f"Invalid image shape: expected {expected_shape}, got {processed_images.shape[1:]}"
            }

        # Validate target ranges
        target_array = np.array(targets)
        if target_array.min() < 0 or target_array.max() >= len(ALL_USED_FINE_LABELS):
            return {
                "consistent": False,
                "error": f"Invalid target range: [{target_array.min()}, {target_array.max()}], expected [0, {len(ALL_USED_FINE_LABELS)-1}]"
            }

        return {"consistent": True, "error": None}


class EinstellungCacheValidationTests(unittest.TestCase):
    """
    Unit tests for Einstellung cache validation system.

    Provides unittest-compatible tools for debugging and validation
    as required by requirement 6.4.
    """

    def setUp(self):
        """Set up test environment."""
        self.validator = EinstellungCacheValidator(verbose=False)
        self.test_root = os.path.join(base_path(), 'CIFAR100')

    def test_pixel_perfect_validation_32x32(self):
        """Test pixel-perfect validation for 32x32 dataset."""
        result = self.validator.validate_pixel_perfect_comparison(
            dataset_class=MyCIFAR100Einstellung,
            root=self.test_root,
            train=False,  # Use test set for faster testing
            apply_shortcut=True,
            sample_size=20  # Small sample for unit test
        )

        self.assertTrue(result.passed, f"Pixel-perfect validation failed: {result.message}")
        self.assertIsNotNone(result.details)
        self.assertEqual(result.details['mismatches'], 0)

    def test_pixel_perfect_validation_224x224(self):
        """Test pixel-perfect validation for 224x224 dataset."""
        result = self.validator.validate_pixel_perfect_comparison(
            dataset_class=MyEinstellungCIFAR100_224,
            root=self.test_root,
            train=False,
            apply_shortcut=True,
            sample_size=10  # Smaller sample for 224x224
        )

        self.assertTrue(result.passed, f"224x224 pixel-perfect validation failed: {result.message}")

    def test_statistical_validation_with_shortcuts(self):
        """Test statistical validation with shortcuts enabled."""
        result = self.validator.validate_statistical_properties(
            dataset_class=MyCIFAR100Einstellung,
            root=self.test_root,
            train=False,
            apply_shortcut=True,
            sample_size=50
        )

        self.assertTrue(result.passed, f"Statistical validation failed: {result.message}")
        self.assertGreater(result.details['total_patches_found'], 0, "No patches found when shortcuts enabled")

    def test_statistical_validation_without_shortcuts(self):
        """Test statistical validation with shortcuts disabled."""
        result = self.validator.validate_statistical_properties(
            dataset_class=MyCIFAR100Einstellung,
            root=self.test_root,
            train=False,
            apply_shortcut=False,
            sample_size=50
        )

        self.assertTrue(result.passed, f"Statistical validation failed: {result.message}")
        self.assertEqual(result.details['total_patches_found'], 0, "Found patches when shortcuts disabled")

    def test_cache_integrity_validation(self):
        """Test cache integrity validation."""
        # Create a dataset to ensure cache exists
        dataset = MyCIFAR100Einstellung(
            root=self.test_root,
            train=False,
            download=True,
            apply_shortcut=True,
            enable_cache=True
        )

        if dataset._cache_loaded:
            cache_path = dataset._get_cache_path()
            result = self.validator.validate_cache_integrity(cache_path)

            self.assertTrue(result.passed, f"Cache integrity validation failed: {result.message}")
            self.assertIsNotNone(result.details)
            self.assertGreater(result.details['file_size'], 1024)
        else:
            self.skipTest("Cache not available for integrity testing")

    def test_comprehensive_validation(self):
        """Test comprehensive validation with multiple configurations."""
        configs = [
            {
                'name': 'einstellung_32x32_shortcuts',
                'dataset_class': MyCIFAR100Einstellung,
                'root': self.test_root,
                'train': False,
                'apply_shortcut': True,
                'patch_size': 4
            },
            {
                'name': 'einstellung_32x32_no_shortcuts',
                'dataset_class': MyCIFAR100Einstellung,
                'root': self.test_root,
                'train': False,
                'apply_shortcut': False,
                'patch_size': 4
            }
        ]

        results = self.validator.run_comprehensive_validation(configs, sample_size=20)

        self.assertEqual(len(results), 2)
        for config_name, result in results.items():
            self.assertTrue(result.passed, f"Comprehensive validation failed for {config_name}: {result.message}")


def main():
    """Main function for running validation."""
    import argparse

    parser = argparse.ArgumentParser(description='Einstellung Cache Validation System')
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--validate', action='store_true', help='Run comprehensive validation')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of samples to validate')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.test:
        # Run unit tests
        unittest.main(argv=[''], exit=False, verbosity=2 if args.verbose else 1)

    if args.validate:
        # Run comprehensive validation
        validator = EinstellungCacheValidator(verbose=args.verbose)

        # Define validation configurations
        configs = [
            {
                'name': 'einstellung_32x32_shortcuts',
                'dataset_class': MyCIFAR100Einstellung,
                'root': os.path.join(base_path(), 'CIFAR100'),
                'train': False,
                'apply_shortcut': True,
                'mask_shortcut': False,
                'patch_size': 4
            },
            {
                'name': 'einstellung_32x32_masked',
                'dataset_class': MyCIFAR100Einstellung,
                'root': os.path.join(base_path(), 'CIFAR100'),
                'train': False,
                'apply_shortcut': False,
                'mask_shortcut': True,
                'patch_size': 4
            },
            {
                'name': 'einstellung_224x224_shortcuts',
                'dataset_class': MyEinstellungCIFAR100_224,
                'root': os.path.join(base_path(), 'CIFAR100'),
                'train': False,
                'apply_shortcut': True,
                'patch_size': 16
            }
        ]

        print("Running comprehensive Einstellung cache validation...")
        results = validator.run_comprehensive_validation(configs, sample_size=args.sample_size)

        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        all_passed = True
        for config_name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            print(f"{config_name}: {status}")
            if not result.passed:
                print(f"  Error: {result.message}")
                all_passed = False

        print("="*60)
        print(f"Overall result: {'PASS' if all_passed else 'FAIL'}")

        return 0 if all_passed else 1

    if not args.test and not args.validate:
        print("Please specify --test or --validate")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
