"""
ERITimelineDataset - Core data structure for ERI timeline data.

This module provides the ERITimelineDataset class for storing and manipulating
timeline data from Einstellung experiments with filtering, alignment, and export
capabilities.

Integrates with the existing Mammoth Einstellung experiment infrastructure:
- Compatible with utils/einstellung_evaluator.py data structure
- Supports datasets/seq_cifar100_einstellung_224.py evaluation splits
- Works with existing experiment runners and checkpoint management
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import interpolate


@dataclass
class ERITimelineDataset:
    """
    Core data structure for ERI timeline data.

    This class holds timeline data from Einstellung experiments with methods
    for filtering, alignment, and export. Provides comprehensive functionality
    for manipulating timeline data across methods, seeds, and evaluation splits.

    Attributes:
        data: DataFrame with columns [method, seed, epoch_eff, split, acc]
        metadata: Dictionary containing dataset metadata and provenance
        methods: List of unique method names in the dataset
        splits: List of unique split names in the dataset
        seeds: List of unique seed values in the dataset
        epoch_range: Tuple of (min_epoch, max_epoch) across all data
    """
    data: pd.DataFrame
    metadata: Dict[str, Any]
    methods: List[str]
    splits: List[str]
    seeds: List[int]
    epoch_range: Tuple[float, float]

    def __post_init__(self):
        """Initialize logger after dataclass creation."""
        self.logger = logging.getLogger(__name__)

    def get_method_data(self, method: str) -> pd.DataFrame:
        """
        Get data for a specific method.

        Args:
            method: Method name to filter by

        Returns:
            DataFrame containing only data for the specified method

        Raises:
            ValueError: If method not found in dataset
        """
        if method not in self.methods:
            raise ValueError(f"Method '{method}' not found. Available methods: {self.methods}")

        return self.data[self.data['method'] == method].copy()

    def get_split_data(self, split: str) -> pd.DataFrame:
        """
        Get data for a specific split.

        Args:
            split: Split name to filter by

        Returns:
            DataFrame containing only data for the specified split

        Raises:
            ValueError: If split not found in dataset
        """
        if split not in self.splits:
            raise ValueError(f"Split '{split}' not found. Available splits: {self.splits}")

        return self.data[self.data['split'] == split].copy()

    def get_seed_data(self, seed: int) -> pd.DataFrame:
        """
        Get data for a specific seed.

        Args:
            seed: Seed value to filter by

        Returns:
            DataFrame containing only data for the specified seed

        Raises:
            ValueError: If seed not found in dataset
        """
        if seed not in self.seeds:
            raise ValueError(f"Seed {seed} not found. Available seeds: {self.seeds}")

        return self.data[self.data['seed'] == seed].copy()

    def filter_data(self,
                   methods: Optional[List[str]] = None,
                   seeds: Optional[List[int]] = None,
                   splits: Optional[List[str]] = None) -> "ERITimelineDataset":
        """
        Filter dataset by multiple criteria.

        Args:
            methods: List of methods to include (None = all methods)
            seeds: List of seeds to include (None = all seeds)
            splits: List of splits to include (None = all splits)

        Returns:
            New ERITimelineDataset with filtered data

        Raises:
            ValueError: If any spefilter values are not found
        """
        # Validate filter values
        if methods is not None:
            invalid_methods = set(methods) - set(self.methods)
            if invalid_methods:
                raise ValueError(f"Invalid methods: {invalid_methods}. Available: {self.methods}")

        if seeds is not None:
            invalid_seeds = set(seeds) - set(self.seeds)
            if invalid_seeds:
                raise ValueError(f"Invalid seeds: {invalid_seeds}. Available: {self.seeds}")

        if splits is not None:
            invalid_splits = set(splits) - set(self.splits)
            if invalid_splits:
                raise ValueError(f"Invalid splits: {invalid_splits}. Available: {self.splits}")

        # Apply filters
        filtered_data = self.data.copy()

        if methods is not None:
            filtered_data = filtered_data[filtered_data['method'].isin(methods)]

        if seeds is not None:
            filtered_data = filtered_data[filtered_data['seed'].isin(seeds)]

        if splits is not None:
            filtered_data = filtered_data[filtered_data['split'].isin(splits)]

        if len(filtered_data) == 0:
            raise ValueError("Filtering resulted in empty dataset")

        # Create new dataset with filtered data
        return self._create_filtered_dataset(filtered_data, {
            'filter_methods': methods,
            'filter_seeds': seeds,
            'filter_splits': splits
        })

    def align_epochs(self, common_epochs: np.ndarray) -> "ERITimelineDataset":
        """
        Align dataset to common epoch grid using interpolation.

        This method handles uneven epoch grids by interpolating accuracy values
        onto a common set of epochs. Essential for cross-method comparisons
        where different methods may have different evaluation frequencies.

        Args:
            common_epochs: Array of epochs to align all data to

        Returns:
            New ERITimelineDataset with data aligned to common epoch grid

        Raises:
            ValueError: If common_epochs is empty or contains invalid values
            RuntimeError: If interpolation fails for any method-seed-split combination
        """
        if len(common_epochs) == 0:
            raise ValueError("common_epochs cannot be empty")

        if not np.all(np.diff(common_epochs) >= 0):
            raise ValueError("common_epochs must be monotonically non-decreasing")

        # Allow negative epochs for extrapolation testing, but warn
        if np.any(common_epochs < 0):
            self.logger.warning("Negative epochs detected - this may indicate extrapolation beyond training data")

        aligned_rows = []
        interpolation_stats = {
            'total_groups': 0,
            'successful_interpolations': 0,
            'failed_interpolations': 0,
            'extrapolation_warnings': 0
        }

        # Group by method, seed, split for interpolation
        for (method, seed, split), group in self.data.groupby(['method', 'seed', 'split']):
            interpolation_stats['total_groups'] += 1

            try:
                # Sort by epoch for interpolation
                group_sorted = group.sort_values('epoch_eff')
                original_epochs = group_sorted['epoch_eff'].values
                original_accs = group_sorted['acc'].values

                # Check for duplicate epochs
                if len(np.unique(original_epochs)) != len(original_epochs):
                    self.logger.warning(
                        f"Duplicate epochs found for {method}, seed {seed}, split {split}. "
                        "Using mean accuracy for duplicates."
                    )
                    # Handle duplicates by taking mean
                    df_temp = pd.DataFrame({
                        'epoch_eff': original_epochs,
                        'acc': original_accs
                    })
                    df_temp = df_temp.groupby('epoch_eff')['acc'].mean().reset_index()
                    original_epochs = df_temp['epoch_eff'].values
                    original_accs = df_temp['acc'].values

                # Check if we need extrapolation
                min_epoch, max_epoch = original_epochs.min(), original_epochs.max()
                extrapolation_needed = (
                    common_epochs.min() < min_epoch or
                    common_epochs.max() > max_epoch
                )

                if extrapolation_needed:
                    interpolation_stats['extrapolation_warnings'] += 1
                    self.logger.warning(
                        f"Extrapolation needed for {method}, seed {seed}, split {split}. "
                        f"Original range: [{min_epoch:.3f}, {max_epoch:.3f}], "
                        f"Target range: [{common_epochs.min():.3f}, {common_epochs.max():.3f}]"
                    )

                # Perform interpolation
                if len(original_epochs) == 1:
                    # Single point - use constant extrapolation
                    interpolated_accs = np.full_like(common_epochs, original_accs[0])
                elif len(original_epochs) == 2:
                    # Two points - use linear interpolation/extrapolation
                    interpolated_accs = np.interp(common_epochs, original_epochs, original_accs)
                else:
                    # Multiple points - use cubic spline with linear extrapolation
                    # Clamp to avoid extrapolation issues
                    epochs_to_interpolate = np.clip(common_epochs, min_epoch, max_epoch)

                    try:
                        # Try cubic spline first
                        spline = interpolate.CubicSpline(original_epochs, original_accs,
                                                       bc_type='natural', extrapolate=False)
                        interpolated_accs = spline(epochs_to_interpolate)

                        # Handle extrapolation with linear interpolation
                        if extrapolation_needed:
                            linear_interp = np.interp(common_epochs, original_epochs, original_accs)
                            # Use spline for interpolation, linear for extrapolation
                            mask_interp = (common_epochs >= min_epoch) & (common_epochs <= max_epoch)
                            interpolated_accs = linear_interp.copy()
                            interpolated_accs[mask_interp] = spline(common_epochs[mask_interp])

                    except Exception as e:
                        self.logger.warning(
                            f"Cubic spline failed for {method}, seed {seed}, split {split}: {e}. "
                            "Falling back to linear interpolation."
                        )
                        interpolated_accs = np.interp(common_epochs, original_epochs, original_accs)

                # Clamp accuracies to valid range [0, 1]
                interpolated_accs = np.clip(interpolated_accs, 0.0, 1.0)

                # Create aligned rows
                for epoch, acc in zip(common_epochs, interpolated_accs):
                    aligned_rows.append({
                        'method': method,
                        'seed': seed,
                        'epoch_eff': float(epoch),
                        'split': split,
                        'acc': float(acc)
                    })

                interpolation_stats['successful_interpolations'] += 1

            except Exception as e:
                interpolation_stats['failed_interpolations'] += 1
                self.logger.error(
                    f"Failed to interpolate {method}, seed {seed}, split {split}: {e}"
                )
                # Skip this group rather than failing completely
                continue

        if not aligned_rows:
            raise RuntimeError("All interpolations failed - no aligned data generated")

        # Create aligned DataFrame
        aligned_df = pd.DataFrame(aligned_rows)

        # Update metadata with alignment information
        alignment_metadata = {
            'aligned': True,
            'common_epochs': common_epochs.tolist(),
            'alignment_stats': interpolation_stats,
            'original_epoch_range': self.epoch_range,
            'aligned_epoch_range': (float(common_epochs.min()), float(common_epochs.max()))
        }

        return self._create_filtered_dataset(aligned_df, alignment_metadata)

    def export_csv(self, filepath: Union[str, Path],
                   include_metadata: bool = True) -> None:
        """
        Export dataset to CSV with deterministic ordering.

        Creates a CSV file with deterministic row ordering and optionally
        generates a metadata sidecar JSON file for reproducibility.

        Args:
            filepath: Path for output CSV file
            include_metadata: Whether to create metadata sidecar JSON file

        Raises:
            IOError: If file writing fails
        """
        filepath = Path(filepath)

        try:
            # Create output directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Sort data deterministically for reproducible output
            sorted_data = self.data.sort_values([
                'method', 'seed', 'split', 'epoch_eff'
            ]).reset_index(drop=True)

            # Export CSV
            sorted_data.to_csv(filepath, index=False, float_format='%.6f')

            self.logger.info(f"Exported {len(sorted_data)} rows to {filepath}")

            # Export metadata sidecar if requested
            if include_metadata:
                metadata_path = filepath.with_suffix('.json')
                self._export_metadata_sidecar(metadata_path, sorted_data)

        except Exception as e:
            raise IOError(f"Failed to export CSV to {filepath}: {e}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset.

        Returns:
            Dictionary containing dataset summary statistics
        """
        stats = {
            'n_rows': len(self.data),
            'n_methods': len(self.methods),
            'n_seeds': len(self.seeds),
            'n_splits': len(self.splits),
            'methods': self.methods,
            'seeds': self.seeds,
            'splits': self.splits,
            'epoch_range': self.epoch_range,
            'accuracy_range': (float(self.data['acc'].min()), float(self.data['acc'].max())),
            'epochs_per_method': {},
            'seeds_per_method': {},
            'completeness': {}
        }

        # Per-method statistics
        for method in self.methods:
            method_data = self.get_method_data(method)
            stats['epochs_per_method'][method] = len(method_data['epoch_eff'].unique())
            stats['seeds_per_method'][method] = len(method_data['seed'].unique())

        # Data completeness analysis
        expected_combinations = len(self.methods) * len(self.seeds) * len(self.splits)
        actual_combinations = len(self.data.groupby(['method', 'seed', 'split']))
        stats['completeness']['expected_combinations'] = expected_combinations
        stats['completeness']['actual_combinations'] = actual_combinations
        stats['completeness']['completeness_ratio'] = actual_combinations / expected_combinations

        return stats

    def _create_filtered_dataset(self, filtered_data: pd.DataFrame,
                               additional_metadata: Dict[str, Any]) -> "ERITimelineDataset":
        """Create new dataset from filtered data."""
        # Extract unique values from filtered data
        methods = sorted(filtered_data['method'].unique())
        splits = sorted(filtered_data['split'].unique())
        seeds = sorted(filtered_data['seed'].unique())

        # Calculate epoch range
        epoch_min = filtered_data['epoch_eff'].min()
        epoch_max = filtered_data['epoch_eff'].max()
        epoch_range = (float(epoch_min), float(epoch_max))

        # Create new metadata
        new_metadata = self.metadata.copy()
        new_metadata.update(additional_metadata)
        new_metadata.update({
            'filtered_n_methods': len(methods),
            'filtered_n_splits': len(splits),
            'filtered_n_seeds': len(seeds),
            'filtered_n_rows': len(filtered_data),
            'filtered_epoch_range': epoch_range
        })

        return ERITimelineDataset(
            data=filtered_data.copy(),
            metadata=new_metadata,
            methods=methods,
            splits=splits,
            seeds=seeds,
            epoch_range=epoch_range
        )

    def _export_metadata_sidecar(self, metadata_path: Path,
                                sorted_data: pd.DataFrame) -> None:
        """Export metadata sidecar JSON file."""
        try:
            # Create comprehensive metadata
            export_metadata = {
                'export_info': {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'csv_file': metadata_path.with_suffix('.csv').name,
                    'n_rows_exported': len(sorted_data),
                    'columns': list(sorted_data.columns)
                },
                'dataset_summary': self.get_summary_stats(),
                'original_metadata': self.metadata,
                'data_hash': self._compute_data_hash(sorted_data)
            }

            # Write metadata JSON
            with open(metadata_path, 'w') as f:
                json.dump(export_metadata, f, indent=2, default=str)

            self.logger.info(f"Exported metadata sidecar to {metadata_path}")

        except Exception as e:
            self.logger.warning(f"Failed to export metadata sidecar: {e}")

    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of data for reproducibility verification."""
        import hashlib

        # Create deterministic string representation
        data_str = data.to_csv(index=False, float_format='%.6f')

        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(data_str.encode('utf-8'))
        return hash_obj.hexdigest()

    def __len__(self) -> int:
        """Return number of rows in dataset."""
        return len(self.data)

    def __repr__(self) -> str:
        """Return string representation of dataset."""
        return (
            f"ERITimelineDataset("
            f"n_rows={len(self.data)}, "
            f"methods={len(self.methods)}, "
            f"seeds={len(self.seeds)}, "
            f"splits={len(self.splits)}, "
            f"epoch_range={self.epoch_range})"
        )
