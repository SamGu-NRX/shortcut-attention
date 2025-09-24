"""
ERIDataLoader - Data loading and validation for ERI visualization system.

This module provides comprehensive data loading, validation, and conversion
functionality for Einstellung Rigidity Index (ERI) experiments.

Integrates with the existing Mammoth Einstellung experiment infrastructure:
- Compatible with utils/einstellung_evaluator.py export format
- Supports datasets/seq_cifar100_einstellung_224.py data structure
- Works with existing experiment runners and checkpoint management
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from .dataset import ERITimelineDataset


class ERIDataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class ERIDataLoader:
    """
    Data loader and validator for ERI visualization system.

    Handles CSV loading, validation, and conversion from various data formats
    including EinstellungEvaluator exports.
    """

    # Required columns for CSV format
    REQUIRED_COLS = ["method", "seed", "epoch_eff", "split", "acc", "top5", "loss"]

    # Valid split names
    VALID_SPLITS = {
        "T1_all",
        "T2_shortcut_normal",
        "T2_shortcut_masked",
        "T2_nonshortcut_normal"
    }

    def __init__(self):
        """Initialize the data loader."""
        self.logger = logging.getLogger(__name__)

    def load_csv(self, filepath: Union[str, Path]) -> ERITimelineDataset:
        """
        Load and validate CSV data.

        Args:
            filepath: Path to CSV file

        Returns:
            ERITimelineDataset with validated data

        Raises:
            ERIDataValidationError: If validation fails
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        try:
            # Load CSV with proper data types
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded CSV with {len(df)} rows from {filepath}")

        except Exception as e:
            raise ERIDataValidationError(f"Failed to read CSV file {filepath}: {e}")

        # Validate format
        self.validate_format(df)

        # Create dataset
        return self._create_dataset_from_dataframe(df, {"source": str(filepath)})

    def load_from_csv(self, filepath: Union[str, Path]) -> ERITimelineDataset:
        """Backward-compatible alias for load_csv."""
        return self.load_csv(filepath)

    def load_from_evaluator_export(self, export: Dict[str, Any]) -> ERITimelineDataset:
        """
        Convert EinstellungEvaluator export to ERITimelineDataset.

        Compatible with utils/einstellung_evaluator.py export format from the
        existing Mammoth Einstellung experiment infrastructure.

        Args:
            export: Dictionary from EinstellungEvaluator.export_results()

        Returns:
            ERITimelineDataset with converted data

        Raises:
            ERIDataValidationError: If conversion fails
        """
        try:
            # Extract timeline data
            if 'timeline_data' not in export:
                raise ERIDataValidationError("Export missing 'timeline_data' field")

            timeline_data = export['timeline_data']
            if not timeline_data:
                raise ERIDataValidationError("Timeline data is empty")

            # Convert to DataFrame format
            rows = []

            for entry in timeline_data:
                epoch = entry.get('epoch', 0)
                task_id = entry.get('task_id', 0)
                subset_metrics = entry.get('subset_metrics') or entry.get('subset_accuracies', {})
                subset_losses = entry.get('subset_losses', {})

                # Extract method from configuration or use default
                method = self._extract_method_from_export(export)

                # Extract seed from configuration or use default
                seed = self._extract_seed_from_export(export)

                # Convert effective epoch (assume Phase 2 epochs map directly)
                epoch_eff = float(epoch)

                # Create rows for each split
                for split, value in subset_metrics.items():
                    if split in self.VALID_SPLITS:
                        if isinstance(value, dict):
                            top1 = float(value.get('top1', value.get('accuracy', 0.0)))
                            top5 = value.get('top5')
                            top5 = float(top5) if top5 is not None else top1
                        else:
                            top1 = float(value)
                            top5 = top1

                        loss = subset_losses.get(split)
                        loss = float(loss) if loss is not None else max(0.0, 1.0 - top1)

                        rows.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch_eff,
                            'split': split,
                            'acc': top1,
                            'top5': top5,
                            'loss': loss
                        })

            if not rows:
                raise ERIDataValidationError("No valid data rows found in export")

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Validate the converted data
            self.validate_format(df)

            # Create metadata from export
            metadata = {
                'source': 'evaluator_export',
                'configuration': export.get('configuration', {}),
                'conversion_timestamp': pd.Timestamp.now().isoformat()
            }

            return self._create_dataset_from_dataframe(df, metadata)

        except Exception as e:
            if isinstance(e, ERIDataValidationError):
                raise
            raise ERIDataValidationError(f"Failed to convert evaluator export: {e}")

    def load_from_mammoth_results(self, results_dir: Union[str, Path]) -> List[ERITimelineDataset]:
        """
        Load data from Mammoth experiment results directory.

        Scans for JSON result files from the existing Mammoth Einstellung
        experiment infrastructure and converts them to ERITimelineDataset objects.

        Args:
            results_dir: Directory containing experiment result files

        Returns:
            List of ERITimelineDataset objects, one per experiment

        Raises:
            ERIDataValidationError: If no valid results found
        """
        results_dir = Path(results_dir)

        if not results_dir.exists():
            raise ERIDataValidationError(f"Results directory not found: {results_dir}")

        datasets = []

        # Look for JSON result files
        json_files = list(results_dir.glob("**/*results*.json"))
        json_files.extend(list(results_dir.glob("**/*einstellung*.json")))

        if not json_files:
            raise ERIDataValidationError(f"No result files found in {results_dir}")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    export_data = json.load(f)

                # Convert to dataset
                dataset = self.load_from_evaluator_export(export_data)

                # Add file source to metadata
                dataset.metadata['source_file'] = str(json_file)
                datasets.append(dataset)

                self.logger.info(f"Loaded dataset from {json_file}")

            except Exception as e:
                self.logger.warning(f"Could not load {json_file}: {e}")
                continue

        if not datasets:
            raise ERIDataValidationError(f"No valid datasets found in {results_dir}")

        return datasets

    def validate_format(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame format and contents.

        Args:
            df: DataFrame to validate

        Raises:
            ERIDataValidationError: If validation fails with detailed context
        """
        # Check required columns
        missing_cols = set(self.REQUIRED_COLS) - set(df.columns)
        if missing_cols:
            raise ERIDataValidationError(
                f"Missing required columns: {missing_cols}. "
                f"Required: {self.REQUIRED_COLS}, Found: {list(df.columns)}"
            )

        # Check for empty DataFrame
        if len(df) == 0:
            raise ERIDataValidationError("DataFrame is empty")

        # Validate data types and ranges
        self._validate_column_types(df)
        self._validate_column_domains(df)
        self._validate_splits(df)
        self._validate_data_consistency(df)

    def convert_legacy_format(self, legacy: Dict[str, Any]) -> ERITimelineDataset:
        """
        Convert legacy data format to current schema.

        Args:
            legacy: Legacy format dictionary

        Returns:
            ERITimelineDataset with converted data

        Raises:
            ERIDataValidationError: If conversion fails
        """
        try:
            # Handle different legacy formats
            if 'accuracy_curves' in legacy:
                return self._convert_accuracy_curves_format(legacy)
            elif 'results' in legacy:
                return self._convert_results_format(legacy)
            else:
                raise ERIDataValidationError(
                    "Unknown legacy format. Expected 'accuracy_curves' or 'results' field."
                )
        except Exception as e:
            if isinstance(e, ERIDataValidationError):
                raise
            raise ERIDataValidationError(f"Failed to convert legacy format: {e}")

    def _create_dataset_from_dataframe(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> ERITimelineDataset:
        """Create ERITimelineDataset from validated DataFrame."""
        # Extract unique values
        methods = sorted(df['method'].unique())
        splits = sorted(df['split'].unique())
        seeds = sorted(df['seed'].unique())

        # Calculate epoch range
        epoch_min = df['epoch_eff'].min()
        epoch_max = df['epoch_eff'].max()
        epoch_range = (float(epoch_min), float(epoch_max))

        # Add derived metadata
        metadata.update({
            'n_methods': len(methods),
            'n_splits': len(splits),
            'n_seeds': len(seeds),
            'n_rows': len(df),
            'epoch_range': epoch_range
        })

        return ERITimelineDataset(
            data=df.copy(),
            metadata=metadata,
            methods=methods,
            splits=splits,
            seeds=seeds,
            epoch_range=epoch_range
        )

    def _validate_column_types(self, df: pd.DataFrame) -> None:
        """Validate column data types."""
        errors = []

        # Check method column (string)
        if not pd.api.types.is_string_dtype(df['method']):
            # Try to convert
            try:
                df['method'] = df['method'].astype(str)
            except Exception:
                errors.append("Column 'method' must be string type")

        # Check seed column (integer)
        if not pd.api.types.is_integer_dtype(df['seed']):
            try:
                df['seed'] = pd.to_numeric(df['seed'], errors='coerce').astype('Int64')
                if df['seed'].isna().any():
                    invalid_rows = df[df['seed'].isna()].index.tolist()
                    errors.append(f"Column 'seed' contains non-integer values at rows: {invalid_rows[:5]}")
            except Exception:
                errors.append("Column 'seed' must be integer type")

        # Check epoch_eff column (numeric)
        if not pd.api.types.is_numeric_dtype(df['epoch_eff']):
            try:
                df['epoch_eff'] = pd.to_numeric(df['epoch_eff'], errors='coerce')
                if df['epoch_eff'].isna().any():
                    invalid_rows = df[df['epoch_eff'].isna()].index.tolist()
                    errors.append(f"Column 'epoch_eff' contains non-numeric values at rows: {invalid_rows[:5]}")
            except Exception:
                errors.append("Column 'epoch_eff' must be numeric type")

        # Check split column (string)
        if not pd.api.types.is_string_dtype(df['split']):
            try:
                df['split'] = df['split'].astype(str)
            except Exception:
                errors.append("Column 'split' must be string type")

        # Check acc column (numeric)
        if not pd.api.types.is_numeric_dtype(df['acc']):
            try:
                df['acc'] = pd.to_numeric(df['acc'], errors='coerce')
                if df['acc'].isna().any():
                    invalid_rows = df[df['acc'].isna()].index.tolist()
                    errors.append(f"Column 'acc' contains non-numeric values at rows: {invalid_rows[:5]}")
            except Exception:
                errors.append("Column 'acc' must be numeric type")

        # Check top5 column (numeric)
        if 'top5' in df.columns and not pd.api.types.is_numeric_dtype(df['top5']):
            try:
                df['top5'] = pd.to_numeric(df['top5'], errors='coerce')
                if df['top5'].isna().any():
                    invalid_rows = df[df['top5'].isna()].index.tolist()
                    errors.append(f"Column 'top5' contains non-numeric values at rows: {invalid_rows[:5]}")
            except Exception:
                errors.append("Column 'top5' must be numeric type")

        # Check loss column (numeric)
        if 'loss' in df.columns and not pd.api.types.is_numeric_dtype(df['loss']):
            try:
                df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
                if df['loss'].isna().any():
                    invalid_rows = df[df['loss'].isna()].index.tolist()
                    errors.append(f"Column 'loss' contains non-numeric values at rows: {invalid_rows[:5]}")
            except Exception:
                errors.append("Column 'loss' must be numeric type")

        if errors:
            raise ERIDataValidationError("Data type validation failed:\n" + "\n".join(errors))

    def _validate_column_domains(self, df: pd.DataFrame) -> None:
        """Validate column value domains."""
        errors = []

        # Validate seed values (positive integers)
        if (df['seed'] < 0).any():
            invalid_rows = df[df['seed'] < 0].index.tolist()
            invalid_values = df.loc[invalid_rows, 'seed'].tolist()
            errors.append(f"Seed values must be non-negative. Invalid at rows {invalid_rows[:5]}: {invalid_values[:5]}")

        # Validate epoch_eff values (non-negative)
        if (df['epoch_eff'] < 0).any():
            invalid_rows = df[df['epoch_eff'] < 0].index.tolist()
            invalid_values = df.loc[invalid_rows, 'epoch_eff'].tolist()
            errors.append(f"Epoch_eff values must be non-negative. Invalid at rows {invalid_rows[:5]}: {invalid_values[:5]}")

        # Validate accuracy values (0 to 1)
        acc_invalid = (df['acc'] < 0) | (df['acc'] > 1)
        if acc_invalid.any():
            invalid_rows = df[acc_invalid].index.tolist()
            invalid_values = df.loc[invalid_rows, 'acc'].tolist()
            errors.append(f"Accuracy values must be in [0, 1]. Invalid at rows {invalid_rows[:5]}: {invalid_values[:5]}")

        if 'top5' in df.columns:
            top5_invalid = (df['top5'] < 0) | (df['top5'] > 1)
            if top5_invalid.any():
                invalid_rows = df[top5_invalid].index.tolist()
                invalid_values = df.loc[invalid_rows, 'top5'].tolist()
                errors.append(f"Top-5 accuracy values must be in [0, 1]. Invalid at rows {invalid_rows[:5]}: {invalid_values[:5]}")

        if 'loss' in df.columns:
            loss_invalid = df['loss'] < 0
            if loss_invalid.any():
                invalid_rows = df[loss_invalid].index.tolist()
                invalid_values = df.loc[invalid_rows, 'loss'].tolist()
                errors.append(f"Loss values must be non-negative. Invalid at rows {invalid_rows[:5]}: {invalid_values[:5]}")

        # Validate method names (non-empty strings)
        empty_methods = df['method'].str.strip() == ''
        if empty_methods.any():
            invalid_rows = df[empty_methods].index.tolist()
            errors.append(f"Method names cannot be empty. Invalid at rows: {invalid_rows[:5]}")

        if errors:
            raise ERIDataValidationError("Domain validation failed:\n" + "\n".join(errors))

    def _validate_splits(self, df: pd.DataFrame) -> None:
        """Validate split names."""
        invalid_splits = set(df['split'].unique()) - self.VALID_SPLITS
        if invalid_splits:
            # Find rows with invalid splits
            invalid_mask = df['split'].isin(invalid_splits)
            invalid_rows = df[invalid_mask].index.tolist()

            raise ERIDataValidationError(
                f"Invalid split names found: {invalid_splits}. "
                f"Valid splits: {self.VALID_SPLITS}. "
                f"Invalid splits found at rows: {invalid_rows[:10]}"
            )

    def _validate_data_consistency(self, df: pd.DataFrame) -> None:
        """Validate data consistency and completeness."""
        errors = []

        # Check for duplicate entries and handle them
        duplicates = df.duplicated(subset=['method', 'seed', 'epoch_eff', 'split'])
        if duplicates.any():
            duplicate_rows = df[duplicates].index.tolist()
            self.logger.warning(f"Found {len(duplicate_rows)} duplicate entries - will average them")

            # Average duplicate entries instead of failing
            df_clean = df.groupby(['method', 'seed', 'epoch_eff', 'split']).agg({
                'acc': 'mean',
                'top5': 'mean',
                'loss': 'mean'
            }).reset_index()

            # Update the original dataframe
            df.drop(df.index, inplace=True)
            df = pd.concat([df, df_clean], ignore_index=True)

        # Check epoch monotonicity within method-seed-split groups
        for (method, seed, split), group in df.groupby(['method', 'seed', 'split']):
            epochs = group['epoch_eff'].values
            if not np.all(epochs[:-1] <= epochs[1:]):
                errors.append(f"Non-monotonic epochs for method={method}, seed={seed}, split={split}")

        if errors:
            raise ERIDataValidationError("Consistency validation failed:\n" + "\n".join(errors))

    def _extract_method_from_export(self, export: Dict[str, Any]) -> str:
        """Extract method name from evaluator export."""
        config = export.get('configuration', {})

        # Try various fields that might contain method info
        if 'method' in config:
            return str(config['method'])
        elif 'model_name' in config:
            return str(config['model_name'])
        elif 'algorithm' in config:
            return str(config['algorithm'])
        elif 'dataset_name' in config:
            # Extract method from dataset name if it contains method info
            dataset_name = config['dataset_name']
            if 'derpp' in dataset_name.lower():
                return 'derpp'
            elif 'ewc' in dataset_name.lower():
                return 'ewc_on'
            elif 'sgd' in dataset_name.lower():
                return 'sgd'
        else:
            # Default fallback
            return 'unknown_method'

    def _extract_seed_from_export(self, export: Dict[str, Any]) -> int:
        """Extract seed from evaluator export."""
        config = export.get('configuration', {})

        # Try various fields that might contain seed info
        if 'seed' in config:
            return int(config['seed'])
        elif 'random_seed' in config:
            return int(config['random_seed'])
        else:
            # Default fallback
            return 42

    def _convert_accuracy_curves_format(self, legacy: Dict[str, Any]) -> ERITimelineDataset:
        """Convert legacy accuracy curves format."""
        curves = legacy['accuracy_curves']
        rows = []

        for subset_name, (epochs, accuracies) in curves.items():
            if subset_name in self.VALID_SPLITS:
                method = legacy.get('method', 'legacy_method')
                seed = legacy.get('seed', 42)

                for epoch, acc in zip(epochs, accuracies):
                    acc = float(acc)
                    rows.append({
                        'method': method,
                        'seed': seed,
                        'epoch_eff': float(epoch),
                        'split': subset_name,
                        'acc': acc,
                        'top5': min(1.0, acc + 0.05),
                        'loss': max(0.0, 1.0 - acc)
                    })

        if not rows:
            raise ERIDataValidationError("No valid data found in legacy accuracy curves")

        df = pd.DataFrame(rows)
        self.validate_format(df)

        metadata = {
            'source': 'legacy_accuracy_curves',
            'original_format': 'accuracy_curves'
        }

        return self._create_dataset_from_dataframe(df, metadata)

    def _convert_results_format(self, legacy: Dict[str, Any]) -> ERITimelineDataset:
        """Convert legacy results format."""
        results = legacy['results']
        rows = []

        method = legacy.get('method', 'legacy_method')
        seed = legacy.get('seed', 42)

        for result in results:
            epoch = result.get('epoch', 0)
            for split, acc in result.get('accuracies', {}).items():
                if split in self.VALID_SPLITS:
                    acc = float(acc)
                    rows.append({
                        'method': method,
                        'seed': seed,
                        'epoch_eff': float(epoch),
                        'split': split,
                        'acc': acc,
                        'top5': min(1.0, acc + 0.05),
                        'loss': max(0.0, 1.0 - acc)
                    })

        if not rows:
            raise ERIDataValidationError("No valid data found in legacy results")

        df = pd.DataFrame(rows)
        self.validate_format(df)

        metadata = {
            'source': 'legacy_results',
            'original_format': 'results'
        }

        return self._create_dataset_from_dataframe(df, metadata)
