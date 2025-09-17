"""
Tests for ERIDataLoader - Data loading and validation functionality.

This module provides comprehensive tests for the ERIDataLoader class,
covering CSV loading, validation, and conversion from various formats.
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from eri_vis.data_loader import ERIDataLoader, ERIDataValidationError
from eri_vis.dataset import ERITimelineDataset


class TestERIDataLoader:
    """Test suite for ERIDataLoader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ERIDataLoader()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_valid_csv_data(self) -> pd.DataFrame:
        """Create valid CSV data for testing."""
        return pd.DataFrame([
            {'method': 'Scratch_T2', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T1_all', 'acc': 0.85},
            {'method': 'Scratch_T2', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T2_shortcut_normal', 'acc': 0.10},
            {'method': 'Scratch_T2', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T2_shortcut_masked', 'acc': 0.05},
            {'method': 'Scratch_T2', 'seed': 42, 'epoch_eff': 1.0, 'split': 'T1_all', 'acc': 0.87},
            {'method': 'Scratch_T2', 'seed': 42, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': 0.30},
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T2_shortcut_normal', 'acc': 0.20},
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': 0.40},
        ])

    def create_valid_evaluator_export(self) -> dict:
        """Create valid evaluator export data for testing."""
        return {
            'configuration': {
                'dataset_name': 'seq-cifar100-einstellung',
                'method': 'sgd',
                'seed': 42,
                'adaptation_threshold': 0.8
            },
            'timeline_data': [
                {
                    'epoch': 0,
                    'task_id': 1,
                    'subset_accuracies': {
                        'T1_all': 0.85,
                        'T2_shortcut_normal': 0.10,
                        'T2_shortcut_masked': 0.05,
                        'T2_nonshortcut_normal': 0.15
                    },
                    'timestamp': 1640995200.0
                },
                {
                    'epoch': 1,
                    'task_id': 1,
                    'subset_accuracies': {
                        'T1_all': 0.87,
                        'T2_shortcut_normal': 0.30,
                        'T2_shortcut_masked': 0.25,
                        'T2_nonshortcut_normal': 0.20
                    },
                    'timestamp': 1640995260.0
                }
            ],
            'final_metrics': {
                'adaptation_delay': 5.0,
                'performance_deficit': 0.15,
                'eri_score': 0.75
            }
        }

    def test_load_csv_valid_file(self):
        """Test loading a valid CSV file."""
        # Create valid CSV file
        df = self.create_valid_csv_data()
        csv_path = Path(self.temp_dir) / "valid_data.csv"
        df.to_csv(csv_path, index=False)

        # Load and validate
        dataset = self.loader.load_csv(csv_path)

        # Assertions
        assert isinstance(dataset, ERITimelineDataset)
        assert len(dataset.data) == len(df)
        assert set(dataset.methods) == {'Scratch_T2', 'sgd'}
        assert set(dataset.splits) == {'T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked'}
        assert dataset.seeds == [42]
        assert dataset.epoch_range == (0.0, 1.0)
        assert dataset.metadata['source'] == str(csv_path)

    def test_load_csv_missing_file(self):
        """Test loading non-existent CSV file."""
        non_existent_path = Path(self.temp_dir) / "missing.csv"

        with pytest.raises(FileNotFoundError):
            self.loader.load_csv(non_existent_path)

    def test_load_csv_missing_columns(self):
        """Test loading CSV with missing required columns."""
        # Create CSV missing 'acc' column
        df = pd.DataFrame([
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T1_all'}
        ])
        csv_path = Path(self.temp_dir) / "missing_cols.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.load_csv(csv_path)

        assert "Missing required columns" in str(exc_info.value)
        assert "acc" in str(exc_info.value)

    def test_load_csv_invalid_data_types(self):
        """Test loading CSV with invalid data types."""
        # Create CSV with invalid seed type
        df = pd.DataFrame([
            {'method': 'sgd', 'seed': 'invalid', 'epoch_eff': 0.0, 'split': 'T1_all', 'acc': 0.5}
        ])
        csv_path = Path(self.temp_dir) / "invalid_types.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.load_csv(csv_path)

        assert "non-integer values" in str(exc_info.value)

    def test_load_csv_invalid_splits(self):
        """Test loading CSV with invalid split names."""
        df = pd.DataFrame([
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 0.0, 'split': 'invalid_split', 'acc': 0.5}
        ])
        csv_path = Path(self.temp_dir) / "invalid_splits.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.load_csv(csv_path)

        assert "Invalid split names" in str(exc_info.value)
        assert "invalid_split" in str(exc_info.value)

    def test_load_csv_invalid_accuracy_range(self):
        """Test loading CSV with accuracy values outside [0,1] range."""
        df = pd.DataFrame([
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T1_all', 'acc': 1.5}
        ])
        csv_path = Path(self.temp_dir) / "invalid_acc.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.load_csv(csv_path)

        assert "Accuracy values must be in [0, 1]" in str(exc_info.value)

    def test_load_csv_negative_epochs(self):
        """Test loading CSV with negative epoch values."""
        df = pd.DataFrame([
            {'method': 'sgd', 'seed': 42, 'epoch_eff': -1.0, 'split': 'T1_all', 'acc': 0.5}
        ])
        csv_path = Path(self.temp_dir) / "negative_epochs.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.load_csv(csv_path)

        assert "Epoch_eff values must be non-negative" in str(exc_info.value)

    def test_load_csv_duplicate_entries(self):
        """Test loading CSV with duplicate entries."""
        df = pd.DataFrame([
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T1_all', 'acc': 0.5},
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T1_all', 'acc': 0.6}  # Duplicate
        ])
        csv_path = Path(self.temp_dir) / "duplicates.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.load_csv(csv_path)

        assert "Duplicate entries found" in str(exc_info.value)

    def test_load_from_evaluator_export_valid(self):
        """Test loading from valid evaluator export."""
        export = self.create_valid_evaluator_export()

        dataset = self.loader.load_from_evaluator_export(export)

        # Assertions
        assert isinstance(dataset, ERITimelineDataset)
        assert len(dataset.data) > 0
        assert 'sgd' in dataset.methods
        assert 42 in dataset.seeds
        assert set(dataset.splits).issubset(self.loader.VALID_SPLITS)
        assert dataset.metadata['source'] == 'evaluator_export'

    def test_load_from_evaluator_export_missing_timeline(self):
        """Test loading from export missing timeline_data."""
        export = {'configuration': {}, 'final_metrics': {}}

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.load_from_evaluator_export(export)

        assert "missing 'timeline_data'" in str(exc_info.value)

    def test_load_from_evaluator_export_empty_timeline(self):
        """Test loading from export with empty timeline."""
        export = {
            'configuration': {},
            'timeline_data': [],
            'final_metrics': {}
        }

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.load_from_evaluator_export(export)

        assert "Timeline data is empty" in str(exc_info.value)

    def test_validate_format_valid_data(self):
        """Test validation of valid DataFrame."""
        df = self.create_valid_csv_data()

        # Should not raise any exception
        self.loader.validate_format(df)

    def test_validate_format_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame(columns=self.loader.REQUIRED_COLS)

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.validate_format(df)

        assert "DataFrame is empty" in str(exc_info.value)

    def test_validate_format_non_monotonic_epochs(self):
        """Test validation with non-monotonic epochs."""
        df = pd.DataFrame([
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 1.0, 'split': 'T1_all', 'acc': 0.5},
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T1_all', 'acc': 0.4}  # Non-monotonic
        ])

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.validate_format(df)

        assert "Non-monotonic epochs" in str(exc_info.value)

    def test_convert_legacy_accuracy_curves_format(self):
        """Test conversion from legacy accuracy curves format."""
        legacy = {
            'method': 'sgd',
            'seed': 42,
            'accuracy_curves': {
                'T1_all': ([0, 1, 2], [0.5, 0.6, 0.7]),
                'T2_shortcut_normal': ([0, 1, 2], [0.1, 0.2, 0.3]),
                'invalid_split': ([0, 1], [0.8, 0.9])  # Should be ignored
            }
        }

        dataset = self.loader.convert_legacy_format(legacy)

        # Assertions
        assert isinstance(dataset, ERITimelineDataset)
        assert len(dataset.data) == 6  # 2 valid splits × 3 epochs
        assert 'sgd' in dataset.methods
        assert 42 in dataset.seeds
        assert 'invalid_split' not in dataset.splits
        assert dataset.metadata['source'] == 'legacy_accuracy_curves'

    def test_convert_legacy_results_format(self):
        """Test conversion from legacy results format."""
        legacy = {
            'method': 'ewc_on',
            'seed': 123,
            'results': [
                {
                    'epoch': 0,
                    'accuracies': {
                        'T1_all': 0.8,
                        'T2_shortcut_normal': 0.1,
                        'invalid_split': 0.9  # Should be ignored
                    }
                },
                {
                    'epoch': 1,
                    'accuracies': {
                        'T1_all': 0.85,
                        'T2_shortcut_masked': 0.05
                    }
                }
            ]
        }

        dataset = self.loader.convert_legacy_format(legacy)

        # Assertions
        assert isinstance(dataset, ERITimelineDataset)
        assert len(dataset.data) == 4  # Valid entries only (T1_all appears twice + T2_shortcut_normal + T2_shortcut_masked)
        assert 'ewc_on' in dataset.methods
        assert 123 in dataset.seeds
        assert 'invalid_split' not in dataset.splits
        assert dataset.metadata['source'] == 'legacy_results'

    def test_convert_legacy_unknown_format(self):
        """Test conversion from unknown legacy format."""
        legacy = {'unknown_field': 'value'}

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.convert_legacy_format(legacy)

        assert "Unknown legacy format" in str(exc_info.value)

    def test_detailed_error_messages_with_context(self):
        """Test that error messages include row/column context."""
        # Create data with multiple validation errors
        df = pd.DataFrame([
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T1_all', 'acc': 0.5},  # Valid
            {'method': '', 'seed': -1, 'epoch_eff': -0.5, 'split': 'invalid', 'acc': 1.5},  # Multiple errors
            {'method': 'ewc', 'seed': 'bad', 'epoch_eff': 'bad', 'split': 'T1_all', 'acc': 'bad'},  # Type errors
        ])

        with pytest.raises(ERIDataValidationError) as exc_info:
            self.loader.validate_format(df)

        error_msg = str(exc_info.value)

        # Check that error messages include row context
        assert "at rows" in error_msg or "Invalid at rows" in error_msg

    def test_valid_splits_constant(self):
        """Test that VALID_SPLITS contains expected values."""
        expected_splits = {
            "T1_all",
            "T2_shortcut_normal",
            "T2_shortcut_masked",
            "T2_nonshortcut_normal"
        }

        assert self.loader.VALID_SPLITS == expected_splits

    def test_required_cols_constant(self):
        """Test that REQUIRED_COLS contains expected values."""
        expected_cols = ["method", "seed", "epoch_eff", "split", "acc"]

        assert self.loader.REQUIRED_COLS == expected_cols

    def test_dataset_metadata_completeness(self):
        """Test that created datasets have complete metadata."""
        df = self.create_valid_csv_data()
        csv_path = Path(self.temp_dir) / "test.csv"
        df.to_csv(csv_path, index=False)

        dataset = self.loader.load_csv(csv_path)

        # Check metadata completeness
        required_metadata = ['source', 'n_methods', 'n_splits', 'n_seeds', 'n_rows', 'epoch_range']
        for key in required_metadata:
            assert key in dataset.metadata

        # Check metadata accuracy
        assert dataset.metadata['n_methods'] == len(dataset.methods)
        assert dataset.metadata['n_splits'] == len(dataset.splits)
        assert dataset.metadata['n_seeds'] == len(dataset.seeds)
        assert dataset.metadata['n_rows'] == len(dataset.data)
        assert dataset.metadata['epoch_range'] == dataset.epoch_range


if __name__ == "__main__":
    pytest.main([__file__])
