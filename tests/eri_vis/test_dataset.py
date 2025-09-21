"""
Tests for ERITimelineDataset - Core data structure and manipulation.

This module tests the ERITimelineDataset class functionality including:
- Filtering by method, seed, and split
- Epoch alignment with uneven grids
- Deterministic CSV export with metadata sidecar
- Data validation and error handling
"""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from eri_vis.dataset import ERITimelineDataset


class TestERITimelineDataset:
    """Test suite for ERITimelineDataset class."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample timeline data for testing."""
        np.random.seed(42)  # For reproducible tests

        methods = ['Scratch_T2', 'sgd', 'ewc_on']
        seeds = [1, 2, 3]
        splits = ['T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked']
        epochs = [0.0, 1.0, 2.0, 3.0, 4.0]

        rows = []
        for method in methods:
            for seed in seeds:
                for split in splits:
                    for epoch in epochs:
                        # Create realistic accuracy progression
                        base_acc = 0.1 if split == 'T2_shortcut_normal' else 0.05
                        if method == 'Scratch_T2':
                            acc = base_acc + 0.1 * epoch + np.random.normal(0, 0.01)
                        else:
                            acc = base_acc + 0.05 * epoch + np.random.normal(0, 0.01)

                        acc = np.clip(acc, 0.0, 1.0)  # Ensure valid range

                        rows.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch,
                            'split': split,
                            'acc': acc
                        })

        return pd.DataFrame(rows)

    @pytest.fixture
    def sample_dataset(self, sample_data) -> ERITimelineDataset:
        """Create sample ERITimelineDataset for testing."""
        methods = sorted(sample_data['method'].unique())
        splits = sorted(sample_data['split'].unique())
        seeds = sorted(sample_data['seed'].unique())
        epoch_range = (
            float(sample_data['epoch_eff'].min()),
            float(sample_data['epoch_eff'].max())
        )

        metadata = {
            'source': 'test_data',
            'created_for': 'unit_testing'
        }

        return ERITimelineDataset(
            data=sample_data,
            metadata=metadata,
            methods=methods,
            splits=splits,
            seeds=seeds,
            epoch_range=epoch_range
        )

    @pytest.fixture
    def uneven_epoch_data(self) -> pd.DataFrame:
        """Create data with uneven epoch grids for alignment testing."""
        rows = []

        # Method 1: epochs [0, 1, 2, 3, 4]
        for epoch in [0.0, 1.0, 2.0, 3.0, 4.0]:
            rows.append({
                'method': 'method1',
                'seed': 1,
                'epoch_eff': epoch,
                'split': 'T2_shortcut_normal',
                'acc': 0.1 + 0.1 * epoch
            })

        # Method 2: epochs [0, 0.5, 1.5, 2.5, 4]
        for epoch in [0.0, 0.5, 1.5, 2.5, 4.0]:
            rows.append({
                'method': 'method2',
                'seed': 1,
                'epoch_eff': epoch,
                'split': 'T2_shortcut_normal',
                'acc': 0.05 + 0.08 * epoch
            })

        # Method 3: epochs [0, 2, 4] (sparse)
        for epoch in [0.0, 2.0, 4.0]:
            rows.append({
                'method': 'method3',
                'seed': 1,
                'epoch_eff': epoch,
                'split': 'T2_shortcut_normal',
                'acc': 0.08 + 0.06 * epoch
            })

        return pd.DataFrame(rows)

    @pytest.fixture
    def uneven_dataset(self, uneven_epoch_data) -> ERITimelineDataset:
        """Create dataset with uneven epochs for alignment testing."""
        methods = sorted(uneven_epoch_data['method'].unique())
        splits = sorted(uneven_epoch_data['split'].unique())
        seeds = sorted(uneven_epoch_data['seed'].unique())
        epoch_range = (
            float(uneven_epoch_data['epoch_eff'].min()),
            float(uneven_epoch_data['epoch_eff'].max())
        )

        metadata = {'source': 'uneven_test_data'}

        return ERITimelineDataset(
            data=uneven_epoch_data,
            metadata=metadata,
            methods=methods,
            splits=splits,
            seeds=seeds,
            epoch_range=epoch_range
        )

    def test_basic_properties(self, sample_dataset):
        """Test basic dataset properties and accessors."""
        assert len(sample_dataset) == 135  # 3 methods × 3 seeds × 3 splits × 5 epochs
        assert sample_dataset.methods == ['Scratch_T2', 'ewc_on', 'sgd']
        assert sample_dataset.splits == ['T1_all', 'T2_shortcut_masked', 'T2_shortcut_normal']
        assert sample_dataset.seeds == [1, 2, 3]
        assert sample_dataset.epoch_range == (0.0, 4.0)

    def test_get_method_data(self, sample_dataset):
        """Test filtering by method."""
        sgd_data = sample_dataset.get_method_data('sgd')

        assert len(sgd_data) == 45  # 3 seeds × 3 splits × 5 epochs
        assert all(sgd_data['method'] == 'sgd')
        assert set(sgd_data['seed'].unique()) == {1, 2, 3}
        assert set(sgd_data['split'].unique()) == {'T1_all', 'T2_shortcut_masked', 'T2_shortcut_normal'}

    def test_get_method_data_invalid(self, sample_dataset):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError, match="Method 'invalid_method' not found"):
            sample_dataset.get_method_data('invalid_method')

    def test_get_split_data(self, sample_dataset):
        """Test filtering by split."""
        split_data = sample_dataset.get_split_data('T2_shortcut_normal')

        assert len(split_data) == 45  # 3 methods × 3 seeds × 5 epochs
        assert all(split_data['split'] == 'T2_shortcut_normal')
        assert set(split_data['method'].unique()) == {'Scratch_T2', 'ewc_on', 'sgd'}

    def test_get_split_data_invalid(self, sample_dataset):
        """Test error handling for invalid split."""
        with pytest.raises(ValueError, match="Split 'invalid_split' not found"):
            sample_dataset.get_split_data('invalid_split')

    def test_get_seed_data(self, sample_dataset):
        """Test filtering by seed."""
        seed_data = sample_dataset.get_seed_data(2)

        assert len(seed_data) == 45  # 3 methods × 3 splits × 5 epochs
        assert all(seed_data['seed'] == 2)
        assert set(seed_data['method'].unique()) == {'Scratch_T2', 'ewc_on', 'sgd'}

    def test_get_seed_data_invalid(self, sample_dataset):
        """Test error handling for invalid seed."""
        with pytest.raises(ValueError, match="Seed 999 not found"):
            sample_dataset.get_seed_data(999)

    def test_filter_data_single_criteria(self, sample_dataset):
        """Test filtering by single criteria."""
        # Filter by methods
        filtered = sample_dataset.filter_data(methods=['sgd', 'ewc_on'])
        assert len(filtered) == 90  # 2 methods × 3 seeds × 3 splits × 5 epochs
        assert set(filtered.methods) == {'sgd', 'ewc_on'}

        # Filter by seeds
        filtered = sample_dataset.filter_data(seeds=[1, 3])
        assert len(filtered) == 90  # 3 methods × 2 seeds × 3 splits × 5 epochs
        assert set(filtered.seeds) == {1, 3}

        # Filter by splits
        filtered = sample_dataset.filter_data(splits=['T2_shortcut_normal'])
        assert len(filtered) == 45  # 3 methods × 3 seeds × 1 split × 5 epochs
        assert filtered.splits == ['T2_shortcut_normal']

    def test_filter_data_multiple_criteria(self, sample_dataset):
        """Test filtering by multiple criteria."""
        filtered = sample_dataset.filter_data(
            methods=['sgd'],
            seeds=[1, 2],
            splits=['T2_shortcut_normal', 'T2_shortcut_masked']
        )

        assert len(filtered) == 20  # 1 method × 2 seeds × 2 splits × 5 epochs
        assert filtered.methods == ['sgd']
        assert set(filtered.seeds) == {1, 2}
        assert set(filtered.splits) == {'T2_shortcut_masked', 'T2_shortcut_normal'}

    def test_filter_data_invalid_values(self, sample_dataset):
        """Test error handling for invalid filter values."""
        with pytest.raises(ValueError, match="Invalid methods"):
            sample_dataset.filter_data(methods=['invalid_method'])

        with pytest.raises(ValueError, match="Invalid seeds"):
            sample_dataset.filter_data(seeds=[999])

        with pytest.raises(ValueError, match="Invalid splits"):
            sample_dataset.filter_data(splits=['invalid_split'])

    def test_filter_data_empty_result(self, sample_dataset):
        """Test error handling when filtering results in empty dataset."""
        # First test invalid seeds error
        with pytest.raises(ValueError, match="Invalid seeds"):
            sample_dataset.filter_data(seeds=[999])

        # Create an empty dataset to test empty filtering
        empty_data = pd.DataFrame(columns=['method', 'seed', 'epoch_eff', 'split', 'acc'])
        empty_dataset = ERITimelineDataset(
            data=empty_data,
            metadata={'source': 'empty_test'},
            methods=[],
            splits=[],
            seeds=[],
            epoch_range=(0.0, 0.0)
        )

        # Any filter on empty dataset should raise the empty result error
        with pytest.raises(ValueError, match="Filtering resulted in empty dataset"):
            empty_dataset.filter_data(methods=[])

    def test_align_epochs_basic(self, uneven_dataset):
        """Test basic epoch alignment functionality."""
        common_epochs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        aligned = uneven_dataset.align_epochs(common_epochs)

        # Check that all methods now have the same epochs
        for method in aligned.methods:
            method_data = aligned.get_method_data(method)
            method_epochs = sorted(method_data['epoch_eff'].unique())
            np.testing.assert_array_equal(method_epochs, common_epochs)

        # Check metadata
        assert aligned.metadata['aligned'] is True
        assert aligned.metadata['common_epochs'] == common_epochs.tolist()

    def test_align_epochs_interpolation_accuracy(self, uneven_dataset):
        """Test that epoch alignment preserves accuracy trends."""
        common_epochs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        aligned = uneven_dataset.align_epochs(common_epochs)

        # Check method1 (originally had exact epochs) - should be unchanged
        method1_data = aligned.get_method_data('method1')
        method1_data = method1_data.sort_values('epoch_eff')
        expected_accs = [0.1, 0.2, 0.3, 0.4, 0.5]  # 0.1 + 0.1 * epoch
        np.testing.assert_array_almost_equal(
            method1_data['acc'].values, expected_accs, decimal=3
        )

        # Check method2 - should have interpolated values
        method2_data = aligned.get_method_data('method2')
        method2_data = method2_data.sort_values('epoch_eff')
        # Original: epochs [0, 0.5, 1.5, 2.5, 4] with acc = 0.05 + 0.08 * epoch
        # At epoch 1.0, should interpolate between (0.5, 0.09) and (1.5, 0.17)
        assert 0.08 < method2_data.iloc[1]['acc'] < 0.18  # Reasonable interpolation

    def test_align_epochs_extrapolation_warning(self, uneven_dataset, caplog):
        """Test that extrapolation generates appropriate warnings."""
        # Request epochs outside the original range
        common_epochs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        aligned = uneven_dataset.align_epochs(common_epochs)

        # Check that warnings were logged
        assert "Extrapolation needed" in caplog.text

        # Check that extrapolated values are reasonable (not NaN or extreme)
        for method in aligned.methods:
            method_data = aligned.get_method_data(method)
            assert not method_data['acc'].isna().any()
            assert all((method_data['acc'] >= 0) & (method_data['acc'] <= 1))

    def test_align_epochs_single_point(self):
        """Test alignment with single data point per group."""
        # Create data with single point per method-seed-split
        single_point_data = pd.DataFrame([
            {'method': 'method1', 'seed': 1, 'epoch_eff': 2.0, 'split': 'T2_shortcut_normal', 'acc': 0.3},
            {'method': 'method2', 'seed': 1, 'epoch_eff': 1.5, 'split': 'T2_shortcut_normal', 'acc': 0.25}
        ])

        dataset = ERITimelineDataset(
            data=single_point_data,
            metadata={'source': 'single_point_test'},
            methods=['method1', 'method2'],
            splits=['T2_shortcut_normal'],
            seeds=[1],
            epoch_range=(1.5, 2.0)
        )

        common_epochs = np.array([0.0, 1.0, 2.0, 3.0])
        aligned = dataset.align_epochs(common_epochs)

        # Single points should result in constant extrapolation
        method1_data = aligned.get_method_data('method1')
        assert all(method1_data['acc'] == 0.3)

    def test_align_epochs_duplicate_epochs(self, caplog):
        """Test handling of duplicate epochs in alignment."""
        # Create data with duplicate epochs
        duplicate_data = pd.DataFrame([
            {'method': 'method1', 'seed': 1, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': 0.2},
            {'method': 'method1', 'seed': 1, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': 0.3},
            {'method': 'method1', 'seed': 1, 'epoch_eff': 2.0, 'split': 'T2_shortcut_normal', 'acc': 0.4}
        ])

        dataset = ERITimelineDataset(
            data=duplicate_data,
            metadata={'source': 'duplicate_test'},
            methods=['method1'],
            splits=['T2_shortcut_normal'],
            seeds=[1],
            epoch_range=(1.0, 2.0)
        )

        common_epochs = np.array([1.0, 1.5, 2.0])
        aligned = dataset.align_epochs(common_epochs)

        # Should log warning about duplicates
        assert "Duplicate epochs found" in caplog.text

        # Should use mean of duplicate values (0.25)
        method1_data = aligned.get_method_data('method1')
        method1_data = method1_data.sort_values('epoch_eff')
        assert abs(method1_data.iloc[0]['acc'] - 0.25) < 0.01  # Mean of 0.2 and 0.3

    def test_align_epochs_invalid_input(self, sample_dataset):
        """Test error handling for invalid alignment input."""
        # Empty epochs
        with pytest.raises(ValueError, match="common_epochs cannot be empty"):
            sample_dataset.align_epochs(np.array([]))

        # Non-monotonic epochs
        with pytest.raises(ValueError, match="must be monotonically non-decreasing"):
            sample_dataset.align_epochs(np.array([2.0, 1.0, 3.0]))

        # Negative epochs should work but generate warnings (tested in other test)

    def test_export_csv_basic(self, sample_dataset):
        """Test basic CSV export functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_export.csv"

            sample_dataset.export_csv(csv_path)

            # Check that file was created
            assert csv_path.exists()

            # Check that data can be read back
            exported_data = pd.read_csv(csv_path)
            assert len(exported_data) == len(sample_dataset)
            assert list(exported_data.columns) == ['method', 'seed', 'epoch_eff', 'split', 'acc']

    def test_export_csv_deterministic_ordering(self, sample_dataset):
        """Test that CSV export has deterministic ordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path1 = Path(tmpdir) / "export1.csv"
            csv_path2 = Path(tmpdir) / "export2.csv"

            # Export twice
            sample_dataset.export_csv(csv_path1)
            sample_dataset.export_csv(csv_path2)

            # Files should be identical
            with open(csv_path1, 'r') as f1, open(csv_path2, 'r') as f2:
                content1 = f1.read()
                content2 = f2.read()
                assert content1 == content2

    def test_export_csv_with_metadata_sidecar(self, sample_dataset):
        """Test CSV export with metadata sidecar JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_export.csv"
            json_path = Path(tmpdir) / "test_export.json"

            sample_dataset.export_csv(csv_path, include_metadata=True)

            # Check that both files were created
            assert csv_path.exists()
            assert json_path.exists()

            # Check metadata content
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            assert 'export_info' in metadata
            assert 'dataset_summary' in metadata
            assert 'original_metadata' in metadata
            assert 'data_hash' in metadata

            assert metadata['export_info']['n_rows_exported'] == len(sample_dataset)
            assert metadata['dataset_summary']['n_methods'] == 3

    def test_export_csv_deterministic_hash(self, sample_dataset):
        """Test that deterministic export produces consistent hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path1 = Path(tmpdir) / "export1.csv"
            csv_path2 = Path(tmpdir) / "export2.csv"
            json_path1 = Path(tmpdir) / "export1.json"
            json_path2 = Path(tmpdir) / "export2.json"

            # Export twice
            sample_dataset.export_csv(csv_path1, include_metadata=True)
            sample_dataset.export_csv(csv_path2, include_metadata=True)

            # Load metadata
            with open(json_path1, 'r') as f:
                metadata1 = json.load(f)
            with open(json_path2, 'r') as f:
                metadata2 = json.load(f)

            # Hashes should be identical (excluding timestamp)
            assert metadata1['data_hash'] == metadata2['data_hash']

    def test_export_csv_directory_creation(self, sample_dataset):
        """Test that export creates directories as needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "export.csv"

            sample_dataset.export_csv(nested_path)

            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_get_summary_stats(self, sample_dataset):
        """Test summary statistics generation."""
        stats = sample_dataset.get_summary_stats()

        # Check basic counts
        assert stats['n_rows'] == 135
        assert stats['n_methods'] == 3
        assert stats['n_seeds'] == 3
        assert stats['n_splits'] == 3

        # Check lists
        assert stats['methods'] == ['Scratch_T2', 'ewc_on', 'sgd']
        assert stats['seeds'] == [1, 2, 3]
        assert stats['splits'] == ['T1_all', 'T2_shortcut_masked', 'T2_shortcut_normal']

        # Check ranges
        assert stats['epoch_range'] == (0.0, 4.0)
        assert 0.0 <= stats['accuracy_range'][0] <= stats['accuracy_range'][1] <= 1.0

        # Check per-method stats
        assert all(method in stats['epochs_per_method'] for method in stats['methods'])
        assert all(method in stats['seeds_per_method'] for method in stats['methods'])

        # Check completeness
        assert stats['completeness']['completeness_ratio'] == 1.0  # Complete dataset

    def test_string_representation(self, sample_dataset):
        """Test string representation of dataset."""
        repr_str = repr(sample_dataset)

        assert "ERITimelineDataset" in repr_str
        assert "n_rows=135" in repr_str
        assert "methods=3" in repr_str
        assert "seeds=3" in repr_str
        assert "splits=3" in repr_str
        assert "epoch_range=(0.0, 4.0)" in repr_str

    def test_len_operator(self, sample_dataset):
        """Test len() operator."""
        assert len(sample_dataset) == 135

    def test_alignment_with_missing_data(self):
        """Test alignment when some method-seed-split combinations are missing."""
        # Create incomplete data
        incomplete_data = pd.DataFrame([
            {'method': 'method1', 'seed': 1, 'epoch_eff': 0.0, 'split': 'T2_shortcut_normal', 'acc': 0.1},
            {'method': 'method1', 'seed': 1, 'epoch_eff': 2.0, 'split': 'T2_shortcut_normal', 'acc': 0.3},
            # Missing method2 data
        ])

        dataset = ERITimelineDataset(
            data=incomplete_data,
            metadata={'source': 'incomplete_test'},
            methods=['method1'],
            splits=['T2_shortcut_normal'],
            seeds=[1],
            epoch_range=(0.0, 2.0)
        )

        common_epochs = np.array([0.0, 1.0, 2.0])
        aligned = dataset.align_epochs(common_epochs)

        # Should still work with available data
        assert len(aligned) == 3  # 1 method × 1 seed × 1 split × 3 epochs
        method1_data = aligned.get_method_data('method1')
        assert len(method1_data) == 3

    def test_alignment_interpolation_stats(self, uneven_dataset):
        """Test that alignment provides interpolation statistics."""
        common_epochs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        aligned = uneven_dataset.align_epochs(common_epochs)

        stats = aligned.metadata['alignment_stats']
        assert 'total_groups' in stats
        assert 'successful_interpolations' in stats
        assert 'failed_interpolations' in stats
        assert 'extrapolation_warnings' in stats

        assert stats['total_groups'] == 3  # 3 methods
        assert stats['successful_interpolations'] > 0
        assert stats['failed_interpolations'] == 0

    def test_filter_preserves_metadata(self, sample_dataset):
        """Test that filtering preserves and updates metadata appropriately."""
        filtered = sample_dataset.filter_data(methods=['sgd'])

        # Original metadata should be preserved
        assert filtered.metadata['source'] == 'test_data'
        assert filtered.metadata['created_for'] == 'unit_testing'

        # Filter metadata should be added
        assert filtered.metadata['filter_methods'] == ['sgd']
        assert filtered.metadata['filtered_n_methods'] == 1
        assert filtered.metadata['filtered_n_rows'] == 45

    def test_edge_case_single_epoch(self):
        """Test dataset with single epoch."""
        single_epoch_data = pd.DataFrame([
            {'method': 'method1', 'seed': 1, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': 0.3},
            {'method': 'method2', 'seed': 1, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': 0.25}
        ])

        dataset = ERITimelineDataset(
            data=single_epoch_data,
            metadata={'source': 'single_epoch_test'},
            methods=['method1', 'method2'],
            splits=['T2_shortcut_normal'],
            seeds=[1],
            epoch_range=(1.0, 1.0)
        )

        # Should handle single epoch gracefully
        assert len(dataset) == 2
        assert dataset.epoch_range == (1.0, 1.0)

        # Alignment should still work
        common_epochs = np.array([0.0, 1.0, 2.0])
        aligned = dataset.align_epochs(common_epochs)
        assert len(aligned) == 6  # 2 methods × 1 seed × 1 split × 3 epochs

    def test_accuracy_clamping_in_alignment(self):
        """Test that alignment clamps accuracies to valid [0, 1] range."""
        # Create data that might extrapolate outside [0, 1]
        extreme_data = pd.DataFrame([
            {'method': 'method1', 'seed': 1, 'epoch_eff': 0.0, 'split': 'T2_shortcut_normal', 'acc': 0.0},
            {'method': 'method1', 'seed': 1, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': 1.0}
        ])

        dataset = ERITimelineDataset(
            data=extreme_data,
            metadata={'source': 'extreme_test'},
            methods=['method1'],
            splits=['T2_shortcut_normal'],
            seeds=[1],
            epoch_range=(0.0, 1.0)
        )

        # Request extrapolation that would go outside [0, 1]
        common_epochs = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        aligned = dataset.align_epochs(common_epochs)

        # All accuracies should be in valid range
        all_accs = aligned.data['acc'].values
        assert all((all_accs >= 0.0) & (all_accs <= 1.0))
