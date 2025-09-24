"""
Integration tests for ERIExperimentHooks with real EinstellungEvaluator

Tests the hooks integration with actual Mammoth components to ensure
the CSV output format matches expectations.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from eri_vis.integration.hooks import (
    ERIExperimentHooks,
    integrate_hooks_with_evaluator
)
from utils.einstellung_evaluator import EinstellungEvaluator


class TestHooksIntegrationWithEvaluator:
    """Integration tests with real EinstellungEvaluator."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "integration_test"

        # Create hooks
        self.hooks = ERIExperimentHooks(
            output_dir=str(self.output_dir),
            export_frequency=1,
            auto_visualize=False  # Disable visualization for integration test
        )

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_integration_with_real_evaluator(self):
        """Test hooks integration with real EinstellungEvaluator."""
        # Create mock args for evaluator
        mock_args = Mock()
        mock_args.model = "sgd"
        mock_args.seed = 42
        mock_args.device = "cpu"
        mock_args.einstellung_evaluation_subsets = True

        # Create evaluator
        evaluator = EinstellungEvaluator(
            args=mock_args,
            dataset_name='seq-cifar100-einstellung',
            adaptation_threshold=0.6,
            extract_attention=False
        )

        # Integrate hooks
        integrate_hooks_with_evaluator(evaluator, self.hooks)

        # Simulate timeline data that would be collected during training
        evaluator.timeline_data = [
            {
                'epoch': 1,
                'task_id': 0,
                'subset_accuracies': {
                    'T1_all': 0.80,
                    'T2_shortcut_normal': 0.50,
                    'T2_shortcut_masked': 0.30,
                    'T2_nonshortcut_normal': 0.60
                },
                'timestamp': 1640995200.0
            },
            {
                'epoch': 2,
                'task_id': 0,
                'subset_accuracies': {
                    'T1_all': 0.82,
                    'T2_shortcut_normal': 0.55,
                    'T2_shortcut_masked': 0.35,
                    'T2_nonshortcut_normal': 0.62
                },
                'timestamp': 1640995260.0
            }
        ]

        # Create mock model and dataset
        mock_model = Mock()
        mock_dataset = Mock()
        mock_dataset.NAME = 'seq-cifar100-einstellung'
        mock_dataset.i = 0
        mock_dataset.N_TASKS = 2

        # Test hooks directly (bypassing the complex evaluator logic)
        # First call with epoch 1 data
        evaluator.timeline_data = [evaluator.timeline_data[0]]  # Only epoch 1 data
        self.hooks.on_epoch_end(1, evaluator)

        # Then call with epoch 2 data
        evaluator.timeline_data = [
            evaluator.timeline_data[0],  # Keep epoch 1
            {
                'epoch': 2,
                'task_id': 0,
                'subset_accuracies': {
                    'T1_all': 0.82,
                    'T2_shortcut_normal': 0.55,
                    'T2_shortcut_masked': 0.35,
                    'T2_nonshortcut_normal': 0.62
                },
                'timestamp': 1640995260.0
            }
        ]
        self.hooks.on_epoch_end(2, evaluator)

        # Verify hooks collected data
        assert len(self.hooks.timeline_data) == 2

        # Test task end hook
        self.hooks.on_task_end(0, evaluator)

        # Verify CSV was created
        task_csv = self.output_dir / "eri_task_0_timeline.csv"
        assert task_csv.exists()

        # Verify CSV format
        df = pd.read_csv(task_csv)

        # Check required columns
        required_cols = ['method', 'seed', 'epoch_eff', 'split', 'acc']
        for col in required_cols:
            assert col in df.columns

        # Check data integrity
        assert len(df) == 8  # 2 epochs × 4 splits
        assert df['method'].iloc[0] == 'sgd'
        assert df['seed'].iloc[0] == 42

        # Check splits are valid
        valid_splits = {'T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal'}
        assert set(df['split'].unique()) == valid_splits

    def test_csv_format_matches_specification(self):
        """Test that generated CSV matches the exact specification."""
        # Add test data
        self.hooks.timeline_data = [
            {
                'epoch': 1,
                'epoch_eff': 1.0,
                'method': 'sgd',
                'seed': 42,
                'subset_accuracies': {
                    'T1_all': 0.80,
                    'T2_shortcut_normal': 0.50,
                    'T2_shortcut_masked': 0.30,
                    'T2_nonshortcut_normal': 0.60
                }
            }
        ]

        # Export CSV
        csv_path = self.output_dir / "test_format.csv"
        self.hooks._export_timeline_csv(csv_path)

        # Read and verify format
        df = pd.read_csv(csv_path)

        # Check exact column names and order
        expected_columns = ['method', 'seed', 'epoch_eff', 'split', 'acc', 'top5', 'loss']
        assert list(df.columns) == expected_columns

        # Check data types
        assert df['method'].dtype == 'object'  # string
        assert df['seed'].dtype == 'int64'
        assert df['epoch_eff'].dtype == 'float64'
        assert df['split'].dtype == 'object'  # string
        assert df['acc'].dtype == 'float64'

        # Check value ranges
        assert all(df['acc'] >= 0.0) and all(df['acc'] <= 1.0)
        assert all(df['seed'] >= 0)
        assert all(df['epoch_eff'] >= 0.0)

        # Check deterministic ordering
        # Should be sorted by method, seed, epoch_eff, split
        for i in range(1, len(df)):
            prev_row = df.iloc[i-1]
            curr_row = df.iloc[i]

            # Check ordering
            if prev_row['method'] == curr_row['method']:
                if prev_row['seed'] == curr_row['seed']:
                    if prev_row['epoch_eff'] == curr_row['epoch_eff']:
                        assert prev_row['split'] <= curr_row['split']
                    else:
                        assert prev_row['epoch_eff'] <= curr_row['epoch_eff']
                else:
                    assert prev_row['seed'] <= curr_row['seed']

    def test_experiment_end_produces_final_csv(self):
        """Test that experiment end produces the required timeline.csv."""
        # Add timeline data
        self.hooks.timeline_data = [
            {
                'epoch': 5,
                'method': 'sgd',
                'seed': 42,
                'subset_accuracies': {
                    'T1_all': 0.85,
                    'T2_shortcut_normal': 0.60,
                    'T2_shortcut_masked': 0.45,
                    'T2_nonshortcut_normal': 0.70
                }
            }
        ]

        # Mock evaluator
        evaluator = Mock()

        # Call experiment end
        result = self.hooks.on_experiment_end(evaluator)

        # Check that timeline.csv was created
        final_csv = self.output_dir / "timeline_sgd.csv"
        assert final_csv.exists()
        assert 'csv' in result
        assert result['csv'] == str(final_csv)

        # Verify CSV content
        df = pd.read_csv(final_csv)
        assert len(df) == 4  # 1 epoch × 4 splits
        assert {'method', 'seed', 'epoch_eff', 'split', 'acc', 'top5', 'loss'}.issubset(df.columns)

        # Check metadata was also created
        metadata_file = self.output_dir / "eri_experiment_metadata.json"
        assert metadata_file.exists()
        assert 'metadata' in result

        # Verify metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        assert 'experiment_info' in metadata
        assert 'processing_config' in metadata
        assert 'timeline_summary' in metadata
        assert metadata['experiment_info']['total_epochs'] == 1


if __name__ == '__main__':
    pytest.main([__file__])
