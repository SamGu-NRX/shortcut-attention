"""
Tests for ERIExperimentHooks - Experiment Lifecycle Integration

Tests the structured callback system for integrating ERI visualization
with the Mammoth experiment lifecycle.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from eri_vis.integration.hooks import (
    ERIExperimentHooks,
    create_eri_hooks,
    integrate_hooks_with_evaluator
)
from eri_vis.styles import PlotStyleConfig


class TestERIExperimentHooks:
    """Test suite for ERIExperimentHooks class."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_output"

        self.hooks = ERIExperimentHooks(
            output_dir=str(self.output_dir),
            export_frequency=2,  # Export every 2 epochs for testing
            auto_visualize=True
        )

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test hooks initialization."""
        assert self.hooks.output_dir == self.output_dir
        assert self.hooks.export_frequency == 2
        assert self.hooks.auto_visualize is True
        assert self.hooks.timeline_data == []
        assert self.hooks.current_task == 0
        assert self.hooks.current_method == "unknown"
        assert self.hooks.current_seed == 42

        # Check directories were created
        assert self.output_dir.exists()
        assert (self.output_dir / "figs").exists()

    def test_on_epoch_end_basic(self):
        """Test basic epoch end callback functionality."""
        # Create mock evaluator
        evaluator = Mock()
        evaluator.args = Mock()
        evaluator.args.model = "test_method"
        evaluator.args.seed = 123

        # Mock timeline data
        evaluator.timeline_data = [{
            'epoch': 5,
            'task_id': 0,
            'subset_accuracies': {
                'T1_all': 0.85,
                'T2_shortcut_normal': 0.60,
                'T2_shortcut_masked': 0.45,
                'T2_nonshortcut_normal': 0.70
            },
            'timestamp': 1640995200.0
        }]

        # Call hook
        self.hooks.on_epoch_end(5, evaluator)

        # Verify data was stored
        assert len(self.hooks.timeline_data) == 1
        entry = self.hooks.timeline_data[0]
        assert entry['epoch'] == 5
        assert entry['method'] == "test_method"
        assert entry['seed'] == 123
        assert entry['subset_accuracies']['T2_shortcut_normal'] == 0.60

    def test_on_epoch_end_with_export(self):
        """Test epoch end callback with CSV export."""
        # Setup hooks with export frequency of 1
        hooks = ERIExperimentHooks(
            output_dir=str(self.output_dir),
            export_frequency=1
        )

        # Create mock evaluator
        evaluator = Mock()
        evaluator.args = Mock()
        evaluator.args.model = "sgd"
        evaluator.args.seed = 42

        # Mock timeline data for multiple epochs
        evaluator.timeline_data = [{
            'epoch': 1,
            'subset_accuracies': {
                'T1_all': 0.80,
                'T2_shortcut_normal': 0.50,
                'T2_shortcut_masked': 0.30
            }
        }]

        # Call hook
        hooks.on_epoch_end(1, evaluator)

        # Check that partial CSV was created
        partial_csv = self.output_dir / "eri_partial_epoch_1.csv"
        assert partial_csv.exists()

        # Verify CSV content
        df = pd.read_csv(partial_csv)
        assert len(df) == 3  # 3 valid splits
        assert 'method' in df.columns
        assert 'seed' in df.columns
        assert 'epoch_eff' in df.columns
        assert 'split' in df.columns
        assert 'acc' in df.columns

    def test_on_epoch_end_no_evaluator(self):
        """Test epoch end callback with no evaluator."""
        # Should not raise exception
        self.hooks.on_epoch_end(5, None)
        assert len(self.hooks.timeline_data) == 0

    def test_on_task_end(self):
        """Test task end callback."""
        # Add some timeline data first
        self.hooks.timeline_data = [{
            'epoch': 10,
            'method': 'test_method',
            'seed': 42,
            'subset_accuracies': {
                'T1_all': 0.85,
                'T2_shortcut_normal': 0.60
            }
        }]

        # Create mock evaluator
        evaluator = Mock()

        # Call hook
        self.hooks.on_task_end(1, evaluator)

        # Check task CSV was created
        task_csv = self.output_dir / "eri_task_1_timeline.csv"
        assert task_csv.exists()

        # Verify current task was updated
        assert self.hooks.current_task == 1

    def test_on_experiment_end(self):
        """Test experiment end callback."""
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
            },
            {
                'epoch': 10,
                'method': 'sgd',
                'seed': 42,
                'subset_accuracies': {
                    'T1_all': 0.88,
                    'T2_shortcut_normal': 0.65,
                    'T2_shortcut_masked': 0.50,
                    'T2_nonshortcut_normal': 0.72
                }
            }
        ]

        # Mock visualization generation to avoid complex dependencies
        with patch.object(self.hooks, '_generate_final_visualizations') as mock_viz:
            mock_viz.return_value = {
                'dynamics': '/path/to/dynamics.pdf',
                'heatmap': '/path/to/heatmap.pdf'
            }

            # Create mock evaluator
            evaluator = Mock()

            # Call hook
            result = self.hooks.on_experiment_end(evaluator)

            # Check results
            assert 'csv' in result
            assert 'metadata' in result
            assert 'dynamics' in result
            assert 'heatmap' in result

            # Check final CSV was created
            final_csv = self.output_dir / "timeline_sgd.csv"
            assert final_csv.exists()

            # Check metadata was created
            metadata_file = self.output_dir / "eri_experiment_metadata.json"
            assert metadata_file.exists()

    def test_export_timeline_csv(self):
        """Test CSV export functionality."""
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
            },
            {
                'epoch': 2,
                'epoch_eff': 2.0,
                'method': 'sgd',
                'seed': 42,
                'subset_accuracies': {
                    'T1_all': 0.82,
                    'T2_shortcut_normal': 0.55,
                    'T2_shortcut_masked': 0.35,
                    'T2_nonshortcut_normal': 0.62
                }
            }
        ]

        # Export CSV
        csv_path = self.output_dir / "test_timeline.csv"
        self.hooks._export_timeline_csv(csv_path)

        # Verify CSV was created and has correct format
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert len(df) == 8  # 2 epochs × 4 splits

        # Check required columns
        required_cols = ['method', 'seed', 'epoch_eff', 'split', 'acc', 'top5', 'loss']
        for col in required_cols:
            assert col in df.columns

        # Check data integrity
        assert df['method'].iloc[0] == 'sgd'
        assert df['seed'].iloc[0] == 42
        assert df['epoch_eff'].iloc[0] == 1.0

        # Check splits are valid
        valid_splits = {'T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal'}
        assert set(df['split'].unique()) == valid_splits

    def test_export_timeline_csv_empty_data(self):
        """Test CSV export with no data."""
        csv_path = self.output_dir / "empty_timeline.csv"

        # Should not raise exception
        self.hooks._export_timeline_csv(csv_path)

        # CSV should not be created
        assert not csv_path.exists()

    def test_get_timeline_summary(self):
        """Test timeline summary generation."""
        # Empty data
        summary = self.hooks.get_timeline_summary()
        assert summary['total_entries'] == 0
        assert summary['epoch_range'] == [0, 0]

        # With data
        self.hooks.timeline_data = [
            {'epoch': 1, 'method': 'sgd'},
            {'epoch': 5, 'method': 'sgd'},
            {'epoch': 10, 'method': 'sgd'}
        ]
        self.hooks.current_method = 'sgd'
        self.hooks.current_seed = 123
        self.hooks.current_task = 2

        summary = self.hooks.get_timeline_summary()
        assert summary['total_entries'] == 3
        assert summary['epoch_range'] == [1, 10]
        assert summary['method'] == 'sgd'
        assert summary['seed'] == 123
        assert summary['tasks'] == 3

    def test_reset(self):
        """Test hooks reset functionality."""
        # Add some data
        self.hooks.timeline_data = [{'epoch': 1}]
        self.hooks.current_task = 2
        self.hooks.last_export_epoch = 5

        # Reset
        self.hooks.reset()

        # Verify reset
        assert self.hooks.timeline_data == []
        assert self.hooks.current_task == 0
        assert self.hooks.last_export_epoch == -1

    def test_custom_style_config(self):
        """Test hooks with custom style configuration."""
        custom_style = PlotStyleConfig(
            figure_size=(10, 8),
            dpi=150,
            color_palette={'sgd': '#ff0000'}
        )

        hooks = ERIExperimentHooks(
            output_dir=str(self.output_dir),
            style_config=custom_style
        )

        assert hooks.style_config.figure_size == (10, 8)
        assert hooks.style_config.dpi == 150
        assert hooks.style_config.color_palette['sgd'] == '#ff0000'


class TestFactoryFunctions:
    """Test factory functions and integration utilities."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_eri_hooks(self):
        """Test factory function for creating hooks."""
        hooks = create_eri_hooks(
            output_dir=self.temp_dir,
            export_frequency=5,
            auto_visualize=False
        )

        assert isinstance(hooks, ERIExperimentHooks)
        assert hooks.export_frequency == 5
        assert hooks.auto_visualize is False
        assert str(hooks.output_dir) == self.temp_dir

    def test_integrate_hooks_with_evaluator(self):
        """Test integration with evaluator."""
        # Create mock evaluator
        evaluator = Mock()
        evaluator.after_training_epoch = Mock()
        evaluator.meta_end_task = Mock()

        # Create hooks
        hooks = create_eri_hooks(output_dir=self.temp_dir)

        # Integrate
        integrate_hooks_with_evaluator(evaluator, hooks)

        # Verify methods were patched
        assert hasattr(evaluator, 'after_training_epoch')
        assert hasattr(evaluator, 'meta_end_task')

        # Test that calling the patched methods works
        mock_model = Mock()
        mock_dataset = Mock()
        mock_dataset.i = 1
        mock_dataset.N_TASKS = 2

        # Should not raise exceptions
        evaluator.after_training_epoch(mock_model, mock_dataset, 5)
        evaluator.meta_end_task(mock_model, mock_dataset)

    def test_integrate_hooks_with_none_evaluator(self):
        """Test integration with None evaluator raises error."""
        hooks = create_eri_hooks(output_dir=self.temp_dir)

        with pytest.raises(ValueError, match="Evaluator cannot be None"):
            integrate_hooks_with_evaluator(None, hooks)


class TestErrorHandling:
    """Test error handling in hooks."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.hooks = ERIExperimentHooks(output_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_on_epoch_end_with_malformed_evaluator(self):
        """Test epoch end with malformed evaluator data."""
        # Create evaluator with missing attributes
        evaluator = Mock()
        evaluator.timeline_data = None  # Missing timeline data

        # Should not raise exception
        self.hooks.on_epoch_end(5, evaluator)
        assert len(self.hooks.timeline_data) == 0

    def test_on_epoch_end_with_exception_in_evaluator(self):
        """Test epoch end when evaluator raises exception."""
        # Create evaluator that raises exception
        evaluator = Mock()
        evaluator.args = Mock()
        evaluator.args.model = "test"
        evaluator.timeline_data = Mock()
        evaluator.timeline_data.__getitem__ = Mock(side_effect=Exception("Test error"))

        # Should not raise exception
        self.hooks.on_epoch_end(5, evaluator)
        assert len(self.hooks.timeline_data) == 0

    def test_export_csv_with_invalid_path(self):
        """Test CSV export with invalid path."""
        # Try to export to a path that would require root permissions
        invalid_path = Path("/root/test.csv")

        # Add some data
        self.hooks.timeline_data = [{
            'epoch': 1,
            'method': 'test',
            'seed': 42,
            'subset_accuracies': {'T1_all': 0.8}
        }]

        # Should not raise exception (error should be logged)
        self.hooks._export_timeline_csv(invalid_path)


if __name__ == '__main__':
    pytest.main([__file__])
