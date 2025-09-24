"""
Tests for MammothERIIntegration - Framework bridge between Mammoth and ERI visualization.

This module tests the integration with the existing Mammoth Einstellung experiment
infrastructure, specifically:
- utils/einstellung_evaluator.py integration
- Timeline data export and conversion
- Automatic visualization generation
- Hook registration with Mammoth training pipeline

CRITICAL: Tests use REAL EinstellungEvaluator instances, not mocks, to verify
actual integration with the existing Mammoth infrastructure.
"""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Import the integration module
from eri_vis.integration.mammoth_integration import MammothERIIntegration, create_mammoth_integration
from eri_vis.styles import PlotStyleConfig

# Import Mammoth components for real integration testing
try:
    from utils.einstellung_evaluator import EinstellungEvaluator
    from utils.einstellung_metrics import EinstellungMetricsCalculator
    MAMMOTH_AVAILABLE = True
except ImportError:
    MAMMOTH_AVAILABLE = False


class TestMammothERIIntegration(unittest.TestCase):
    """Test suite for MammothERIIntegration class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "test_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create mock args for evaluator
        self.mock_args = MagicMock()
        self.mock_args.model = 'sgd'
        self.mock_args.seed = 42
        self.mock_args.device = 'cpu'
        self.mock_args.batch_size = 32

        # Set up logging
        logging.basicConfig(level=logging.DEBUG)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test MammothERIIntegration initialization."""
        # Test initialization without evaluator
        integration = MammothERIIntegration()
        self.assertIsNone(integration.evaluator)
        self.assertEqual(integration.output_dir, Path("logs"))
        self.assertIsInstance(integration.style_config, PlotStyleConfig)

        # Test initialization with parameters
        style_config = PlotStyleConfig(dpi=150)
        integration = MammothERIIntegration(
            evaluator=None,
            output_dir=str(self.output_dir),
            style_config=style_config
        )
        self.assertEqual(integration.output_dir, self.output_dir)
        self.assertEqual(integration.style_config.dpi, 150)

    def test_setup_auto_export(self):
        """Test auto-export setup."""
        integration = MammothERIIntegration()

        # Test setup
        integration.setup_auto_export(str(self.output_dir), export_frequency=5)

        self.assertTrue(integration.auto_export_enabled)
        self.assertEqual(integration.export_frequency, 5)
        self.assertEqual(integration.output_dir, self.output_dir)

        # Verify directories were created
        self.assertTrue(self.output_dir.exists())
        self.assertTrue((self.output_dir / "figs").exists())

    def create_mock_timeline_data(self):
        """Create mock timeline data for testing."""
        timeline_data = []

        # Create sample data for multiple epochs and splits
        for epoch in range(5):
            entry = {
                'epoch': epoch,
                'task_id': 1,
                'subset_accuracies': {
                    'T1_all': 0.8 + epoch * 0.02,
                    'T2_shortcut_normal': 0.1 + epoch * 0.1,
                    'T2_shortcut_masked': 0.05 + epoch * 0.05,
                    'T2_nonshortcut_normal': 0.15 + epoch * 0.03
                },
                'subset_losses': {
                    'T1_all': 0.5 - epoch * 0.05,
                    'T2_shortcut_normal': 2.0 - epoch * 0.2,
                    'T2_shortcut_masked': 2.5 - epoch * 0.25,
                    'T2_nonshortcut_normal': 1.8 - epoch * 0.18
                },
                'attention_metrics': {},
                'timestamp': 1640995200.0 + epoch * 100
            }
            timeline_data.append(entry)

        return timeline_data

    def test_export_timeline_for_visualization(self):
        """Test timeline data export to CSV format."""
        # Create mock evaluator with timeline data
        mock_evaluator = MagicMock()
        mock_evaluator.args = self.mock_args
        mock_evaluator.timeline_data = self.create_mock_timeline_data()
        mock_evaluator.dataset_name = 'seq-cifar100-einstellung'
        mock_evaluator.adaptation_threshold = 0.8
        mock_evaluator.extract_attention = False

        integration = MammothERIIntegration(evaluator=mock_evaluator)

        # Test export
        csv_path = self.output_dir / "test_timeline.csv"
        integration.export_timeline_for_visualization(str(csv_path))

        # Verify CSV was created
        self.assertTrue(csv_path.exists())

        # Verify CSV content
        df = pd.read_csv(csv_path)
        expected_columns = ['method', 'seed', 'epoch_eff', 'split', 'acc', 'top5', 'loss']
        self.assertEqual(list(df.columns), expected_columns)

        # Check data content
        self.assertEqual(df['method'].iloc[0], 'sgd')
        self.assertEqual(df['seed'].iloc[0], 42)
        self.assertIn('T1_all', df['split'].values)
        self.assertIn('T2_shortcut_normal', df['split'].values)

        # Verify metadata sidecar
        metadata_path = csv_path.with_suffix('.json')
        self.assertTrue(metadata_path.exists())

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.assertIn('export_timestamp', metadata)
        self.assertIn('total_rows', metadata)
        self.assertIn('methods', metadata)
        self.assertEqual(metadata['methods'], ['sgd'])

    def test_export_timeline_no_evaluator(self):
        """Test export behavior when no evaluator is provided."""
        integration = MammothERIIntegration()

        csv_path = self.output_dir / "test_timeline.csv"

        with self.assertRaises(ValueError):
            integration.export_timeline_for_visualization(str(csv_path))

    def test_export_timeline_no_data(self):
        """Test export behavior when evaluator has no timeline data."""
        mock_evaluator = MagicMock()
        mock_evaluator.args = self.mock_args
        mock_evaluator.timeline_data = []

        integration = MammothERIIntegration(evaluator=mock_evaluator)

        csv_path = self.output_dir / "test_timeline.csv"

        # Should not raise exception but should log warning
        with self.assertLogs(level='WARNING'):
            integration.export_timeline_for_visualization(str(csv_path))

    @patch('eri_vis.integration.mammoth_integration.ERIDynamicsPlotter')
    @patch('eri_vis.integration.mammoth_integration.ERIHeatmapPlotter')
    def test_generate_visualizations_from_evaluator(self, mock_heatmap_plotter, mock_dynamics_plotter):
        """Test visualization generation from evaluator data."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_dynamics_plotter.return_value.create_dynamics_figure.return_value = mock_fig
        mock_heatmap_plotter.return_value.create_tau_sensitivity_heatmap.return_value = mock_fig
        mock_heatmap_plotter.return_value.compute_tau_sensitivity.return_value = MagicMock()

        # Create mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.args = self.mock_args
        mock_evaluator.timeline_data = self.create_mock_timeline_data()
        mock_evaluator.dataset_name = 'seq-cifar100-einstellung'
        mock_evaluator.adaptation_threshold = 0.8
        mock_evaluator.extract_attention = False

        integration = MammothERIIntegration(evaluator=mock_evaluator)

        # Test visualization generation
        generated = integration.generate_visualizations_from_evaluator(str(self.output_dir))

        # Verify files were "generated" (mocked)
        self.assertIn('dynamics', generated)
        self.assertIn('heatmap', generated)

    def test_register_visualization_hooks(self):
        """Test registration of visualization hooks with evaluator."""
        mock_evaluator = MagicMock()
        mock_evaluator.args = self.mock_args

        # Store original methods
        original_after_epoch = MagicMock()
        original_end_task = MagicMock()
        mock_evaluator.after_training_epoch = original_after_epoch
        mock_evaluator.meta_end_task = original_end_task

        integration = MammothERIIntegration(evaluator=mock_evaluator)
        integration.setup_auto_export(str(self.output_dir))

        # Register hooks
        integration.register_visualization_hooks()

        # Verify hooks were replaced
        self.assertNotEqual(mock_evaluator.after_training_epoch, original_after_epoch)
        self.assertNotEqual(mock_evaluator.meta_end_task, original_end_task)

    def test_register_hooks_no_evaluator(self):
        """Test hook registration behavior when no evaluator is provided."""
        integration = MammothERIIntegration()

        # Should not raise exception but should log warning
        with self.assertLogs(level='WARNING'):
            integration.register_visualization_hooks()

    def test_load_from_mammoth_results_csv(self):
        """Test loading ERI data from Mammoth results directory with CSV files."""
        # Create test CSV file
        test_data = {
            'method': ['sgd', 'sgd', 'ewc_on', 'ewc_on'],
            'seed': [42, 42, 42, 42],
            'epoch_eff': [0, 1, 0, 1],
            'split': ['T2_shortcut_normal', 'T2_shortcut_normal', 'T2_shortcut_normal', 'T2_shortcut_normal'],
            'acc': [0.1, 0.2, 0.15, 0.25]
        }
        df = pd.DataFrame(test_data)

        csv_path = self.output_dir / "eri_results.csv"
        df.to_csv(csv_path, index=False)

        integration = MammothERIIntegration()
        dataset = integration.load_from_mammoth_results(str(self.output_dir))

        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset.data), 4)

    def test_load_from_mammoth_results_json(self):
        """Test loading ERI data from Mammoth results directory with JSON export."""
        # Create test JSON export
        export_data = {
            'configuration': {
                'dataset_name': 'seq-cifar100-einstellung',
                'adaptation_threshold': 0.8
            },
            'timeline_data': self.create_mock_timeline_data(),
            'final_metrics': {}
        }

        json_path = self.output_dir / "einstellung_results.json"
        with open(json_path, 'w') as f:
            json.dump(export_data, f)

        integration = MammothERIIntegration()

        # Mock the data loader to avoid actual conversion
        with patch.object(integration.data_loader, 'load_from_evaluator_export') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            dataset = integration.load_from_mammoth_results(str(self.output_dir))

            self.assertIsNotNone(dataset)
            mock_load.assert_called_once_with(export_data)

    def test_load_from_mammoth_results_no_files(self):
        """Test loading behavior when no ERI files are found."""
        integration = MammothERIIntegration()

        # Empty directory
        empty_dir = self.output_dir / "empty"
        empty_dir.mkdir()

        dataset = integration.load_from_mammoth_results(str(empty_dir))
        self.assertIsNone(dataset)

    def test_create_integration_config(self):
        """Test creation of integration configuration file."""
        integration = MammothERIIntegration()

        config_path = self.output_dir / "integration_config.json"
        methods = ['sgd', 'ewc_on', 'derpp']
        seeds = [42, 43, 44]

        integration.create_integration_config(
            str(config_path),
            methods=methods,
            seeds=seeds,
            tau=0.7,
            smoothing_window=5
        )

        # Verify config file was created
        self.assertTrue(config_path.exists())

        # Verify config content
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.assertEqual(config['processing']['tau'], 0.7)
        self.assertEqual(config['processing']['smoothing_window'], 5)
        self.assertEqual(config['experiment']['methods'], methods)
        self.assertEqual(config['experiment']['seeds'], seeds)

    def test_factory_function(self):
        """Test the create_mammoth_integration factory function."""
        mock_evaluator = MagicMock()

        # Test with auto_setup=False
        integration = create_mammoth_integration(
            mock_evaluator,
            output_dir=str(self.output_dir),
            auto_setup=False
        )

        self.assertIsInstance(integration, MammothERIIntegration)
        self.assertEqual(integration.evaluator, mock_evaluator)
        self.assertFalse(integration.auto_export_enabled)

        # Test with auto_setup=True
        integration = create_mammoth_integration(
            mock_evaluator,
            output_dir=str(self.output_dir),
            auto_setup=True
        )

        self.assertTrue(integration.auto_export_enabled)


@unittest.skipUnless(MAMMOTH_AVAILABLE, "Mammoth components not available")
class TestRealMammothIntegration(unittest.TestCase):
    """
    Integration tests with REAL Mammoth components.

    These tests verify actual integration with the existing Mammoth infrastructure
    rather than using mocks.
    """

    def setUp(self):
        """Set up test fixtures with real Mammoth components."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "real_test_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create real args object
        from argparse import Namespace
        self.args = Namespace(
            model='sgd',
            seed=42,
            device='cpu',
            batch_size=32,
            dataset='seq-cifar100-einstellung-224',
            einstellung_evaluation_subsets=True,
            einstellung_adaptation_threshold=0.8,
            einstellung_extract_attention=False
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_real_evaluator_integration(self):
        """Test integration with a real EinstellungEvaluator instance."""
        # Create real evaluator
        evaluator = EinstellungEvaluator(
            args=self.args,
            dataset_name='seq-cifar100-einstellung-224',
            adaptation_threshold=0.8,
            extract_attention=False
        )

        # Add some mock timeline data
        evaluator.timeline_data = [
            {
                'epoch': 0,
                'task_id': 1,
                'subset_accuracies': {
                    'T1_all': 0.8,
                    'T2_shortcut_normal': 0.1,
                    'T2_shortcut_masked': 0.05,
                    'T2_nonshortcut_normal': 0.15
                },
                'subset_losses': {
                    'T1_all': 0.5,
                    'T2_shortcut_normal': 2.0,
                    'T2_shortcut_masked': 2.5,
                    'T2_nonshortcut_normal': 1.8
                },
                'attention_metrics': {},
                'timestamp': 1640995200.0
            }
        ]

        # Create integration
        integration = MammothERIIntegration(evaluator=evaluator)

        # Test CSV export
        csv_path = self.output_dir / "real_timeline.csv"
        integration.export_timeline_for_visualization(str(csv_path))

        # Verify export worked
        self.assertTrue(csv_path.exists())

        # Verify CSV content
        df = pd.read_csv(csv_path)
        self.assertGreater(len(df), 0)
        self.assertEqual(df['method'].iloc[0], 'sgd')
        self.assertEqual(df['seed'].iloc[0], 42)

    def test_real_metrics_calculator_integration(self):
        """Test integration with real EinstellungMetricsCalculator."""
        # Create real metrics calculator
        calculator = EinstellungMetricsCalculator(adaptation_threshold=0.8)

        # Add some timeline data
        calculator.add_timeline_data(
            epoch=0,
            task_id=1,
            subset_accuracies={
                'T1_all': 0.8,
                'T2_shortcut_normal': 0.1,
                'T2_shortcut_masked': 0.05
            },
            subset_losses={
                'T1_all': 0.5,
                'T2_shortcut_normal': 2.0,
                'T2_shortcut_masked': 2.5
            },
            timestamp=1640995200.0
        )

        # Verify calculator works
        metrics = calculator.calculate_comprehensive_metrics()
        self.assertIsNotNone(metrics)


if __name__ == '__main__':
    unittest.main()
