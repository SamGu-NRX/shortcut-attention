#!/usr/bin/env python3
"""
Unit tests for comparative results aggregation functionality.

Tests the CSV aggregation pipeline including file discovery, validation,
and merging of multiple experiment results for comparative analysis.
"""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

# Import the functions we're testing
import sys
sys.path.append(str(Path(__file__).parent.parent))

from run_einstellung_experiment import (
    aggregate_comparative_results,
    find_csv_file,
    validate_csv_file
)


class TestComparativeAggregation(unittest.TestCase):
    """Test suite for comparative results aggregation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_sample_csv(self, method: str, seed: int, output_dir: Path) -> str:
        """Create a sample ERI CSV file for testing."""
        # Create realistic ERI data
        epochs = np.linspace(0.1, 2.0, 10)
        data_rows = []

        for epoch in epochs:
            # Generate realistic accuracy patterns
            base_acc = 0.6 + 0.2 * (1 - np.exp(-epoch))
            noise = np.random.normal(0, 0.02)

            data_rows.extend([
                [method, seed, epoch, "T1_all", max(0, min(1, base_acc * 0.8 + noise))],
                [method, seed, epoch, "T2_shortcut_normal", max(0, min(1, base_acc * 1.1 + noise))],
                [method, seed, epoch, "T2_shortcut_masked", max(0, min(1, base_acc * 0.9 + noise))],
                [method, seed, epoch, "T2_nonshortcut_normal", max(0, min(1, base_acc + noise))]
            ])

        df = pd.DataFrame(data_rows, columns=["method", "seed", "epoch_eff", "split", "acc"])

        # Create output directory and save CSV
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "eri_sc_metrics.csv"
        df.to_csv(csv_path, index=False)

        return str(csv_path)

    def create_invalid_csv(self, output_dir: Path) -> str:
        """Create an invalid CSV file for testing error handling."""
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "invalid.csv"

        # Create CSV with wrong columns
        df = pd.DataFrame({
            'wrong_col1': [1, 2, 3],
            'wrong_col2': ['a', 'b', 'c']
        })
        df.to_csv(csv_path, index=False)

        return str(csv_path)

    def test_validate_csv_file_valid(self):
        """Test CSV validation with valid file."""
        csv_path = self.create_sample_csv("sgd", 42, self.temp_path / "test1")
        self.assertTrue(validate_csv_file(csv_path))

    def test_validate_csv_file_invalid_columns(self):
        """Test CSV validation with invalid columns."""
        csv_path = self.create_invalid_csv(self.temp_path / "invalid")
        self.assertFalse(validate_csv_file(csv_path))

    def test_validate_csv_file_nonexistent(self):
        """Test CSV validation with nonexistent file."""
        self.assertFalse(validate_csv_file("/nonexistent/file.csv"))

    def test_validate_csv_file_empty(self):
        """Test CSV validation with empty file."""
        empty_path = self.temp_path / "empty.csv"
        empty_path.touch()
        self.assertFalse(validate_csv_file(str(empty_path)))

    def test_find_csv_file_eri_pattern(self):
        """Test CSV file discovery with ERI-specific patterns."""
        # Create a valid ERI CSV
        output_dir = self.temp_path / "experiment1"
        csv_path = self.create_sample_csv("sgd", 42, output_dir)

        # Should find the ERI CSV
        found_path = find_csv_file(str(output_dir))
        self.assertEqual(found_path, csv_path)

    def test_find_csv_file_no_valid_files(self):
        """Test CSV file discovery when no valid files exist."""
        # Create directory with invalid CSV
        output_dir = self.temp_path / "invalid_exp"
        self.create_invalid_csv(output_dir)

        # Should return None
        found_path = find_csv_file(str(output_dir))
        self.assertIsNone(found_path)

    def test_find_csv_file_nonexistent_dir(self):
        """Test CSV file discovery with nonexistent directory."""
        found_path = find_csv_file("/nonexistent/directory")
        self.assertIsNone(found_path)

    def test_aggregate_comparative_results_success(self):
        """Test successful aggregation of multiple CSV files."""
        # Create multiple experiment results
        results_list = []

        for i, method in enumerate(['sgd', 'derpp', 'ewc_on']):
            output_dir = self.temp_path / f"experiment_{method}"
            csv_path = self.create_sample_csv(method, 42, output_dir)

            results_list.append({
                'strategy': method,
                'backbone': 'resnet18',
                'seed': 42,
                'success': True,
                'output_dir': str(output_dir)
            })

        # Test aggregation
        output_dir = str(self.temp_path / "comparative")
        aggregated_path = aggregate_comparative_results(results_list, output_dir)

        # Verify aggregated file exists and is valid
        self.assertTrue(os.path.exists(aggregated_path))
        self.assertTrue(validate_csv_file(aggregated_path))

        # Load and verify content
        df = pd.read_csv(aggregated_path)

        # Should have data from all three methods
        methods = sorted(df['method'].unique())
        self.assertEqual(methods, ['derpp', 'ewc_on', 'sgd'])

        # Should have all required splits
        splits = set(df['split'].unique())
        expected_splits = {"T1_all", "T2_shortcut_normal", "T2_shortcut_masked", "T2_nonshortcut_normal"}
        self.assertEqual(splits, expected_splits)

    def test_aggregate_comparative_results_no_valid_files(self):
        """Test aggregation when no valid CSV files are found."""
        # Create results with no valid CSV files
        results_list = [
            {
                'strategy': 'sgd',
                'success': True,
                'output_dir': str(self.temp_path / "nonexistent")
            }
        ]

        output_dir = str(self.temp_path / "comparative")

        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            aggregate_comparative_results(results_list, output_dir)

        self.assertIn("No valid CSV files found", str(context.exception))

    def test_aggregate_comparative_results_mixed_success(self):
        """Test aggregation with mix of successful and failed experiments."""
        results_list = []

        # Add successful experiment
        output_dir1 = self.temp_path / "experiment_sgd"
        self.create_sample_csv("sgd", 42, output_dir1)
        results_list.append({
            'strategy': 'sgd',
            'success': True,
            'output_dir': str(output_dir1)
        })

        # Add failed experiment
        results_list.append({
            'strategy': 'derpp',
            'success': False,
            'output_dir': str(self.temp_path / "failed")
        })

        # Add successful experiment with invalid CSV
        output_dir2 = self.temp_path / "experiment_invalid"
        self.create_invalid_csv(output_dir2)
        results_list.append({
            'strategy': 'ewc_on',
            'success': True,
            'output_dir': str(output_dir2)
        })

        # Should still work with the one valid experiment
        output_dir = str(self.temp_path / "comparative")
        aggregated_path = aggregate_comparative_results(results_list, output_dir)

        # Verify only SGD data is present
        df = pd.read_csv(aggregated_path)
        methods = df['method'].unique()
        self.assertEqual(list(methods), ['sgd'])

    def test_aggregate_comparative_results_duplicate_removal(self):
        """Test that duplicate entries are properly removed during aggregation."""
        # Create two experiments with overlapping data (same method, seed, epochs)
        results_list = []

        # First experiment
        output_dir1 = self.temp_path / "experiment_sgd_1"
        self.create_sample_csv("sgd", 42, output_dir1)
        results_list.append({
            'strategy': 'sgd',
            'success': True,
            'output_dir': str(output_dir1)
        })

        # Second experiment with same method/seed (simulating re-run)
        output_dir2 = self.temp_path / "experiment_sgd_2"
        self.create_sample_csv("sgd", 42, output_dir2)
        results_list.append({
            'strategy': 'sgd',
            'success': True,
            'output_dir': str(output_dir2)
        })

        # Aggregate
        output_dir_agg = str(self.temp_path / "comparative")
        aggregated_path = aggregate_comparative_results(results_list, output_dir_agg)

        # Verify duplicates were removed
        df_aggregated = pd.read_csv(aggregated_path)

        # Should have no duplicates in final result
        duplicates = df_aggregated.duplicated(subset=['method', 'seed', 'epoch_eff', 'split'])
        self.assertFalse(duplicates.any())

        # Should have data for only one SGD experiment (duplicates removed)
        methods = df_aggregated['method'].unique()
        self.assertEqual(list(methods), ['sgd'])

    def test_aggregate_comparative_results_baseline_methods(self):
        """Test aggregation including baseline methods."""
        results_list = []

        # Add baseline methods
        for method in ['scratch_t2', 'interleaved']:
            output_dir = self.temp_path / f"experiment_{method}"
            self.create_sample_csv(method, 42, output_dir)
            results_list.append({
                'strategy': method,
                'success': True,
                'output_dir': str(output_dir)
            })

        # Add continual learning method
        output_dir = self.temp_path / "experiment_derpp"
        self.create_sample_csv("derpp", 42, output_dir)
        results_list.append({
            'strategy': 'derpp',
            'success': True,
            'output_dir': str(output_dir)
        })

        # Aggregate
        output_dir_agg = str(self.temp_path / "comparative")
        aggregated_path = aggregate_comparative_results(results_list, output_dir_agg)

        # Verify all methods are present
        df = pd.read_csv(aggregated_path)
        methods = sorted(df['method'].unique())
        self.assertEqual(methods, ['derpp', 'interleaved', 'scratch_t2'])

    def test_eri_visualization_integration(self):
        """Test that aggregated data works with ERI visualization system."""
        # Create sample data
        results_list = []
        for method in ['sgd', 'derpp']:
            output_dir = self.temp_path / f"experiment_{method}"
            self.create_sample_csv(method, 42, output_dir)
            results_list.append({
                'strategy': method,
                'success': True,
                'output_dir': str(output_dir)
            })

        # Aggregate
        output_dir_agg = str(self.temp_path / "comparative")
        aggregated_path = aggregate_comparative_results(results_list, output_dir_agg)

        # Test that ERI data loader can load the aggregated file
        try:
            from eri_vis.data_loader import ERIDataLoader
            loader = ERIDataLoader()
            dataset = loader.load_csv(aggregated_path)

            # Verify dataset properties
            self.assertEqual(len(dataset.methods), 2)
            self.assertIn('sgd', dataset.methods)
            self.assertIn('derpp', dataset.methods)
            self.assertEqual(len(dataset.splits), 4)

        except ImportError:
            # Skip test if ERI visualization not available
            self.skipTest("ERI visualization system not available")


if __name__ == '__main__':
    # Set random seed for reproducible tests
    np.random.seed(42)

    unittest.main()
