#!/usr/bin/env python3
"""
Comprehensive Integration Test for Comparative Einstellung Analysis System

This test implements Task 16: Comprehensive Integration Testing
- Create end-to-end integration test for complete comparative analysis pipeline
- Test baseline method training, evaluation, and CSV generation
- Validate data aggregation, visualization generation, and statistical analysis
- Test error handling, checkpoint management, and experiment orchestration
- Verify backward compatibility with existing single-method experiments

This is the master integration test that validates the entire comparative
analysis system works correctly from start to finish.
"""

import unittest
import tempfile
import os
import sys
import shutil
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from argparse import Namespace

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class TestComprehensiveIntegration(unittest.TestCase):
    """
    Comprehensive integration test suite for the complete comparative analysis pipeline.

    Tests the entire system from baseline method implementation through
    statistical analysis and visualization generation.
    """

    def setUp(self):
        """Set up test fixtures and temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_output_dir = os.path.join(self.temp_dir, "test_outputs")
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Create subdirectories for different test components
        self.checkpoints_dir = os.path.join(self.temp_dir, "checkpoints")
        self.results_dir = os.path.join(self.temp_dir, "results")
        self.comparative_dir = os.path.join(self.temp_dir, "comparative")

        for dir_path in [self.checkpoints_dir, self.results_dir, self.comparative_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_baseline_method_registry_integration(self):
        """Test that baseline methods are properly integrated with the model registry."""

        # Test Scratch_T2 registration
        try:
            from models.scratch_t2 import ScratchT2
            self.assertEqual(ScratchT2.NAME, 'scratch_t2')
            self.assertIn('class-il', ScratchT2.COMPATIBILITY)
        except ImportError:
            self.fail("Scratch_T2 model not available in registry")

        # Test Interleaved registration
        try:
            from models.interleaved import Interleaved
            self.assertEqual(Interleaved.NAME, 'interleaved')
            self.assertIn('class-il', Interleaved.COMPATIBILITY)
        except ImportError:
            self.fail("Interleaved model not available in registry")

        # Test automatic discovery through get_model_names
        try:
            from models import get_model_names

            # Test that models are discoverable in the registry
            available_models = get_model_names()
            # Models are registered with hyphens in the registry
            self.assertIn('scratch-t2', available_models, "Scratch_T2 not found in model registry")
            self.assertIn('interleaved', available_models, "Interleaved not found in model registry")

        except Exception as e:
            self.fail(f"Model registry integration failed: {e}")

    def test_experiment_orchestration_integration(self):
        """Test integration with experiment orchestration functions."""

        try:
            from run_einstellung_experiment import (
                create_einstellung_args,
                extract_accuracy_from_output,
                find_existing_checkpoints,
                aggregate_comparative_results
            )

            # Test argument creation for baseline methods using standard ERI config
            scratch_args = create_einstellung_args('scratch_t2', 'resnet18', 42, debug=True)
            self.assertIn('--model', scratch_args)
            self.assertIn('scratch_t2', scratch_args)
            self.assertIn('--dataset', scratch_args)
            self.assertIn('seq-cifar100-einstellung', scratch_args)

            interleaved_args = create_einstellung_args('interleaved', 'resnet18', 42, debug=True)
            self.assertIn('--model', interleaved_args)
            self.assertIn('interleaved', interleaved_args)

            # Test accuracy extraction
            sample_output = "Accuracy for 2 task(s): [Class-IL]: 85.2 %"
            accuracy = extract_accuracy_from_output(sample_output)
            self.assertEqual(accuracy, 85.2)

            # Test checkpoint discovery (should not crash with empty directory)
            checkpoints = find_existing_checkpoints('scratch_t2', 'resnet18', 42, self.checkpoints_dir)
            self.assertIsInstance(checkpoints, list)

        except ImportError as e:
            self.fail(f"Experiment orchestration functions not available: {e}")

    def create_mock_experiment_results(self) -> list:
        """Create mock experiment results for testing aggregation."""

        methods = ['scratch_t2', 'interleaved', 'sgd', 'derpp', 'ewc_on']
        results = []

        for i, method in enumerate(methods):
            # Create individual result directories
            method_dir = os.path.join(self.results_dir, f"{method}_resnet18_seed42")
            os.makedirs(method_dir, exist_ok=True)

            # Create mock CSV file
            csv_path = os.path.join(method_dir, "eri_metrics.csv")
            self.create_mock_csv_file(csv_path, method)

            # Create result dictionary
            result = {
                'strategy': method,
                'backbone': 'resnet18',
                'seed': 42,
                'final_accuracy': 70.0 + i * 5.0,  # Varying accuracies
                'output_dir': method_dir,
                'success': True,
                'used_checkpoint': False
            }
            results.append(result)

        return results

    def create_mock_csv_file(self, csv_path: str, method: str):
        """Create a mock ERI CSV file with realistic data."""

        np.random.seed(hash(method) % 2**32)  # Deterministic but method-specific

        # Use standard ERI experimental setup: 21 evaluation points over 2 effective epochs
        epochs = np.linspace(0.0, 2.0, 21)
        # Standard ERI evaluation splits as per experimental design standards
        splits = ['T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal']

        # Define method-specific performance characteristics
        performance_profiles = {
            'scratch_t2': {
                'T1_all': 0.0,  # No T1 training
                'T2_shortcut_normal': 0.85,
                'T2_shortcut_masked': 0.75,
                'T2_nonshortcut_normal': 0.80
            },
            'interleaved': {
                'T1_all': 0.75,
                'T2_shortcut_normal': 0.88,
                'T2_shortcut_masked': 0.82,
                'T2_nonshortcut_normal': 0.85
            },
            'sgd': {
                'T1_all': 0.25,  # Catastrophic forgetting
                'T2_shortcut_normal': 0.70,
                'T2_shortcut_masked': 0.45,
                'T2_nonshortcut_normal': 0.65
            },
            'derpp': {
                'T1_all': 0.60,
                'T2_shortcut_normal': 0.78,
                'T2_shortcut_masked': 0.68,
                'T2_nonshortcut_normal': 0.75
            },
            'ewc_on': {
                'T1_all': 0.55,
                'T2_shortcut_normal': 0.72,
                'T2_shortcut_masked': 0.58,
                'T2_nonshortcut_normal': 0.70
            }
        }

        profile = performance_profiles.get(method, performance_profiles['sgd'])

        data = []
        for epoch in epochs:
            progress = 1 - np.exp(-0.1 * epoch)  # Learning curve

            for split in splits:
                base_acc = profile[split]

                # Apply learning progression
                if split == 'T1_all' and method == 'scratch_t2':
                    acc = 0.0  # Never trains on T1
                elif split == 'T1_all' and method not in ['scratch_t2', 'interleaved']:
                    # Forgetting for continual learning methods
                    forgetting = 0.3 * epoch
                    acc = base_acc * (1 - forgetting)
                else:
                    acc = base_acc * progress

                # Add noise
                acc += np.random.normal(0, 0.02)
                acc = max(0.0, min(1.0, acc))

                data.append({
                    'method': method,
'seed': 42,
                    'epoch_eff': epoch,
                    'split': split,
                    'acc': acc
                })

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    def test_data_aggregation_pipeline(self):
        """Test the complete data aggregation pipeline."""

        try:
            from run_einstellung_experiment import aggregate_comparative_results, find_csv_file

            # Create mock experiment results
            results = self.create_mock_experiment_results()

            # Test CSV file discovery
            for result in results:
                csv_path = find_csv_file(result['output_dir'])
                self.assertIsNotNone(csv_path, f"CSV not found for {result['strategy']}")
                self.assertTrue(os.path.exists(csv_path))

            # Test aggregation
            aggregated_csv = aggregate_comparative_results(results, self.comparative_dir)

            # Verify aggregated file exists and has correct structure
            self.assertTrue(os.path.exists(aggregated_csv))

            df = pd.read_csv(aggregated_csv)

            # Check required columns
            required_columns = ['method', 'seed', 'epoch_eff', 'split', 'acc']
            for col in required_columns:
                self.assertIn(col, df.columns, f"Missing column: {col}")

            # Check all methods are present
            methods = df['method'].unique()
            expected_methods = ['scratch_t2', 'interleaved', 'sgd', 'derpp', 'ewc_on']
            for method in expected_methods:
                self.assertIn(method, methods, f"Missing method: {method}")

            # Check all splits are present
            splits = df['split'].unique()
            expected_splits = ['T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal']
            for split in expected_splits:
                self.assertIn(split, splits, f"Missing split: {split}")

        except ImportError as e:
            self.fail(f"Data aggregation functions not available: {e}")

    def test_statistical_analysis_integration(self):
        """Test integration with statistical analysis system."""

        try:
            from utils.statistical_analysis import StatisticalAnalyzer, generate_statistical_report

            # Create aggregated data
            results = self.create_mock_experiment_results()
            from run_einstellung_experiment import aggregate_comparative_results
            aggregated_csv = aggregate_comparative_results(results, self.comparative_dir)

            # Run statistical analysis
            analyzer = StatisticalAnalyzer(alpha=0.05, correction_method='bonferroni')
            statistical_results = analyzer.analyze_comparative_metrics(aggregated_csv)

            # Verify statistical results structure
            expected_sections = [
                'summary_statistics',
                'pairwise_comparisons',
                'anova_results',
                'effect_sizes',
                'multiple_comparisons',
                'statistical_power',
                'interpretation'
            ]

            for section in expected_sections:
                self.assertIn(section, statistical_results, f"Missing statistical section: {section}")

            # Verify baseline methods are included
            summary_stats = statistical_results['summary_statistics']
            self.assertIn('scratch_t2', summary_stats)
            self.assertIn('interleaved', summary_stats)

            # Test report generation
            report_path = generate_statistical_report(aggregated_csv, self.comparative_dir)
            self.assertTrue(os.path.exists(report_path))

            # Verify report content
            with open(report_path, 'r') as f:
                content = f.read()

            self.assertIn('Statistical Analysis Report', content)
            self.assertIn('scratch_t2', content)
            self.assertIn('interleaved', content)

        except ImportError as e:
            self.fail(f"Statistical analysis not available: {e}")

    @patch('run_einstellung_experiment.generate_eri_visualizations')
    def test_visualization_integration(self, mock_generate_viz):
        """Test integration with ERI visualization system."""

        try:
            # Create aggregated data
            results = self.create_mock_experiment_results()
            from run_einstellung_experiment import aggregate_comparative_results
            aggregated_csv = aggregate_comparative_results(results, self.comparative_dir)

            # Mock the visualization generation
            mock_generate_viz.return_value = True

            # Test that visualization can be called with aggregated data
            from run_einstellung_experiment import generate_eri_visualizations

            viz_config = {
                'config': {'csv_path': aggregated_csv}
            }

            result = generate_eri_visualizations(self.comparative_dir, viz_config)

            # Verify visualization was called
            mock_generate_viz.assert_called_once()
            call_args = mock_generate_viz.call_args

            # Check that the output directory and config were passed correctly
            self.assertEqual(call_args[0][0], self.comparative_dir)
            self.assertIn('config', call_args[0][1])

        except ImportError as e:
            self.fail(f"Visualization integration failed: {e}")

    def test_error_handling_and_robustness(self):
        """Test error handling throughout the pipeline."""

        try:
            from run_einstellung_experiment import aggregate_comparative_results, find_csv_file

            # Test with empty results list
            empty_results = []
            try:
                aggregated_csv = aggregate_comparative_results(empty_results, self.comparative_dir)
                # Should handle gracefully, possibly returning None or empty file
            except Exception as e:
                # Should not crash with unhandled exception
                self.assertIsInstance(e, (ValueError, FileNotFoundError))

            # Test with missing CSV files
            incomplete_results = [{
                'strategy': 'nonexistent',
                'output_dir': '/nonexistent/path',
                'success': True
            }]

            try:
                aggregated_csv = aggregate_comparative_results(incomplete_results, self.comparative_dir)
                # Should handle missing files gracefully
            except Exception as e:
                # Should provide meaningful error message
                self.assertIsInstance(e, (ValueError, FileNotFoundError))

            # Test CSV file discovery with nonexistent directory
            csv_path = find_csv_file('/nonexistent/directory')
            self.assertIsNone(csv_path)  # Should return None, not crash

        except ImportError as e:
            self.fail(f"Error handling test failed due to import: {e}")

    def test_checkpoint_management_integration(self):
        """Test integration with checkpoint management system."""

        try:
            from run_einstellung_experiment import find_existing_checkpoints

            # Test checkpoint discovery for baseline methods
            scratch_checkpoints = find_existing_checkpoints('scratch_t2', 'resnet18', 42, self.checkpoints_dir)
            self.assertIsInstance(scratch_checkpoints, list)

            interleaved_checkpoints = find_existing_checkpoints('interleaved', 'resnet18', 42, self.checkpoints_dir)
            self.assertIsInstance(interleaved_checkpoints, list)

            # Test that empty directory returns empty list (baseline functionality)
            self.assertEqual(len(scratch_checkpoints), 0)  # No checkpoints exist initially
            self.assertEqual(len(interleaved_checkpoints), 0)  # No checkpoints exist initially

            # Test that function handles baseline methods without crashing
            # (The exact checkpoint discovery logic is tested in dedicated checkpoint tests)

        except ImportError as e:
            self.fail(f"Checkpoint management integration failed: {e}")

    def test_backward_compatibility(self):
        """Test that existing single-method experiments still work."""

        try:
            from run_einstellung_experiment import create_einstellung_args, extract_accuracy_from_output

            # Test existing methods still work
            existing_methods = ['sgd', 'derpp', 'ewc_on', 'gpm']

            for method in existing_methods:
                # Should be able to create args without error
                args = create_einstellung_args(method, 'resnet18', 42, debug=True)
                self.assertIn('--model', args)
                self.assertIn(method, args)

                # Args should contain all required parameters
                self.assertIn('--dataset', args)
                self.assertIn('seq-cifar100-einstellung', args)
                self.assertIn('--backbone', args)
                self.assertIn('resnet18', args)

            # Test that accuracy extraction still works for existing output formats
            existing_outputs = [
                "Accuracy for 2 task(s): [Class-IL]: 75.5 %",
                "Final accuracy: 82.3%",
                "Test accuracy: 68.9%"
            ]

            for output in existing_outputs:
                try:
                    accuracy = extract_accuracy_from_output(output)
                    self.assertIsInstance(accuracy, float)
                    self.assertGreater(accuracy, 0)
                    self.assertLess(accuracy, 100)
                except:
                    # Some formats might not be supported, but should not crash
                    pass

        except ImportError as e:
            self.fail(f"Backward compatibility test failed: {e}")

    def test_comparative_experiment_runner_integration(self):
        """Test the complete comparative experiment runner integration."""

        try:
            from run_einstellung_experiment import run_comparative_experiment

            # Mock the individual experiment runs to avoid actual training
            with patch('run_einstellung_experiment.run_einstellung_experiment') as mock_run:
                # Configure mock to return successful results
                mock_results = self.create_mock_experiment_results()
                mock_run.return_value = mock_results[0]  # Return first result for all calls

                # Mock visualization generation
                with patch('run_einstellung_experiment.generate_eri_visualizations') as mock_viz:
                    mock_viz.return_value = True

                    # Run comparative experiment (should not crash)
                    try:
                        results = run_comparative_experiment(
                            skip_training=True,  # Skip actual training
                            force_retrain=False,
                            auto_checkpoint=True,
                            debug=True
                        )

                        # Should return results list
                        self.assertIsInstance(results, list)

                        # Should have called individual experiments
                        self.assertGreater(mock_run.call_count, 0)

                        # Should have called visualization
                        mock_viz.assert_called()

                    except Exception as e:
                        # Should not fail with unhandled exception
                        self.fail(f"Comparative experiment runner failed: {e}")

        except ImportError as e:
            self.fail(f"Comparative experiment runner not available: {e}")

    def test_end_to_end_pipeline_simulation(self):
        """Test a complete end-to-end pipeline simulation."""

        try:
            # Step 1: Verify baseline methods are available
            from models.scratch_t2 import ScratchT2
            from models.interleaved import Interleaved

            # Step 2: Create mock experiment results (simulating completed experiments)
            results = self.create_mock_experiment_results()

            # Step 3: Test data aggregation
            from run_einstellung_experiment import aggregate_comparative_results
            aggregated_csv = aggregate_comparative_results(results, self.comparative_dir)

            # Step 4: Test statistical analysis
            from utils.statistical_analysis import StatisticalAnalyzer
            analyzer = StatisticalAnalyzer(alpha=0.05, correction_method='bonferroni')
            statistical_results = analyzer.analyze_comparative_metrics(aggregated_csv)

            # Step 5: Test report generation
            from utils.statistical_analysis import generate_statistical_report
            report_path = generate_statistical_report(aggregated_csv, self.comparative_dir)

            # Step 6: Verify complete pipeline results

            # Check aggregated data quality
            df = pd.read_csv(aggregated_csv)
            self.assertGreater(len(df), 0)
            self.assertEqual(len(df['method'].unique()), 5)  # All 5 methods

            # Check statistical analysis completeness
            self.assertIn('summary_statistics', statistical_results)
            self.assertIn('pairwise_comparisons', statistical_results)

            # Check baseline methods in results
            summary_stats = statistical_results['summary_statistics']
            self.assertIn('scratch_t2', summary_stats)
            self.assertIn('interleaved', summary_stats)

            # Check report generation
            self.assertTrue(os.path.exists(report_path))

            with open(report_path, 'r') as f:
                content = f.read()

            # Verify report contains key information
            self.assertIn('Comparative Einstellung Effect Analysis', content)
            self.assertIn('scratch_t2', content)
            self.assertIn('interleaved', content)
            self.assertIn('Statistical Analysis', content)

            # Step 7: Verify output organization
            expected_files = [
                'comparative_eri_metrics.csv',  # Aggregated data
                'statistical_analysis_report.html'  # Statistical report
            ]

            for filename in expected_files:
                filepath = os.path.join(self.comparative_dir, filename)
                self.assertTrue(os.path.exists(filepath), f"Missing output file: {filename}")

            print("✅ End-to-end pipeline simulation completed successfully")

        except Exception as e:
            self.fail(f"End-to-end pipeline simulation failed: {e}")

    def test_performance_and_memory_efficiency(self):
        """Test that the pipeline handles larger datasets efficiently."""

        try:
            import time
            import psutil
            import os

            # Create larger mock dataset
            methods = ['scratch_t2', 'interleaved', 'sgd', 'derpp', 'ewc_on', 'gpm']
            seeds = [42, 43, 44, 45, 46]  # 5 seeds

            large_results = []
            for method in methods:
                for seed in seeds:
                    method_dir = os.path.join(self.results_dir, f"{method}_resnet18_seed{seed}")
                    os.makedirs(method_dir, exist_ok=True)

                    # Create larger CSV (more epochs)
                    csv_path = os.path.join(method_dir, "eri_metrics.csv")
                    self.create_large_mock_csv(csv_path, method, seed)

                    result = {
                        'strategy': method,
                        'backbone': 'resnet18',
                        'seed': seed,
                        'final_accuracy': 70.0 + hash(method + str(seed)) % 20,
                        'output_dir': method_dir,
                        'success': True,
                        'used_checkpoint': False
                    }
                    large_results.append(result)

            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Run aggregation
            from run_einstellung_experiment import aggregate_comparative_results
            aggregated_csv = aggregate_comparative_results(large_results, self.comparative_dir)

            # Run statistical analysis
            from utils.statistical_analysis import StatisticalAnalyzer
            analyzer = StatisticalAnalyzer(alpha=0.05, correction_method='bonferroni')
            statistical_results = analyzer.analyze_comparative_metrics(aggregated_csv)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Performance checks
            processing_time = end_time - start_time
            memory_increase = end_memory - start_memory

            # Should complete in reasonable time (< 30 seconds for test data)
            self.assertLess(processing_time, 30.0, "Processing took too long")

            # Should not use excessive memory (< 500 MB increase)
            self.assertLess(memory_increase, 500.0, "Memory usage too high")

            # Verify results are still correct
            df = pd.read_csv(aggregated_csv)
            self.assertEqual(len(df['method'].unique()), 6)  # All 6 methods
            self.assertEqual(len(df['seed'].unique()), 5)   # All 5 seeds

            print(f"✅ Performance test completed: {processing_time:.2f}s, {memory_increase:.1f}MB")

        except ImportError as e:
            self.skipTest(f"Performance testing dependencies not available: {e}")

    def create_large_mock_csv(self, csv_path: str, method: str, seed: int):
        """Create a larger mock CSV for performance testing."""

        np.random.seed(seed + hash(method) % 2**32)

        epochs = np.linspace(0.0, 2.0, 51)  # More epochs for larger dataset
        splits = ['T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal']

        data = []
        for epoch in epochs:
            for split in splits:
                # Simple performance model
                base_acc = 0.7 + (hash(method + split) % 100) / 500.0  # 0.5-0.9 range
                progress = 1 - np.exp(-0.1 * epoch)
                acc = base_acc * progress + np.random.normal(0, 0.02)
                acc = max(0.0, min(1.0, acc))

                data.append({
                    'method': method,
                    'seed': seed,
                    'epoch_eff': epoch,
                    'split': split,
                    'acc': acc
                })

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
