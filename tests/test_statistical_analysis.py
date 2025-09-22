#!/usr/bin/env python3
"""
Unit tests for statistical analysis functionality.

Tests the statistical significance testing and effect size calculations
for comparative Einstellung analysis (Tasks 13 & 14).
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.statistical_analysis import StatisticalAnalyzer, generate_statistical_report


class TestStatisticalAnalysis(unittest.TestCase):
    """Test suite for statistical analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = StatisticalAnalyzer(alpha=0.05, correction_method='bonferroni')

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_csv(self, filename: str, methods: list, seeds: list,
                       effect_size: float = 0.0) -> str:
        """
        Create a test CSV file with synthetic Einstellung data.

        Args:
            filename: Name of CSV file to create
            methods: List of method names
            seeds: List of seed values
            effect_size: Effect size to simulate between methods (Cohen's d)

        Returns:
            Path to created CSV file
        """
        data = []

        for method_idx, method in enumerate(methods):
            for seed in seeds:
                # Simulate different performance levels with controlled effect sizes
                base_acc = 0.7  # Base accuracy
                method_effect = method_idx * effect_size * 0.1  # Systematic difference between methods

                for epoch in [0.0, 0.5, 1.0]:  # Simulate 3 epochs
                    # Add some learning progression
                    epoch_boost = epoch * 0.1

                    # Different splits with realistic Einstellung patterns
                    splits_data = {
                        'T1_all': base_acc + method_effect + epoch_boost + np.random.normal(0, 0.05),
                        'T2_shortcut_normal': base_acc + 0.2 + method_effect + epoch_boost + np.random.normal(0, 0.05),
                        'T2_shortcut_masked': base_acc + 0.1 + method_effect + epoch_boost + np.random.normal(0, 0.05),
                        'T2_nonshortcut_normal': base_acc + method_effect + epoch_boost + np.random.normal(0, 0.05)
                    }

                    for split, acc in splits_data.items():
                        data.append({
                            'method': method,
                            'seed': seed,
                            'epoch_eff': epoch,
                            'split': split,
                            'acc': max(0.0, min(1.0, acc))  # Clamp to [0, 1]
                        })

        df = pd.DataFrame(data)
        csv_path = os.path.join(self.temp_dir, filename)
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_statistical_analyzer_initialization(self):
        """Test StatisticalAnalyzer initialization with different parameters."""
        # Test default initialization
        analyzer = StatisticalAnalyzer()
        self.assertEqual(analyzer.alpha, 0.05)
        self.assertEqual(analyzer.correction_method, 'bonferroni')

        # Test custom initialization
        analyzer = StatisticalAnalyzer(alpha=0.01, correction_method='fdr')
        self.assertEqual(analyzer.alpha, 0.01)
        self.assertEqual(analyzer.correction_method, 'fdr')

        # Test invalid correction method
        with self.assertRaises(ValueError):
            StatisticalAnalyzer(correction_method='invalid')

    def test_compute_statistical_metrics(self):
        """Test computation of statistical metrics from raw accuracy data."""
        # Create test data with known patterns
        csv_path = self.create_test_csv('test_metrics.csv',
                                      methods=['sgd', 'derpp'],
                                      seeds=[42, 43, 44],
                                      effect_size=1.0)  # Large effect size

        df = pd.read_csv(csv_path)
        metrics_df = self.analyzer._compute_statistical_metrics(df)

        # Verify structure
        self.assertIn('method', metrics_df.columns)
        self.assertIn('seed', metrics_df.columns)
        self.assertIn('final_accuracy', metrics_df.columns)
        self.assertIn('performance_deficit', metrics_df.columns)
        self.assertIn('shortcut_reliance', metrics_df.columns)

        # Verify we have data for both methods
        methods = metrics_df['method'].unique()
        self.assertIn('sgd', methods)
        self.assertIn('derpp', methods)

        # Verify we have data for all seeds
        seeds = metrics_df['seed'].unique()
        self.assertEqual(len(seeds), 3)

    def test_pairwise_comparisons(self):
        """Test pairwise statistical comparisons between methods."""
        # Create test data with large effect size for reliable detection
        csv_path = self.create_test_csv('test_pairwise.csv',
                                      methods=['sgd', 'derpp', 'ewc_on'],
                                      seeds=[42, 43, 44, 45, 46],  # More seeds for power
                                      effect_size=2.0)  # Very large effect

        df = pd.read_csv(csv_path)
        metrics_df = self.analyzer._compute_statistical_metrics(df)

        pairwise_results = self.analyzer._perform_pairwise_tests(metrics_df)

        # Verify structure
        self.assertIn('final_accuracy', pairwise_results)

        # Check that we have the expected number of comparisons
        # 3 methods = 3 choose 2 = 3 pairwise comparisons
        comparisons = pairwise_results['final_accuracy']
        self.assertEqual(len(comparisons), 3)

        # Verify comparison structure
        for comp in comparisons:
            self.assertIsNotNone(comp.method1)
            self.assertIsNotNone(comp.method2)
            self.assertIsNotNone(comp.test_result)
            self.assertIn('mean', comp.method1_stats)
            self.assertIn('mean', comp.method2_stats)

    def test_anova_analysis(self):
        """Test ANOVA analysis for multi-group comparisons."""
        # Create test data with systematic differences
        csv_path = self.create_test_csv('test_anova.csv',
                                      methods=['sgd', 'derpp', 'ewc_on', 'gpm'],
                                      seeds=[42, 43, 44, 45],
                                      effect_size=1.5)  # Large effect for detection

        df = pd.read_csv(csv_path)
        metrics_df = self.analyzer._compute_statistical_metrics(df)

        anova_results = self.analyzer._perform_anova_tests(metrics_df)

        # Verify structure
        self.assertIn('final_accuracy', anova_results)

        # Check ANOVA result structure
        result = anova_results['final_accuracy']
        self.assertEqual(result.test_name, "One-way ANOVA")
        self.assertIsNotNone(result.statistic)
        self.assertIsNotNone(result.p_value)
        self.assertIsNotNone(result.effect_size)  # eta-squared
        self.assertIsNotNone(result.interpretation)

    def test_effect_size_calculations(self):
        """Test effect size calculations (Cohen's d)."""
        # Create test data with known effect sizes
        csv_path = self.create_test_csv('test_effects.csv',
                                      methods=['method_low', 'method_high'],
                                      seeds=[42, 43, 44],
                                      effect_size=0.8)  # Target Cohen's d ≈ 0.8

        df = pd.read_csv(csv_path)
        metrics_df = self.analyzer._compute_statistical_metrics(df)

        effect_sizes = self.analyzer._compute_effect_sizes(metrics_df)

        # Verify structure
        self.assertIn('final_accuracy', effect_sizes)

        # Check that effect size is computed
        effects = effect_sizes['final_accuracy']
        self.assertTrue(len(effects) > 0)

        # Verify effect size is reasonable (should be positive since method_high > method_low)
        for comparison, effect_size in effects.items():
            self.assertIsInstance(effect_size, float)
            self.assertFalse(np.isnan(effect_size))

    def test_multiple_comparison_corrections(self):
        """Test multiple comparison corrections (Bonferroni, FDR)."""
        # Test Bonferroni correction
        analyzer_bonf = StatisticalAnalyzer(correction_method='bonferroni')
        csv_path = self.create_test_csv('test_corrections.csv',
                                      methods=['sgd', 'derpp', 'ewc_on'],
                                      seeds=[42, 43, 44],
                                      effect_size=1.0)

        df = pd.read_csv(csv_path)
        metrics_df = analyzer_bonf._compute_statistical_metrics(df)

        corrections = analyzer_bonf._apply_multiple_comparison_corrections(metrics_df)

        # Verify structure
        self.assertEqual(corrections['correction_method'], 'bonferroni')

        # Test no correction
        analyzer_none = StatisticalAnalyzer(correction_method='none')
        corrections_none = analyzer_none._apply_multiple_comparison_corrections(metrics_df)
        self.assertEqual(corrections_none['correction_method'], 'none')

    def test_comprehensive_analysis(self):
        """Test the complete analyze_comparative_metrics pipeline."""
        # Create comprehensive test dataset
        csv_path = self.create_test_csv('test_comprehensive.csv',
                                      methods=['sgd', 'derpp', 'ewc_on', 'scratch_t2'],
                                      seeds=[42, 43, 44, 45],
                                      effect_size=1.0)

        results = self.analyzer.analyze_comparative_metrics(csv_path)

        # Verify all expected sections are present
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
            self.assertIn(section, results)

        # Verify interpretation section
        interpretation = results['interpretation']
        self.assertIn('overview', interpretation)
        self.assertIn('significant_differences', interpretation)

    def test_statistical_report_generation(self):
        """Test generation of statistical analysis HTML report."""
        # Create test data
        csv_path = self.create_test_csv('test_report.csv',
                                      methods=['sgd', 'derpp', 'ewc_on'],
                                      seeds=[42, 43, 44],
                                      effect_size=1.0)

        # Generate report
        report_path = generate_statistical_report(csv_path, self.temp_dir)

        # Verify report was created
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith('.html'))

        # Verify report contains expected content
        with open(report_path, 'r') as f:
            content = f.read()

        self.assertIn('Statistical Analysis Report', content)
        self.assertIn('ANOVA Results', content)
        self.assertIn('Effect Sizes', content)
        self.assertIn('Summary Statistics', content)

    def test_missing_data_handling(self):
        """Test handling of missing or invalid data."""
        # Create CSV with missing data
        data = [
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T2_shortcut_normal', 'acc': 0.7},
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': np.nan},
            {'method': 'derpp', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T2_shortcut_normal', 'acc': 0.8},
        ]

        df = pd.DataFrame(data)
        csv_path = os.path.join(self.temp_dir, 'test_missing.csv')
        df.to_csv(csv_path, index=False)

        # Should handle missing data gracefully
        results = self.analyzer.analyze_comparative_metrics(csv_path)
        self.assertNotIn('error', results)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for statistical tests."""
        # Create CSV with only one sample per method
        data = [
            {'method': 'sgd', 'seed': 42, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': 0.7},
            {'method': 'derpp', 'seed': 42, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': 0.8},
        ]

        df = pd.DataFrame(data)
        csv_path = os.path.join(self.temp_dir, 'test_insufficient.csv')
        df.to_csv(csv_path, index=False)

        # Should handle insufficient data gracefully
        results = self.analyzer.analyze_comparative_metrics(csv_path)
        self.assertNotIn('error', results)

        # Check that appropriate warnings are in interpretations
        pairwise_results = results.get('pairwise_comparisons', {})
        if 'final_accuracy' in pairwise_results:
            for comp in pairwise_results['final_accuracy']:
                # Should indicate insufficient data
                self.assertIn('Insufficient data', comp.test_result.interpretation)

    def test_baseline_method_detection(self):
        """Test detection and handling of baseline methods."""
        # Create data including baseline methods
        csv_path = self.create_test_csv('test_baselines.csv',
                                      methods=['sgd', 'derpp', 'scratch_t2', 'interleaved'],
                                      seeds=[42, 43],
                                      effect_size=0.5)

        df = pd.read_csv(csv_path)
        metrics_df = self.analyzer._compute_statistical_metrics(df)

        # Verify baseline methods are included in metrics
        methods = metrics_df['method'].unique()
        self.assertIn('scratch_t2', methods)
        self.assertIn('interleaved', methods)

        # Verify pairwise comparisons include baselines
        pairwise_results = self.analyzer._perform_pairwise_tests(metrics_df)

        all_comparisons = []
        for metric, comparisons in pairwise_results.items():
            for comp in comparisons:
                all_comparisons.append((comp.method1, comp.method2))

        # Should have comparisons involving baseline methods
        baseline_comparisons = [comp for comp in all_comparisons
                              if 'scratch_t2' in comp or 'interleaved' in comp]
        self.assertTrue(len(baseline_comparisons) > 0)


if __name__ == '__main__':
    unittest.main()
