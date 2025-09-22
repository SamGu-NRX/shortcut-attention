#!/usr/bin/env python3
"""
Integration tests for statistical analysis with comparative experiment runner.

Tests the integration of statistical analysis (Tasks 13 & 14) with the
existing comparative experiment infrastructure.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from run_einstellung_experiment import get_significance_indicator


class TestStatisticalIntegration(unittest.TestCase):
    """Test suite for statistical analysis integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_significance_indicator_function(self):
        """Test the significance indicator function for summary table."""

        # Test case 1: No statistical results
        sig = get_significance_indicator('sgd', {})
        self.assertEqual(sig, "")

        # Test case 2: No pairwise comparisons
        statistical_results = {'other_data': 'value'}
        sig = get_significance_indicator('sgd', statistical_results)
        self.assertEqual(sig, "")

        # Test case 3: Method with highly significant comparison (p < 0.001)
        mock_comparison = MagicMock()
        mock_comparison.method1 = 'sgd'
        mock_comparison.method2 = 'derpp'
        mock_comparison.test_result.p_value = 0.0005

        statistical_results = {
            'pairwise_comparisons': {
                'final_accuracy': [mock_comparison]
            }
        }
        sig = get_significance_indicator('sgd', statistical_results)
        self.assertEqual(sig, "***")

        # Test case 4: Method with moderately significant comparison (p < 0.01)
        mock_comparison.test_result.p_value = 0.005
        sig = get_significance_indicator('sgd', statistical_results)
        self.assertEqual(sig, "**")

        # Test case 5: Method with marginally significant comparison (p < 0.05)
        mock_comparison.test_result.p_value = 0.03
        sig = get_significance_indicator('sgd', statistical_results)
        self.assertEqual(sig, "*")

        # Test case 6: Method with non-significant comparison (p >= 0.05)
        mock_comparison.test_result.p_value = 0.1
        sig = get_significance_indicator('sgd', statistical_results)
        self.assertEqual(sig, "")

        # Test case 7: Method not involved in any comparisons
        mock_comparison.method1 = 'derpp'
        mock_comparison.method2 = 'ewc_on'
        sig = get_significance_indicator('sgd', statistical_results)
        self.assertEqual(sig, "")

        # Test case 8: Handle NaN p-values gracefully
        mock_comparison.method1 = 'sgd'
        mock_comparison.test_result.p_value = np.nan
        sig = get_significance_indicator('sgd', statistical_results)
        self.assertEqual(sig, "")

    def test_significance_indicator_multiple_comparisons(self):
        """Test significance indicator with multiple comparisons for same method."""

        # Create multiple comparisons with different p-values
        mock_comp1 = MagicMock()
        mock_comp1.method1 = 'sgd'
        mock_comp1.method2 = 'derpp'
        mock_comp1.test_result.p_value = 0.03  # Significant

        mock_comp2 = MagicMock()
        mock_comp2.method1 = 'ewc_on'
        mock_comp2.method2 = 'sgd'  # sgd as method2
        mock_comp2.test_result.p_value = 0.0001  # Highly significant

        mock_comp3 = MagicMock()
        mock_comp3.method1 = 'derpp'
        mock_comp3.method2 = 'ewc_on'
        mock_comp3.test_result.p_value = 0.2  # Not significant

        statistical_results = {
            'pairwise_comparisons': {
                'final_accuracy': [mock_comp1, mock_comp2, mock_comp3]
            }
        }

        # Should return the most significant indicator (lowest p-value)
        sig = get_significance_indicator('sgd', statistical_results)
        self.assertEqual(sig, "***")  # Based on p=0.0001 from mock_comp2

    def test_significance_indicator_across_metrics(self):
        """Test significance indicator considers all metrics."""

        # Create comparisons across different metrics
        mock_comp_acc = MagicMock()
        mock_comp_acc.method1 = 'sgd'
        mock_comp_acc.method2 = 'derpp'
        mock_comp_acc.test_result.p_value = 0.08  # Not significant

        mock_comp_pd = MagicMock()
        mock_comp_pd.method1 = 'sgd'
        mock_comp_pd.method2 = 'ewc_on'
        mock_comp_pd.test_result.p_value = 0.02  # Significant

        statistical_results = {
            'pairwise_comparisons': {
                'final_accuracy': [mock_comp_acc],
                'performance_deficit': [mock_comp_pd]
            }
        }

        # Should find the significant comparison across all metrics
        sig = get_significance_indicator('sgd', statistical_results)
        self.assertEqual(sig, "*")

    def create_mock_statistical_results(self, methods: list, significant_pairs: list = None):
        """
        Create mock statistical results for testing.

        Args:
            methods: List of method names
            significant_pairs: List of (method1, method2, p_value) tuples for significant comparisons

        Returns:
            Mock statistical results dictionary
        """
        if significant_pairs is None:
            significant_pairs = []

        comparisons = []

        # Create all pairwise comparisons
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                mock_comp = MagicMock()
                mock_comp.method1 = method1
                mock_comp.method2 = method2

                # Check if this pair should be significant
                p_value = 0.1  # Default non-significant
                for sig_method1, sig_method2, sig_p in significant_pairs:
                    if ((method1 == sig_method1 and method2 == sig_method2) or
                        (method1 == sig_method2 and method2 == sig_method1)):
                        p_value = sig_p
                        break

                mock_comp.test_result.p_value = p_value
                comparisons.append(mock_comp)

        return {
            'pairwise_comparisons': {
                'final_accuracy': comparisons
            },
            'interpretation': {
                'overview': f'Analysis of {len(methods)} methods',
                'significant_differences': 'Some significant differences detected',
                'large_effects': 'Some large effects detected'
            }
        }

    def test_integration_with_comparative_summary(self):
        """Test integration of statistical indicators with comparative summary table."""

        methods = ['sgd', 'derpp', 'ewc_on', 'gpm']
        significant_pairs = [
            ('sgd', 'derpp', 0.001),  # Highly significant
            ('ewc_on', 'gpm', 0.03),  # Marginally significant
        ]

        statistical_results = self.create_mock_statistical_results(methods, significant_pairs)

        # Test significance indicators for each method
        sgd_sig = get_significance_indicator('sgd', statistical_results)
        self.assertEqual(sgd_sig, "***")

        derpp_sig = get_significance_indicator('derpp', statistical_results)
        self.assertEqual(derpp_sig, "***")

        ewc_sig = get_significance_indicator('ewc_on', statistical_results)
        self.assertEqual(ewc_sig, "*")

        gpm_sig = get_significance_indicator('gpm', statistical_results)
        self.assertEqual(gpm_sig, "*")

    def test_statistical_results_structure_validation(self):
        """Test that statistical results have expected structure for integration."""

        # Test with properly structured results
        proper_results = {
            'pairwise_comparisons': {
                'final_accuracy': [],
                'performance_deficit': []
            },
            'anova_results': {
                'final_accuracy': MagicMock()
            },
            'interpretation': {
                'overview': 'Test overview',
                'significant_differences': 'Test differences',
                'large_effects': 'Test effects'
            }
        }

        # Should handle properly structured results
        sig = get_significance_indicator('sgd', proper_results)
        self.assertEqual(sig, "")  # No significant comparisons in empty list

        # Test with malformed results
        malformed_results = {
            'pairwise_comparisons': 'not_a_dict',
            'interpretation': None
        }

        # Should handle malformed results gracefully
        sig = get_significance_indicator('sgd', malformed_results)
        self.assertEqual(sig, "")

    def test_baseline_method_significance_indicators(self):
        """Test significance indicators for baseline methods."""

        methods = ['sgd', 'derpp', 'scratch_t2', 'interleaved']
        significant_pairs = [
            ('sgd', 'scratch_t2', 0.001),  # CL method vs baseline
            ('derpp', 'scratch_t2', 0.02),
            ('scratch_t2', 'interleaved', 0.3),  # Baseline vs baseline (not significant)
        ]

        statistical_results = self.create_mock_statistical_results(methods, significant_pairs)

        # Test that baseline methods get appropriate significance indicators
        scratch_sig = get_significance_indicator('scratch_t2', statistical_results)
        self.assertEqual(scratch_sig, "***")  # Most significant comparison

        interleaved_sig = get_significance_indicator('interleaved', statistical_results)
        self.assertEqual(interleaved_sig, "")  # No significant comparisons


if __name__ == '__main__':
    unittest.main()
