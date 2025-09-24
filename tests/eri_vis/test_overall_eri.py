"""
Tests for overall ERI metric visualization.

This module tests the overall ERI metric computation and visualization
functionality, including composite score calculation and bar chart generation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt

from eri_vis.plot_overall_eri import OverallERIPlotter
from eri_vis.eri_composite import CompositeERICalculator, CompositeERIConfig
from eri_vis.metrics_calculator import ERIMetrics


class TestOverallERIPlotter:
    """Test overall ERI metric plotting functionality."""

    def create_test_method_stats(self):
        """Create test method statistics for plotting."""
        return {
            'derpp': {
                'AD_mean': 5.0, 'AD_ci': 1.0, 'AD_n': 3,
                'PD_mean': 0.05, 'PD_ci': 0.01, 'PD_n': 3,
                'SFR_rel_mean': 0.02, 'SFR_rel_ci': 0.005, 'SFR_rel_n': 3,
                'censored_runs': 0, 'total_runs': 3
            },
            'ewc_on': {
                'AD_mean': 8.0, 'AD_ci': 1.5, 'AD_n': 3,
                'PD_mean': 0.08, 'PD_ci': 0.02, 'PD_n': 3,
                'SFR_rel_mean': 0.04, 'SFR_rel_ci': 0.01, 'SFR_rel_n': 3,
                'censored_runs': 0, 'total_runs': 3
            },
            'sgd': {
                'AD_mean': 12.0, 'AD_ci': 2.0, 'AD_n': 3,
                'PD_mean': 0.12, 'PD_ci': 0.03, 'PD_n': 3,
                'SFR_rel_mean': 0.06, 'SFR_rel_ci': 0.015, 'SFR_rel_n': 3,
                'censored_runs': 1, 'total_runs': 3
            },
            'scratch_t2': {
                'AD_mean': 0.0, 'AD_ci': 0.0, 'AD_n': 3,
                'PD_mean': 0.0, 'PD_ci': 0.0, 'PD_n': 3,
                'SFR_rel_mean': 0.0, 'SFR_rel_ci': 0.0, 'SFR_rel_n': 3,
                'censored_runs': 0, 'total_runs': 3
            }
        }

    def test_overall_eri_computation(self):
        """Test overall ERI score computation."""
        plotter = OverallERIPlotter()
        method_stats = self.create_test_method_stats()

        # Compute ERI scores
        eri_scores = plotter._compute_overall_eri_scores(
            method_stats, ad_weight=0.4, pd_weight=0.4, sfr_weight=0.2, ad_max=50.0
        )

        # Check that scores are computed for all methods
        assert 'derpp' in eri_scores
        assert 'ewc_on' in eri_scores
        assert 'sgd' in eri_scores

        # Check that DER++ has the lowest (best) score
        derpp_score = eri_scores['derpp']['mean']
        ewc_score = eri_scores['ewc_on']['mean']
        sgd_score = eri_scores['sgd']['mean']

        assert derpp_score < ewc_score < sgd_score

        # Check that scores are in reasonable range
        for method, score_data in eri_scores.items():
            assert 0.0 <= score_data['mean'] <= 1.0
            assert score_data['ci'] >= 0.0

    def test_plot_generation(self):
        """Test overall ERI plot generation."""
        plotter = OverallERIPlotter()
        method_stats = self.create_test_method_stats()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_overall_eri.pdf"

            # Generate plot
            plotter.plot_overall_eri(
                method_stats, output_path, figsize=(8, 6), dpi=150
            )

            # Check that file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_method_name_formatting(self):
        """Test method name formatting for display."""
        plotter = OverallERIPlotter()

        # Test various method name formats
        assert plotter._format_method_name('derpp') == 'DER++'
        assert plotter._format_method_name('ewc_on') == 'EWC On'
        assert plotter._format_method_name('sgd') == 'SGD'
        assert plotter._format_method_name('gpm') == 'GPM'
        assert plotter._format_method_name('scratch_t2') == 'Scratch T2'

    def test_rigidity_color_mapping(self):
        """Test color mapping for rigidity levels."""
        plotter = OverallERIPlotter()

        # Test color mapping
        low_color = plotter._get_rigidity_color(0.1)   # Should be greenish
        mid_color = plotter._get_rigidity_color(0.5)   # Should be yellowish
        high_color = plotter._get_rigidity_color(0.9)  # Should be reddish

        # Colors should be different
        assert low_color != mid_color != high_color

        # Should handle edge cases
        edge_low = plotter._get_rigidity_color(0.0)
        edge_high = plotter._get_rigidity_color(1.0)
        assert edge_low is not None
        assert edge_high is not None

    def test_comparison_table_generation(self):
        """Test ERI comparison table generation."""
        plotter = OverallERIPlotter()
        method_stats = self.create_test_method_stats()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_comparison_table.csv"

            # Generate table
            plotter.create_eri_comparison_table(
                method_stats, output_path
            )

            # Check that file was created
            assert output_path.exists()

            # Load and check content
            df = pd.read_csv(output_path)
            assert len(df) > 0
            assert 'Method' in df.columns
            assert 'Overall_ERI' in df.columns
            assert 'AD_mean' in df.columns
            assert 'PD_mean' in df.columns
            assert 'SFR_rel_mean' in df.columns


class TestCompositeERICalculator:
    """Test composite ERI score calculation."""

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = CompositeERIConfig(ad_weight=0.4, pd_weight=0.4, sfr_weight=0.2)
        assert config.ad_weight == 0.4

        # Invalid config (weights don't sum to 1)
        with pytest.raises(ValueError):
            CompositeERIConfig(ad_weight=0.5, pd_weight=0.4, sfr_weight=0.2)

        # Invalid ad_max
        with pytest.raises(ValueError):
            CompositeERIConfig(ad_max=-1.0)

    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        calculator = CompositeERICalculator()

        # Create test metrics
        metrics = ERIMetrics(
            adaptation_delay=10.0,
            performance_deficit=0.1,
            shortcut_feature_reliance=0.05,
            method='test_method',
            seed=42
        )

        composite = calculator.compute_composite_score(metrics)

        # Should be a valid score
        assert composite is not None
        assert 0.0 <= composite <= 1.0

    def test_method_ranking(self):
        """Test method ranking by composite score."""
        calculator = CompositeERICalculator()

        # Create test metrics for multiple methods
        method_metrics = {
            'good_method': [
                ERIMetrics(adaptation_delay=5.0, performance_deficit=0.02,
                          shortcut_feature_reliance=0.01, method='good_method', seed=i)
                for i in range(3)
            ],
            'bad_method': [
                ERIMetrics(adaptation_delay=20.0, performance_deficit=0.15,
                          shortcut_feature_reliance=0.08, method='bad_method', seed=i)
                for i in range(3)
            ]
        }

        rankings = calculator.rank_methods_by_composite(method_metrics)

        # Good method should rank first (lower score)
        assert len(rankings) == 2
        assert rankings[0][0] == 'good_method'
        assert rankings[1][0] == 'bad_method'
        assert rankings[0][1] < rankings[1][1]  # Lower score is better

    def test_component_analysis(self):
        """Test component contribution analysis."""
        calculator = CompositeERICalculator()

        # Create test metrics
        method_metrics = {
            'test_method': [
                ERIMetrics(adaptation_delay=10.0, performance_deficit=0.1,
                          shortcut_feature_reliance=0.05, method='test_method', seed=42)
            ]
        }

        analysis = calculator.analyze_component_contributions(method_metrics)

        # Check analysis structure
        assert 'test_method' in analysis
        method_analysis = analysis['test_method']

        assert 'ad_contribution' in method_analysis
        assert 'pd_contribution' in method_analysis
        assert 'sfr_contribution' in method_analysis
        assert 'ad_percentage' in method_analysis
        assert 'pd_percentage' in method_analysis
        assert 'sfr_percentage' in method_analysis

        # Percentages should sum to 100
        total_percentage = (method_analysis['ad_percentage'] +
                          method_analysis['pd_percentage'] +
                          method_analysis['sfr_percentage'])
        assert abs(total_percentage - 100.0) < 1e-6

    def test_interpretation_generation(self):
        """Test interpretation text generation."""
        calculator = CompositeERICalculator()

        # Create test data
        method_scores = [('good_method', 0.1), ('bad_method', 0.3)]
        component_analysis = {
            'good_method': {
                'ad_contribution': 0.04, 'pd_contribution': 0.04, 'sfr_contribution': 0.02,
                'ad_percentage': 40.0, 'pd_percentage': 40.0, 'sfr_percentage': 20.0,
                'total_score': 0.1
            },
            'bad_method': {
                'ad_contribution': 0.12, 'pd_contribution': 0.12, 'sfr_contribution': 0.06,
                'ad_percentage': 40.0, 'pd_percentage': 40.0, 'sfr_percentage': 20.0,
                'total_score': 0.3
            }
        }

        interpretation = calculator.generate_interpretation_text(
            method_scores, component_analysis
        )

        # Check that interpretation contains expected elements
        assert 'good_method' in interpretation
        assert 'bad_method' in interpretation
        assert 'Best Method' in interpretation
        assert 'Worst Method' in interpretation
        assert 'Method Ranking' in interpretation

    def test_validation(self):
        """Test composite score validation."""
        calculator = CompositeERICalculator()

        # Create valid test metrics
        valid_metrics = {
            'method1': [
                ERIMetrics(adaptation_delay=5.0, performance_deficit=0.05,
                          shortcut_feature_reliance=0.02, method='method1', seed=i)
                for i in range(3)
            ],
            'method2': [
                ERIMetrics(adaptation_delay=10.0, performance_deficit=0.08,
                          shortcut_feature_reliance=0.04, method='method2', seed=i)
                for i in range(3)
            ]
        }

        # Should pass validation
        assert calculator.validate_composite_scores(valid_metrics) is True

        # Test with empty metrics
        empty_metrics = {}
        assert calculator.validate_composite_scores(empty_metrics) is False


class TestIntegrationWithRealData:
    """Test integration with real data patterns."""

    def test_realistic_eri_scores(self):
        """Test with realistic ERI score ranges."""
        plotter = OverallERIPlotter()

        # Create realistic method statistics based on expected ranges
        realistic_stats = {
            'derpp': {
                'AD_mean': -2.0, 'AD_ci': 0.5, 'AD_n': 5,  # Negative AD (faster than scratch)
                'PD_mean': -0.02, 'PD_ci': 0.01, 'PD_n': 5,  # Negative PD (better than scratch)
                'SFR_rel_mean': 0.01, 'SFR_rel_ci': 0.005, 'SFR_rel_n': 5,  # Small positive SFR_rel
                'censored_runs': 0, 'total_runs': 5
            },
            'ewc_on': {
                'AD_mean': 3.0, 'AD_ci': 1.0, 'AD_n': 5,  # Positive AD (slower than scratch)
                'PD_mean': 0.05, 'PD_ci': 0.015, 'PD_n': 5,  # Positive PD (worse than scratch)
                'SFR_rel_mean': 0.03, 'SFR_rel_ci': 0.01, 'SFR_rel_n': 5,  # Higher SFR_rel
                'censored_runs': 0, 'total_runs': 5
            },
            'sgd': {
                'AD_mean': 8.0, 'AD_ci': 2.0, 'AD_n': 4,  # Much slower
                'PD_mean': 0.10, 'PD_ci': 0.03, 'PD_n': 4,  # Much worse
                'SFR_rel_mean': 0.05, 'SFR_rel_ci': 0.02, 'SFR_rel_n': 4,  # High SFR_rel
                'censored_runs': 1, 'total_runs': 5  # Some censored runs
            }
        }

        # Compute ERI scores
        eri_scores = plotter._compute_overall_eri_scores(
            realistic_stats, ad_weight=0.4, pd_weight=0.4, sfr_weight=0.2, ad_max=50.0
        )

        # Check realistic ordering: DER++ should be best, SGD worst
        derpp_score = eri_scores['derpp']['mean']
        ewc_score = eri_scores['ewc_on']['mean']
        sgd_score = eri_scores['sgd']['mean']

        assert derpp_score < ewc_score < sgd_score

        # All scores should be reasonable
        for method, score_data in eri_scores.items():
            assert 0.0 <= score_data['mean'] <= 1.0
            assert score_data['ci'] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])
