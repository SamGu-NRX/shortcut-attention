"""
Statistical Analysis Module for Comparative Einstellung Analysis

This module provides statistical significance testing and effect size calculations
for comparing continual learning methods on key Einstellung metrics.

Integrates with the existing comparative experiment infrastructure:
- Works with aggregated CSV data from run_einstellung_experiment.py
- Uses ERI metrics computed by eri_vis/processing.py
- Provides statistical validation for comparative claims
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
from dataclasses import dataclass


@dataclass
class StatisticalTest:
    """
    Results of a statistical significance test.

    Attributes:
        test_name: Name of the statistical test performed
        statistic: Test statistic value
        p_value: P-value of the test
        effect_size: Effect size measure (Cohen's d, eta-squared, etc.)
        confidence_interval: 95% confidence interval for effect size
        interpretation: Text interpretation of the result
        significant: Whether result is significant at α=0.05
    """
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    significant: bool = False


@dataclass
class ComparisonResult:
    """
    Results of pairwise method comparison.

    Attributes:
        method1: First method name
        method2: Second method name
        metric: Metric being compared
        test_result: Statistical test results
        method1_stats: Descriptive statistics for method 1
        method2_stats: Descriptive statistics for method 2
    """
    method1: str
    method2: str
    metric: str
    test_result: StatisticalTest
    method1_stats: Dict[str, float]
    method2_stats: Dict[str, float]


class StatisticalAnalyzer:
    """
    Statistical analysis for comparative Einstellung experiments.

    Provides methods for:
    - Pairwise t-tests between methods
    - ANOVA for multi-group comparisons
    - Multiple comparison corrections (Bonferroni, FDR)
    - Effect size calculations (Cohen's d, eta-squared)
    - Confidence intervals for effect sizes
    """

    def __init__(self, alpha: float = 0.05, correction_method: str = 'bonferroni'):
        """
        Initialize the statistical analyzer.

        Args:
            alpha: Significance level (default 0.05)
            correction_method: Multiple comparison correction ('bonferroni', 'fdr', 'none')
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.logger = logging.getLogger(__name__)

        if correction_method not in ['bonferroni', 'fdr', 'none']:
            raise ValueError("correction_method must be 'bonferroni', 'fdr', or 'none'")

    def analyze_comparative_metrics(self, aggregated_csv_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on comparative metrics.

        Args:
            aggregated_csv_path: Path to aggregated CSV file with all methods

        Returns:
            Dictionary containing all statistical analysis results
        """
        try:
            # Load aggregated data
            df = pd.read_csv(aggregated_csv_path)

            # Validate required columns
            required_cols = ['method', 'seed', 'epoch_eff', 'split', 'acc']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Compute derived metrics for statistical analysis
            metrics_df = self._compute_statistical_metrics(df)

            # Perform statistical tests
            results = {
                'summary_statistics': self._compute_summary_statistics(metrics_df),
                'pairwise_comparisons': self._perform_pairwise_tests(metrics_df),
                'anova_results': self._perform_anova_tests(metrics_df),
                'effect_sizes': self._compute_effect_sizes(metrics_df),
                'multiple_comparisons': self._apply_multiple_comparison_corrections(metrics_df),
                'statistical_power': self._estimate_statistical_power(metrics_df),
                'interpretation': self._generate_interpretation(metrics_df)
            }

            self.logger.info(f"Statistical analysis completed for {len(df['method'].unique())} methods")
            return results

        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            return {'error': str(e)}

    def _compute_statistical_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
      """
      Compute key metrics for statistical analysis from raw accuracy data.

      Args:
         df: Raw accuracy data with columns [method, seed, epoch_eff, split, acc]

      Returns:
         DataFrame with computed metrics per method-seed combination
      """
      metrics_list = []

      # Group by method and seed to compute metrics per experimental run
      for (method, seed), group in df.groupby(['method', 'seed']):
         metrics = {'method': method, 'seed': seed}

         # Final accuracy (last epoch, any split - use T2_shortcut_normal if available)
         final_splits = ['T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal', 'T1_all']
         final_acc = None

         for split in final_splits:
               split_data = group[group['split'] == split]
               if len(split_data) > 0:
                  final_acc = split_data.loc[split_data['epoch_eff'].idxmax(), 'acc']
                  break

         if final_acc is not None:
               metrics['final_accuracy'] = final_acc

         # Performance Deficit (if we have both shortcut splits)
         shortcut_normal = group[group['split'] == 'T2_shortcut_normal']
         shortcut_masked = group[group['split'] == 'T2_shortcut_masked']

         if len(shortcut_normal) > 0 and len(shortcut_masked) > 0:
               # Use final epoch values
               normal_final = shortcut_normal.loc[shortcut_normal['epoch_eff'].idxmax(), 'acc']
               masked_final = shortcut_masked.loc[shortcut_masked['epoch_eff'].idxmax(), 'acc']

               if normal_final > 0:
                  metrics['performance_deficit'] = (normal_final - masked_final) / normal_final
               else:
                  metrics['performance_deficit'] = 0.0

         # Shortcut Feature Reliance
         nonshortcut_normal = group[group['split'] == 'T2_nonshortcut_normal']

         if len(shortcut_normal) > 0 and len(nonshortcut_normal) > 0:
               normal_final = shortcut_normal.loc[shortcut_normal['epoch_eff'].idxmax(), 'acc']
               nonshortcut_final = nonshortcut_normal.loc[nonshortcut_normal['epoch_eff'].idxmax(), 'acc']

               total_acc = normal_final + nonshortcut_final
               if total_acc > 0:
                  metrics['shortcut_reliance'] = normal_final / total_acc
               else:
                  metrics['shortcut_reliance'] = 0.0

         # Task 1 retention (negative transfer measure)
         t1_data = group[group['split'] == 'T1_all']
         if len(t1_data) > 0:
               metrics['t1_retention'] = t1_data.loc[t1_data['epoch_eff'].idxmax(), 'acc']

         # Adaptation speed (slope of accuracy improvement in first few epochs)
         if len(shortcut_normal) >= 3:  # Need at least 3 points for slope
               epochs = shortcut_normal['epoch_eff'].values[:3]
               accs = shortcut_normal['acc'].values[:3]
               if len(epochs) == len(accs) and len(set(epochs)) > 1:
                  slope, _, _, _, _ = stats.linregress(epochs, accs)
                  metrics['adaptation_speed'] = slope

         metrics_list.append(metrics)

      return pd.DataFrame(metrics_list)

    def _compute_summary_statistics(self, metrics_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute descriptive statistics for each method and metric."""
        summary = {}

        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'seed']

        for method in metrics_df['method'].unique():
            method_data = metrics_df[metrics_df['method'] == method]
            summary[method] = {}

            for metric in numeric_cols:
                if metric in method_data.columns:
                    values = method_data[metric].dropna()
                    if len(values) > 0:
                        summary[method][metric] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'median': float(values.median()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'n': len(values),
                            'sem': float(values.std() / np.sqrt(len(values))) if len(values) > 1 else 0.0
                        }

        return summary

    def _perform_pairwise_tests(self, metrics_df: pd.DataFrame) -> Dict[str, List[ComparisonResult]]:
        """Perform pairwise t-tests between all method pairs for each metric."""
        pairwise_results = {}

        methods = metrics_df['method'].unique()
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'seed']

        for metric in numeric_cols:
            pairwise_results[metric] = []

            # Get data for this metric
            metric_data = {}
            for method in methods:
                method_values = metrics_df[metrics_df['method'] == method][metric].dropna()
                if len(method_values) > 0:
                    metric_data[method] = method_values.values

            # Perform pairwise comparisons
            method_list = list(metric_data.keys())
            for i in range(len(method_list)):
                for j in range(i + 1, len(method_list)):
                    method1, method2 = method_list[i], method_list[j]

                    try:
                        comparison = self._compare_two_methods(
                            method1, method2, metric,
                            metric_data[method1], metric_data[method2]
                        )
                        pairwise_results[metric].append(comparison)
                    except Exception as e:
                        self.logger.warning(f"Failed pairwise test {method1} vs {method2} on {metric}: {e}")

        return pairwise_results

    def _compare_two_methods(self, method1: str, method2: str, metric: str,
                           data1: np.ndarray, data2: np.ndarray) -> ComparisonResult:
        """Perform statistical comparison between two methods on a specific metric."""

        # Perform t-test
        if len(data1) < 2 or len(data2) < 2:
            # Not enough data for t-test
            test_result = StatisticalTest(
                test_name="t-test",
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient data for statistical test"
            )
        else:
            # Welch's t-test (unequal variances)
            statistic, p_value = ttest_ind(data1, data2, equal_var=False)

            # Compute Cohen's d effect size
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) +
                                 (len(data2) - 1) * np.var(data2, ddof=1)) /
                                (len(data1) + len(data2) - 2))

            if pooled_std > 0:
                cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
            else:
                cohens_d = 0.0

            # Effect size interpretation
            if abs(cohens_d) < 0.2:
                effect_interp = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interp = "small"
            elif abs(cohens_d) < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"

            # Statistical significance
            significant = p_value < self.alpha

            interpretation = f"{'Significant' if significant else 'Non-significant'} difference (p={p_value:.4f}), {effect_interp} effect size (d={cohens_d:.3f})"

            test_result = StatisticalTest(
                test_name="Welch's t-test",
                statistic=statistic,
                p_value=p_value,
                effect_size=cohens_d,
                interpretation=interpretation,
                significant=significant
            )

        # Descriptive statistics
        method1_stats = {
            'mean': float(np.mean(data1)),
            'std': float(np.std(data1, ddof=1)) if len(data1) > 1 else 0.0,
            'n': len(data1)
        }

        method2_stats = {
            'mean': float(np.mean(data2)),
            'std': float(np.std(data2, ddof=1)) if len(data2) > 1 else 0.0,
            'n': len(data2)
        }

        return ComparisonResult(
            method1=method1,
            method2=method2,
            metric=metric,
            test_result=test_result,
            method1_stats=method1_stats,
            method2_stats=method2_stats
        )

    def _perform_anova_tests(self, metrics_df: pd.DataFrame) -> Dict[str, StatisticalTest]:
        """Perform one-way ANOVA for each metric across all methods."""
        anova_results = {}

        methods = metrics_df['method'].unique()
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'seed']

        for metric in numeric_cols:
            # Collect data for all methods
            method_groups = []
            for method in methods:
                method_values = metrics_df[metrics_df['method'] == method][metric].dropna()
                if len(method_values) > 0:
                    method_groups.append(method_values.values)

            if len(method_groups) < 2:
                anova_results[metric] = StatisticalTest(
                    test_name="One-way ANOVA",
                    statistic=np.nan,
                    p_value=np.nan,
                    interpretation="Insufficient groups for ANOVA"
                )
                continue

            # Check if all groups have sufficient data
            if any(len(group) < 2 for group in method_groups):
                anova_results[metric] = StatisticalTest(
                    test_name="One-way ANOVA",
                    statistic=np.nan,
                    p_value=np.nan,
                    interpretation="Insufficient data in one or more groups"
                )
                continue

            try:
                # Perform one-way ANOVA
                f_statistic, p_value = f_oneway(*method_groups)

                # Compute eta-squared (effect size for ANOVA)
                # eta² = SS_between / SS_total
                all_data = np.concatenate(method_groups)
                grand_mean = np.mean(all_data)

                ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in method_groups)
                ss_total = sum((x - grand_mean)**2 for x in all_data)

                eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

                # Effect size interpretation for eta-squared
                if eta_squared < 0.01:
                    effect_interp = "negligible"
                elif eta_squared < 0.06:
                    effect_interp = "small"
                elif eta_squared < 0.14:
                    effect_interp = "medium"
                else:
                    effect_interp = "large"

                significant = p_value < self.alpha
                interpretation = f"{'Significant' if significant else 'Non-significant'} group differences (p={p_value:.4f}), {effect_interp} effect size (η²={eta_squared:.3f})"

                anova_results[metric] = StatisticalTest(
                    test_name="One-way ANOVA",
                    statistic=f_statistic,
                    p_value=p_value,
                    effect_size=eta_squared,
                    interpretation=interpretation,
                    significant=significant
                )

            except Exception as e:
                self.logger.warning(f"ANOVA failed for {metric}: {e}")
                anova_results[metric] = StatisticalTest(
                    test_name="One-way ANOVA",
                    statistic=np.nan,
                    p_value=np.nan,
                    interpretation=f"ANOVA failed: {str(e)}"
                )

        return anova_results

    def _apply_multiple_comparison_corrections(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Apply multiple comparison corrections to control family-wise error rate."""
        if self.correction_method == 'none':
            return {'correction_method': 'none', 'note': 'No multiple comparison correction applied'}

        # Get all p-values from pairwise tests
        pairwise_results = self._perform_pairwise_tests(metrics_df)

        corrections = {}

        for metric, comparisons in pairwise_results.items():
            p_values = [comp.test_result.p_value for comp in comparisons if not np.isnan(comp.test_result.p_value)]

            if len(p_values) == 0:
                continue

            if self.correction_method == 'bonferroni':
                # Bonferroni correction
                corrected_alpha = self.alpha / len(p_values)
                corrected_p_values = [min(p * len(p_values), 1.0) for p in p_values]

            elif self.correction_method == 'fdr':
                # Benjamini-Hochberg FDR correction
                from scipy.stats import false_discovery_control
                corrected_p_values = false_discovery_control(p_values, alpha=self.alpha)
                corrected_alpha = self.alpha

            corrections[metric] = {
                'original_p_values': p_values,
                'corrected_p_values': corrected_p_values,
                'corrected_alpha': corrected_alpha,
                'n_comparisons': len(p_values),
                'significant_after_correction': sum(1 for p in corrected_p_values if p < corrected_alpha)
            }

        corrections['correction_method'] = self.correction_method
        return corrections

    def _compute_effect_sizes(self, metrics_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute effect sizes for all pairwise comparisons."""
        effect_sizes = {}

        methods = metrics_df['method'].unique()
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'seed']

        for metric in numeric_cols:
            effect_sizes[metric] = {}

            # Get data for this metric
            metric_data = {}
            for method in methods:
                method_values = metrics_df[metrics_df['method'] == method][metric].dropna()
                if len(method_values) > 0:
                    metric_data[method] = method_values.values

            # Compute pairwise effect sizes
            method_list = list(metric_data.keys())
            for i in range(len(method_list)):
                for j in range(i + 1, len(method_list)):
                    method1, method2 = method_list[i], method_list[j]

                    data1, data2 = metric_data[method1], metric_data[method2]

                    if len(data1) >= 2 and len(data2) >= 2:
                        # Cohen's d
                        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) +
                                             (len(data2) - 1) * np.var(data2, ddof=1)) /
                                            (len(data1) + len(data2) - 2))

                        if pooled_std > 0:
                            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                            effect_sizes[metric][f"{method1}_vs_{method2}"] = cohens_d

        return effect_sizes

    def _estimate_statistical_power(self, metrics_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Estimate statistical power for detecting differences between methods."""
        power_analysis = {}

        methods = metrics_df['method'].unique()
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'seed']

        for metric in numeric_cols:
            power_analysis[metric] = {}

            # Estimate power based on observed effect sizes and sample sizes
            for method in methods:
                method_data = metrics_df[metrics_df['method'] == method][metric].dropna()
                if len(method_data) > 1:
                    n = len(method_data)
                    std = method_data.std()

                    # Estimate power to detect small (d=0.2), medium (d=0.5), and large (d=0.8) effects
                    effect_sizes = [0.2, 0.5, 0.8]

                    for effect_size in effect_sizes:
                        # Simplified power calculation (assumes equal group sizes)
                        # More sophisticated power analysis would use scipy.stats.power
                        delta = effect_size * std
                        se = std * np.sqrt(2/n)  # Standard error for two-sample test

                        if se > 0:
                            z_score = delta / se
                            # Approximate power using normal distribution
                            power = 1 - stats.norm.cdf(1.96 - z_score)  # Two-tailed test
                            power_analysis[metric][f"{method}_power_d{effect_size}"] = min(power, 1.0)

        return power_analysis

    def _generate_interpretation(self, metrics_df: pd.DataFrame) -> Dict[str, str]:
        """Generate human-readable interpretation of statistical results."""

        # Get summary statistics
        summary_stats = self._compute_summary_statistics(metrics_df)

        # Get ANOVA results
        anova_results = self._perform_anova_tests(metrics_df)

        interpretations = {}

        # Overall interpretation
        n_methods = len(metrics_df['method'].unique())
        n_metrics = len([col for col in metrics_df.select_dtypes(include=[np.number]).columns if col != 'seed'])

        interpretations['overview'] = f"""
Statistical analysis of {n_methods} continual learning methods across {n_metrics} Einstellung metrics.
Significance level: α = {self.alpha}
Multiple comparison correction: {self.correction_method}
        """.strip()

        # Method performance ranking
        if 'final_accuracy' in summary_stats[list(summary_stats.keys())[0]]:
            method_accs = [(method, stats['final_accuracy']['mean'])
                          for method, stats in summary_stats.items()
                          if 'final_accuracy' in stats]
            method_accs.sort(key=lambda x: x[1], reverse=True)

            interpretations['performance_ranking'] = f"""
Final accuracy ranking (highest to lowest):
{chr(10).join([f"{i+1}. {method}: {acc:.2f}%" for i, (method, acc) in enumerate(method_accs)])}
            """.strip()

        # Significant differences
        significant_metrics = [metric for metric, result in anova_results.items()
                             if result.significant]

        if significant_metrics:
            interpretations['significant_differences'] = f"""
Metrics showing significant differences between methods:
{chr(10).join([f"• {metric}: {anova_results[metric].interpretation}" for metric in significant_metrics])}
            """.strip()
        else:
            interpretations['significant_differences'] = "No significant differences detected between methods on any metric."

        # Effect size interpretation
        effect_sizes = self._compute_effect_sizes(metrics_df)
        large_effects = []

        for metric, comparisons in effect_sizes.items():
            for comparison, effect_size in comparisons.items():
                if abs(effect_size) >= 0.8:  # Large effect
                    large_effects.append(f"{comparison} on {metric} (d={effect_size:.3f})")

        if large_effects:
            interpretations['large_effects'] = f"""
Comparisons with large effect sizes (|d| ≥ 0.8):
{chr(10).join([f"• {effect}" for effect in large_effects[:5]])}  # Limit to top 5
            """.strip()
        else:
            interpretations['large_effects'] = "No large effect sizes detected between methods."

        return interpretations


def generate_statistical_report(aggregated_csv_path: str, output_dir: str) -> str:
    """
    Generate comprehensive statistical analysis report.

    Args:
        aggregated_csv_path: Path to aggregated CSV with all methods
        output_dir: Directory to save statistical report

    Returns:
        Path to generated statistical report HTML file
    """
    import os
    from datetime import datetime

    # Perform statistical analysis
    analyzer = StatisticalAnalyzer(alpha=0.05, correction_method='bonferroni')
    results = analyzer.analyze_comparative_metrics(aggregated_csv_path)

    if 'error' in results:
        return f"Statistical analysis failed: {results['error']}"

    # Generate HTML report
    report_path = os.path.join(output_dir, "statistical_analysis_report.html")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Statistical Analysis Report - Comparative Einstellung Analysis</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #007bff; }}
        .significant {{ color: #28a745; font-weight: bold; }}
        .non-significant {{ color: #6c757d; }}
        .large-effect {{ color: #dc3545; font-weight: bold; }}
        .medium-effect {{ color: #ffc107; font-weight: bold; }}
        .small-effect {{ color: #17a2b8; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Statistical Analysis Report</h1>
        <h2>Comparative Einstellung Effect Analysis</h2>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Analysis:</strong> Statistical significance testing and effect size calculations</p>
    </div>

    <div class="section">
        <h2>📋 Analysis Overview</h2>
        <pre>{results.get('interpretation', {}).get('overview', 'No overview available')}</pre>
    </div>

    <div class="section">
        <h2>🏆 Method Performance Ranking</h2>
        <pre>{results.get('interpretation', {}).get('performance_ranking', 'No ranking available')}</pre>
    </div>

    <div class="section">
        <h2>🔍 Significant Differences (ANOVA Results)</h2>
        <pre>{results.get('interpretation', {}).get('significant_differences', 'No significant differences')}</pre>

        <h3>Detailed ANOVA Results</h3>
        <table>
            <tr><th>Metric</th><th>F-statistic</th><th>p-value</th><th>Effect Size (η²)</th><th>Interpretation</th></tr>"""

    # Add ANOVA results to table
    anova_results = results.get('anova_results', {})
    for metric, result in anova_results.items():
        significance_class = 'significant' if result.significant else 'non-significant'
        effect_size_str = f"{result.effect_size:.3f}" if result.effect_size is not None else "N/A"

        html_content += f"""
            <tr>
                <td><strong>{metric}</strong></td>
                <td>{result.statistic:.3f}</td>
                <td class="{significance_class}">{result.p_value:.4f}</td>
                <td>{effect_size_str}</td>
                <td>{result.interpretation}</td>
            </tr>"""

    html_content += """
        </table>
    </div>

    <div class="section">
        <h2>📏 Effect Sizes</h2>
        <pre>{}</pre>

        <h3>Pairwise Comparisons with Large Effects</h3>
        <table>
            <tr><th>Comparison</th><th>Metric</th><th>Method 1 Mean</th><th>Method 2 Mean</th><th>Cohen's d</th><th>Interpretation</th></tr>""".format(
        results.get('interpretation', {}).get('large_effects', 'No large effects detected')
    )

    # Add pairwise comparisons with large effects
    pairwise_results = results.get('pairwise_comparisons', {})
    for metric, comparisons in pairwise_results.items():
        for comp in comparisons:
            if comp.test_result.effect_size is not None and abs(comp.test_result.effect_size) >= 0.5:  # Medium+ effects
                effect_class = 'large-effect' if abs(comp.test_result.effect_size) >= 0.8 else 'medium-effect'

                html_content += f"""
            <tr>
                <td>{comp.method1} vs {comp.method2}</td>
                <td>{metric}</td>
                <td>{comp.method1_stats['mean']:.3f}</td>
                <td>{comp.method2_stats['mean']:.3f}</td>
                <td class="{effect_class}">{comp.test_result.effect_size:.3f}</td>
                <td>{comp.test_result.interpretation}</td>
            </tr>"""

    html_content += """
        </table>
    </div>

    <div class="section">
        <h2>📈 Summary Statistics</h2>
        <div class="metric-grid">"""

    # Add summary statistics cards
    summary_stats = results.get('summary_statistics', {})
    for method, stats in summary_stats.items():
        html_content += f"""
            <div class="metric-card">
                <h4>{method}</h4>
                <table>
                    <tr><th>Metric</th><th>Mean ± SEM</th><th>N</th></tr>"""

        for metric, values in stats.items():
            if isinstance(values, dict) and 'mean' in values:
                html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{values['mean']:.3f} ± {values['sem']:.3f}</td>
                        <td>{values['n']}</td>
                    </tr>"""

        html_content += """
                </table>
            </div>"""

    html_content += """
        </div>
    </div>

    <div class="section">
        <h2>🔧 Multiple Comparison Corrections</h2>
        <pre>Correction method: {}</pre>
    </div>

    <hr>
    <p><em>Statistical analysis generated by Mammoth Comparative Einstellung Analysis System</em></p>
</body>
</html>""".format(results.get('multiple_comparisons', {}).get('correction_method', 'Unknown'))

    # Write HTML file
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(html_content)

    return report_path
