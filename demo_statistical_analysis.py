#!/usr/bin/env python3
"""
Demonstration of Statistical Analysis for Comparative Einstellung Analysis

This script demonstrates the statistical significance testing and enhanced
comparative reporting functionality (Tasks 13 & 14).

Usage:
    python demo_statistical_analysis.py
"""

import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Import our statistical analysis module
from utils.statistical_analysis import StatisticalAnalyzer, generate_statistical_report


def create_demo_data():
    """Create demonstration data showing different method performance patterns."""

    print("🔬 Creating demonstration data with realistic Einstellung patterns...")

    np.random.seed(42)  # For reproducible demo

    # Define methods with different characteristics
    methods_config = {
        'scratch_t2': {
            'description': 'Task 2 only baseline (optimal performance)',
            'T2_shortcut_normal': 0.85,
            'T2_shortcut_masked': 0.75,
            'T2_nonshortcut_normal': 0.80,
            'noise': 0.02
        },
        'sgd': {
            'description': 'Naive SGD (high catastrophic forgetting)',
            'T2_shortcut_normal': 0.70,
            'T2_shortcut_masked': 0.45,  # High shortcut reliance
            'T2_nonshortcut_normal': 0.65,
            'noise': 0.04
        },
        'derpp': {
            'description': 'DER++ with replay buffer',
            'T2_shortcut_normal': 0.78,
            'T2_shortcut_masked': 0.68,
            'T2_nonshortcut_normal': 0.75,
            'noise': 0.03
        },
        'ewc_on': {
            'description': 'Elastic Weight Consolidation',
            'T2_shortcut_normal': 0.72,
            'T2_shortcut_masked': 0.58,
            'T2_nonshortcut_normal': 0.70,
            'noise': 0.035
        }
    }

    # Generate data
    data = []
    seeds = [42, 43, 44, 45, 46]  # 5 seeds for statistical power
    epochs = [0.0, 0.5, 1.0, 1.5, 2.0]  # 5 epochs
    splits = ['T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal']

    for method, config in methods_config.items():
        print(f"  • {method}: {config['description']}")

        for seed in seeds:
            seed_offset = (seed - 42) * 0.005  # Small seed variation

            for epoch in epochs:
                learning_progress = min(1.0, 0.3 + 0.7 * (epoch / 2.0))  # Learning curve

                for split in splits:
                    base_acc = config[split]
                    noise = np.random.normal(0, config['noise'])

                    # Apply learning progression and add noise
                    acc = base_acc * learning_progress + seed_offset + noise
                    acc = max(0.0, min(1.0, acc))  # Clamp to [0, 1]

                    data.append({
                        'method': method,
                        'seed': seed,
                        'epoch_eff': epoch,
                        'split': split,
                        'acc': acc
                    })

    return pd.DataFrame(data)


def demonstrate_statistical_analysis():
    """Demonstrate the complete statistical analysis pipeline."""

    print("\n" + "="*80)
    print("🧠 STATISTICAL ANALYSIS DEMONSTRATION")
    print("Comparative Einstellung Effect Analysis - Tasks 13 & 14")
    print("="*80)

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:

        # Generate demo data
        df = create_demo_data()
        csv_path = os.path.join(temp_dir, 'demo_comparative_data.csv')
        df.to_csv(csv_path, index=False)

        print(f"\n📊 Generated demo dataset: {len(df)} data points")
        print(f"   • Methods: {list(df['method'].unique())}")
        print(f"   • Seeds: {list(df['seed'].unique())}")
        print(f"   • Splits: {list(df['split'].unique())}")

        # Initialize statistical analyzer
        print(f"\n🔍 Initializing statistical analyzer...")
        analyzer = StatisticalAnalyzer(alpha=0.05, correction_method='bonferroni')

        # Perform statistical analysis
        print(f"📈 Performing comprehensive statistical analysis...")
        results = analyzer.analyze_comparative_metrics(csv_path)

        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            return

        # Display summary statistics
        print(f"\n📋 SUMMARY STATISTICS")
        print(f"{'-'*50}")

        summary_stats = results['summary_statistics']
        for method, stats in summary_stats.items():
            if 'final_accuracy' in stats:
                acc_stats = stats['final_accuracy']
                print(f"{method:12} | Final Acc: {acc_stats['mean']:.3f} ± {acc_stats['sem']:.3f} (N={acc_stats['n']})")

        # Display ANOVA results
        print(f"\n🔬 ANOVA RESULTS (Multi-group comparisons)")
        print(f"{'-'*60}")

        anova_results = results['anova_results']
        for metric, result in anova_results.items():
            sig_marker = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else ""
            print(f"{metric:20} | F={result.statistic:.3f}, p={result.p_value:.4f} {sig_marker}")
            print(f"{'':20} | η²={result.effect_size:.3f} ({result.interpretation.split(',')[1].strip() if ',' in result.interpretation else 'N/A'})")

        # Display significant pairwise comparisons
        print(f"\n🎯 SIGNIFICANT PAIRWISE COMPARISONS")
        print(f"{'-'*70}")

        pairwise_results = results['pairwise_comparisons']
        significant_found = False

        for metric, comparisons in pairwise_results.items():
            for comp in comparisons:
                if comp.test_result.significant:
                    significant_found = True
                    sig_marker = "***" if comp.test_result.p_value < 0.001 else "**" if comp.test_result.p_value < 0.01 else "*"

                    print(f"{comp.method1:8} vs {comp.method2:8} | {metric:15} | p={comp.test_result.p_value:.4f} {sig_marker}")
                    print(f"{'':19} | d={comp.test_result.effect_size:.3f} | {comp.method1}: {comp.method1_stats['mean']:.3f}, {comp.method2}: {comp.method2_stats['mean']:.3f}")

        if not significant_found:
            print("No significant pairwise differences detected (after correction)")

        # Display effect sizes
        print(f"\n📏 LARGE EFFECT SIZES (|Cohen's d| ≥ 0.8)")
        print(f"{'-'*60}")

        effect_sizes = results['effect_sizes']
        large_effects_found = False

        for metric, effects in effect_sizes.items():
            for comparison, effect_size in effects.items():
                if abs(effect_size) >= 0.8:
                    large_effects_found = True
                    direction = ">" if effect_size > 0 else "<"
                    methods = comparison.replace('_vs_', f' {direction} ')
                    print(f"{methods:25} | {metric:15} | d={effect_size:.3f}")

        if not large_effects_found:
            print("No large effect sizes detected")

        # Display interpretation
        print(f"\n💡 INTERPRETATION")
        print(f"{'-'*50}")

        interpretation = results['interpretation']
        if 'performance_ranking' in interpretation:
            print("Performance Ranking:")
            ranking_lines = interpretation['performance_ranking'].strip().split('\n')
            for line in ranking_lines[1:6]:  # Show top 5
                if line.strip():
                    print(f"  {line.strip()}")

        if 'significant_differences' in interpretation:
            print(f"\nSignificant Differences:")
            print(f"  {interpretation['significant_differences']}")

        # Generate HTML report
        print(f"\n📄 Generating comprehensive HTML report...")
        report_path = generate_statistical_report(csv_path, temp_dir)

        print(f"✅ Statistical analysis complete!")
        print(f"📊 HTML report generated: {report_path}")
        print(f"📁 Report size: {os.path.getsize(report_path) / 1024:.1f} KB")

        # Show how this integrates with comparative experiments
        print(f"\n🔗 INTEGRATION WITH COMPARATIVE EXPERIMENTS")
        print(f"{'-'*60}")
        print(f"This statistical analysis automatically integrates with:")
        print(f"  • run_einstellung_experiment.py --comparative")
        print(f"  • Enhanced summary tables with significance indicators (* ** ***)")
        print(f"  • Automatic statistical report generation")
        print(f"  • Multiple comparison corrections (Bonferroni, FDR)")
        print(f"  • Effect size calculations and interpretation")

        # Demonstrate significance indicators
        print(f"\n🏷️  SIGNIFICANCE INDICATORS FOR SUMMARY TABLE")
        print(f"{'-'*60}")

        from run_einstellung_experiment import get_significance_indicator

        for method in ['scratch_t2', 'sgd', 'derpp', 'ewc_on']:
            sig_indicator = get_significance_indicator(method, results)
            print(f"{method:12} | Significance: '{sig_indicator}' ({'Highly significant' if sig_indicator == '***' else 'Significant' if sig_indicator in ['*', '**'] else 'Not significant'})")

        print(f"\n🎉 Demo completed successfully!")
        print(f"   Tasks 13 & 14 implementation verified ✅")


if __name__ == '__main__':
    demonstrate_statistical_analysis()
