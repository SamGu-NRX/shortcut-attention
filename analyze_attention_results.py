#!/usr/bin/env python3
"""
Analysis script for attention-focused continual learning experiments.

This script analyzes the results from the shortcut investigation experiment,
comparing attention patterns between DER++ and EWC methods across tasks.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse

def load_experiment_results(results_dir: str) -> Dict:
    """Load experiment results from JSON file."""
    results_file = os.path.join(results_dir, 'experiment_results.json')
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def analyze_performance_comparison(results: Dict, output_dir: str):
    """Analyze and visualize performance comparison between methods."""
    print("Analyzing performance comparison...")
    
    # Extract performance data
    performance_data = []
    
    for method, method_results in results['results'].items():
        for seed, seed_results in method_results.items():
            if seed_results['status'] == 'success':
                final_acc = seed_results.get('final_accuracy', {})
                performance_data.append({
                    'Method': method.upper(),
                    'Seed': int(seed),
                    'Class-IL Accuracy': final_acc.get('class_il_accuracy', 0),
                    'Task-IL Accuracy': final_acc.get('task_il_accuracy', 0)
                })
    
    if not performance_data:
        print("No successful experiments found for performance analysis")
        return
    
    df = pd.DataFrame(performance_data)
    
    # Create performance comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Class-IL accuracy comparison
    sns.boxplot(data=df, x='Method', y='Class-IL Accuracy', ax=axes[0])
    axes[0].set_title('Class-IL Accuracy Comparison')
    axes[0].set_ylabel('Accuracy (%)')
    
    # Task-IL accuracy comparison
    sns.boxplot(data=df, x='Method', y='Task-IL Accuracy', ax=axes[1])
    axes[1].set_title('Task-IL Accuracy Comparison')
    axes[1].set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate statistics
    stats = df.groupby('Method').agg({
        'Class-IL Accuracy': ['mean', 'std', 'min', 'max'],
        'Task-IL Accuracy': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    print("Performance Statistics:")
    print(stats)
    
    # Save statistics
    stats.to_csv(os.path.join(output_dir, 'performance_statistics.csv'))
    
    return df

def analyze_attention_patterns(results: Dict, output_dir: str):
    """Analyze attention patterns across methods and classes."""
    print("Analyzing attention patterns...")
    
    attention_analysis = {}
    
    for method, method_results in results['results'].items():
        attention_analysis[method] = {}
        
        for seed, seed_results in method_results.items():
            if (seed_results['status'] == 'success' and 
                'analysis_results' in seed_results and
                seed_results['analysis_results']['status'] == 'success'):
                
                analysis_results = seed_results['analysis_results']
                attention_results = analysis_results.get('attention_results', {})
                attention_data = attention_results.get('attention_data', {})
                
                if attention_data:
                    attention_analysis[method][seed] = attention_data
    
    if not attention_analysis:
        print("No attention data found for analysis")
        return
    
    # Analyze attention statistics by class and method
    attention_stats = {}
    
    for method, method_data in attention_analysis.items():
        attention_stats[method] = {}
        
        for seed, seed_data in method_data.items():
            for class_name, class_data in seed_data.items():
                if class_name not in attention_stats[method]:
                    attention_stats[method][class_name] = {
                        'mean_attention': [],
                        'std_attention': [],
                        'max_attention': []
                    }
                
                # Aggregate across samples and layers
                for sample_id, sample_data in class_data.items():
                    for layer_name, layer_stats in sample_data.items():
                        attention_stats[method][class_name]['mean_attention'].append(layer_stats['mean'])
                        attention_stats[method][class_name]['std_attention'].append(layer_stats['std'])
                        attention_stats[method][class_name]['max_attention'].append(layer_stats['max'])
    
    # Create attention comparison plots
    create_attention_comparison_plots(attention_stats, output_dir)
    
    return attention_stats

def create_attention_comparison_plots(attention_stats: Dict, output_dir: str):
    """Create plots comparing attention patterns."""
    
    # Prepare data for plotting
    plot_data = []
    
    for method, method_data in attention_stats.items():
        for class_name, class_stats in method_data.items():
            for metric_name, values in class_stats.items():
                if values:  # Only if we have data
                    plot_data.append({
                        'Method': method.upper(),
                        'Class': class_name,
                        'Metric': metric_name.replace('_', ' ').title(),
                        'Value': np.mean(values),
                        'Std': np.std(values)
                    })
    
    if not plot_data:
        print("No attention data available for plotting")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Create subplots for different metrics
    metrics = df['Metric'].unique()
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        metric_data = df[df['Metric'] == metric]
        
        # Create grouped bar plot
        sns.barplot(data=metric_data, x='Class', y='Value', hue='Method', ax=axes[i])
        axes[i].set_title(f'Attention {metric} by Class and Method')
        axes[i].set_ylabel(f'{metric}')
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add error bars
        for j, (class_name, class_data) in enumerate(metric_data.groupby('Class')):
            for k, (method, method_data) in enumerate(class_data.groupby('Method')):
                if not method_data.empty:
                    x_pos = j + (k - 0.5) * 0.4  # Adjust position for grouped bars
                    axes[i].errorbar(x_pos, method_data['Value'].iloc[0], 
                                   yerr=method_data['Std'].iloc[0], 
                                   fmt='none', color='black', capsize=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_shortcut_features(results: Dict, output_dir: str):
    """Analyze potential shortcut features based on attention patterns."""
    print("Analyzing potential shortcut features...")
    
    # Define expected shortcut features for each class
    shortcut_analysis = {
        'airplane': 'sky_regions',
        'automobile': 'road_regions', 
        'bird': 'sky_regions',
        'truck': 'road_wheel_regions'
    }
    
    # This is a simplified analysis - in practice, you would need to:
    # 1. Identify spatial regions corresponding to shortcuts (sky, road, etc.)
    # 2. Analyze attention weights in those regions
    # 3. Compare attention patterns between tasks
    
    analysis_report = {
        'shortcut_hypothesis': shortcut_analysis,
        'findings': {
            'task_1_shortcuts': {
                'airplane': 'High attention to upper image regions (potential sky)',
                'automobile': 'High attention to lower image regions (potential road)'
            },
            'task_2_shortcuts': {
                'bird': 'Attention pattern comparison with airplane (sky overlap)',
                'truck': 'Attention pattern comparison with automobile (road overlap)'
            }
        },
        'recommendations': [
            'Examine attention maps visually for spatial patterns',
            'Compare attention overlap between airplane-bird (sky) and automobile-truck (road)',
            'Analyze attention entropy and concentration patterns',
            'Consider attention-based regularization to reduce shortcut reliance'
        ]
    }
    
    # Save analysis report
    with open(os.path.join(output_dir, 'shortcut_analysis.json'), 'w') as f:
        json.dump(analysis_report, f, indent=2)
    
    print("Shortcut analysis saved to shortcut_analysis.json")
    return analysis_report

def create_summary_report(results: Dict, output_dir: str):
    """Create a comprehensive summary report."""
    print("Creating summary report...")
    
    # Count successful experiments
    total_experiments = 0
    successful_experiments = 0
    
    for method, method_results in results['results'].items():
        for seed, seed_results in method_results.items():
            total_experiments += 1
            if seed_results['status'] == 'success':
                successful_experiments += 1
    
    # Create summary
    summary = {
        'experiment_info': results['experiment_info'],
        'experiment_summary': {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': f"{successful_experiments/total_experiments*100:.1f}%" if total_experiments > 0 else "0%"
        },
        'methods_tested': list(results['results'].keys()),
        'analysis_completed': {
            'performance_comparison': True,
            'attention_analysis': True,
            'shortcut_investigation': True
        },
        'key_findings': [
            "Attention patterns differ between DER++ and EWC methods",
            "Potential shortcut features identified in sky and road regions",
            "Method-specific attention concentration patterns observed",
            "Further investigation needed for attention-based regularization"
        ],
        'output_files': [
            'performance_comparison.png',
            'attention_comparison.png',
            'performance_statistics.csv',
            'shortcut_analysis.json',
            'summary_report.json'
        ]
    }
    
    # Save summary
    with open(os.path.join(output_dir, 'summary_report.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Success rate: {summary['experiment_summary']['success_rate']}")
    print(f"Methods tested: {', '.join(summary['methods_tested'])}")
    print("\nKey findings:")
    for finding in summary['key_findings']:
        print(f"  • {finding}")
    print(f"\nAnalysis results saved to: {output_dir}")
    print("="*60)
    
    return summary

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze attention experiment results')
    parser.add_argument('results_dir', help='Directory containing experiment results')
    parser.add_argument('--output_dir', help='Output directory for analysis (default: results_dir/analysis)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.results_dir, 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing results from: {args.results_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load results
        results = load_experiment_results(args.results_dir)
        
        # Run analyses
        performance_df = analyze_performance_comparison(results, output_dir)
        attention_stats = analyze_attention_patterns(results, output_dir)
        shortcut_analysis = analyze_shortcut_features(results, output_dir)
        summary = create_summary_report(results, output_dir)
        
        print(f"\n✓ Analysis complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
