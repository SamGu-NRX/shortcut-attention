"""
Analysis script for shortcut investigation experiment results.

This script analyzes the results from the shortcut investigation experiments,
comparing attention patterns and network activations between different methods.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
import argparse

# Add mammoth path
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, mammoth_path)


class ResultsAnalyzer:
    """
    Class for analyzing experiment results and generating comparative visualizations.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize the results analyzer.
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = results_dir
        self.results_file = os.path.join(results_dir, 'experiment_results.json')
        
        # Load results
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        else:
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        self.analysis_dir = os.path.join(results_dir, 'analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)
    
    def compare_attention_patterns(self) -> None:
        """
        Compare attention patterns between different methods.
        """
        print("Analyzing attention patterns...")
        
        methods = list(self.results['methods'].keys())
        
        # Collect attention statistics
        attention_data = []
        
        for method in methods:
            method_results = self.results['methods'][method]
            
            for seed, seed_results in method_results.items():
                if 'task_analyses' in seed_results and 'final' in seed_results['task_analyses']:
                    final_analysis = seed_results['task_analyses']['final']
                    
                    if 'attention_stats' in final_analysis:
                        attention_stats = final_analysis['attention_stats']
                        
                        for layer_name, stats in attention_stats.items():
                            attention_data.append({
                                'method': method,
                                'seed': seed,
                                'layer': layer_name,
                                'attention_entropy': stats['attention_entropy']
                            })
        
        if attention_data:
            # Create DataFrame
            df = pd.DataFrame(attention_data)
            
            # Plot attention entropy comparison
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df, x='layer', y='attention_entropy', hue='method')
            plt.title('Attention Entropy Comparison Between Methods')
            plt.xlabel('Layer')
            plt.ylabel('Attention Entropy')
            plt.xticks(rotation=45)
            plt.legend(title='Method')
            plt.tight_layout()
            plt.savefig(os.path.join(self.analysis_dir, 'attention_entropy_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Statistical summary
            summary = df.groupby(['method', 'layer'])['attention_entropy'].agg(['mean', 'std']).reset_index()
            summary.to_csv(os.path.join(self.analysis_dir, 'attention_entropy_summary.csv'), index=False)
            
            print(f"Attention analysis saved to {self.analysis_dir}")
        else:
            print("No attention data found for analysis")
    
    def analyze_method_performance(self) -> None:
        """
        Analyze and compare performance metrics between methods.
        """
        print("Analyzing method performance...")
        
        methods = list(self.results['methods'].keys())
        performance_data = []
        
        for method in methods:
            method_results = self.results['methods'][method]
            
            for seed, seed_results in method_results.items():
                # Try to extract performance metrics from results
                # Note: This would need to be adapted based on actual result structure
                exp_results = seed_results.get('experiment_results', {})
                
                if exp_results.get('status') == 'success':
                    performance_data.append({
                        'method': method,
                        'seed': seed,
                        'status': 'success'
                    })
                else:
                    performance_data.append({
                        'method': method,
                        'seed': seed,
                        'status': 'failed'
                    })
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            
            # Success rate by method
            success_rate = df.groupby('method')['status'].apply(
                lambda x: (x == 'success').sum() / len(x)
            ).reset_index()
            success_rate.columns = ['method', 'success_rate']
            
            plt.figure(figsize=(8, 6))
            sns.barplot(data=success_rate, x='method', y='success_rate')
            plt.title('Experiment Success Rate by Method')
            plt.ylabel('Success Rate')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.analysis_dir, 'success_rate_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            success_rate.to_csv(os.path.join(self.analysis_dir, 'success_rate_summary.csv'), index=False)
            
            print(f"Performance analysis saved to {self.analysis_dir}")
    
    def generate_summary_report(self) -> None:
        """
        Generate a comprehensive summary report.
        """
        print("Generating summary report...")
        
        report_lines = []
        report_lines.append("# Shortcut Investigation Experiment Summary")
        report_lines.append(f"Experiment: {self.results['experiment_name']}")
        report_lines.append(f"Timestamp: {self.results['timestamp']}")
        report_lines.append("")
        
        # Methods and seeds
        methods = list(self.results['methods'].keys())
        report_lines.append(f"Methods tested: {', '.join(methods)}")
        
        # Count total experiments
        total_experiments = 0
        successful_experiments = 0
        
        for method in methods:
            method_results = self.results['methods'][method]
            seeds = list(method_results.keys())
            total_experiments += len(seeds)
            
            for seed, seed_results in method_results.items():
                if seed_results.get('experiment_results', {}).get('status') == 'success':
                    successful_experiments += 1
        
        report_lines.append(f"Total experiments: {total_experiments}")
        report_lines.append(f"Successful experiments: {successful_experiments}")
        report_lines.append(f"Success rate: {successful_experiments/total_experiments:.2%}")
        report_lines.append("")
        
        # Method-specific results
        report_lines.append("## Method-specific Results")
        
        for method in methods:
            method_results = self.results['methods'][method]
            seeds = list(method_results.keys())
            method_success = sum(1 for seed, results in method_results.items() 
                               if results.get('experiment_results', {}).get('status') == 'success')
            
            report_lines.append(f"### {method.upper()}")
            report_lines.append(f"- Seeds tested: {', '.join(seeds)}")
            report_lines.append(f"- Successful runs: {method_success}/{len(seeds)}")
            report_lines.append(f"- Success rate: {method_success/len(seeds):.2%}")
            
            # Analysis results
            analysis_count = 0
            for seed, seed_results in method_results.items():
                if 'task_analyses' in seed_results and seed_results['task_analyses']:
                    analysis_count += 1
            
            report_lines.append(f"- Analyses completed: {analysis_count}")
            report_lines.append("")
        
        # Key findings
        report_lines.append("## Key Findings")
        report_lines.append("- Custom CIFAR-10 dataset with airplane/automobile vs bird/truck tasks")
        report_lines.append("- Vision Transformer backbone used for attention analysis")
        report_lines.append("- Attention maps and network activations extracted for analysis")
        report_lines.append("- Comparison between DER++ (memory-based) and EWC (regularization-based) methods")
        report_lines.append("")
        
        # Files generated
        report_lines.append("## Generated Files")
        analysis_files = [f for f in os.listdir(self.analysis_dir) if f.endswith(('.png', '.csv'))]
        for file in sorted(analysis_files):
            report_lines.append(f"- {file}")
        
        # Save report
        report_file = os.path.join(self.analysis_dir, 'summary_report.md')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Summary report saved to {report_file}")
    
    def create_visualization_dashboard(self) -> None:
        """
        Create a comprehensive visualization dashboard.
        """
        print("Creating visualization dashboard...")
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Subplot 1: Method comparison
        ax1 = plt.subplot(2, 2, 1)
        methods = list(self.results['methods'].keys())
        method_counts = [len(self.results['methods'][method]) for method in methods]
        
        plt.bar(methods, method_counts, color=['skyblue', 'lightcoral'])
        plt.title('Number of Experiments per Method')
        plt.ylabel('Number of Runs')
        
        # Subplot 2: Success rate
        ax2 = plt.subplot(2, 2, 2)
        success_rates = []
        for method in methods:
            method_results = self.results['methods'][method]
            success_count = sum(1 for seed, results in method_results.items() 
                              if results.get('experiment_results', {}).get('status') == 'success')
            success_rates.append(success_count / len(method_results))
        
        plt.bar(methods, success_rates, color=['lightgreen', 'orange'])
        plt.title('Success Rate by Method')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        
        # Subplot 3: Analysis completion
        ax3 = plt.subplot(2, 2, 3)
        analysis_counts = []
        for method in methods:
            method_results = self.results['methods'][method]
            analysis_count = sum(1 for seed, results in method_results.items() 
                               if 'task_analyses' in results and results['task_analyses'])
            analysis_counts.append(analysis_count)
        
        plt.bar(methods, analysis_counts, color=['gold', 'mediumpurple'])
        plt.title('Analyses Completed by Method')
        plt.ylabel('Number of Analyses')
        
        # Subplot 4: Experiment timeline (placeholder)
        ax4 = plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, f"Experiment: {self.results['experiment_name']}\n"
                           f"Timestamp: {self.results['timestamp']}\n"
                           f"Total Methods: {len(methods)}\n"
                           f"Total Runs: {sum(method_counts)}",
                 ha='center', va='center', transform=ax4.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.title('Experiment Summary')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'experiment_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dashboard saved to {self.analysis_dir}/experiment_dashboard.png")
    
    def run_full_analysis(self) -> None:
        """
        Run the complete analysis pipeline.
        """
        print(f"Starting analysis of results in {self.results_dir}")
        
        self.compare_attention_patterns()
        self.analyze_method_performance()
        self.create_visualization_dashboard()
        self.generate_summary_report()
        
        print(f"Analysis complete! Results saved in {self.analysis_dir}")


def main():
    """Main function for running the analysis."""
    parser = argparse.ArgumentParser(description='Analyze shortcut investigation experiment results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    analyzer = ResultsAnalyzer(args.results_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
