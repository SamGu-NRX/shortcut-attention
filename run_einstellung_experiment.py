#!/usr/bin/env python3
# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Enhanced Einstellung Effect Experiment Runner with Checkpoint Management

This script provides comprehensive checkpoint management for Einstellung Effect experiments:

1. Automatic checkpoint discovery and reuse
2. Skip-training mode for evaluation-only runs
3. Interactive prompts for checkpoint handling
4. Integration with Mammoth's native checkpointing system
5. Basic usage with DER++ and EWC strategies
6. Configuration of Einstellung parameters
7. Results collection and analysis
8. Attention analysis for ViT models

Usage:
    # Basic usage (auto-discover checkpoints)
    python run_einstellung_experiment.py --model derpp --backbone resnet18

    # Skip training, only evaluate existing checkpoints
    python run_einstellung_experiment.py --model derpp --backbone resnet18 --skip_training

    # Force retraining even if checkpoints exist
    python run_einstellung_experiment.py --model derpp --backbone resnet18 --force_retrain

    # Comparative analysis with checkpoint reuse
    python run_einstellung_experiment.py --comparative --auto_checkpoint
"""

import argparse
import glob
import logging
import os
import re
import subprocess
import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from experiments.einstellung import (
    ComparativeExperimentPlan,
    ExperimentConfig,
    ExecutionMode,
    EinstellungRunner,
    run_comparative_suite,
)
from experiments.einstellung.args_builder import build_mammoth_args, determine_dataset
from experiments.einstellung.reporting import write_single_run_report

LOGGER = logging.getLogger("einstellung.cli")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Einstellung Effect experiments")

    parser.add_argument("--comparative", action="store_true", help="Run comparative suite")
    parser.add_argument("--model", default="derpp", choices=["sgd", "derpp", "ewc_on", "gpm", "dgr", "scratch_t2", "interleaved"], help="Strategy to run")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "vit"], help="Backbone architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, help="Override training epochs")

    parser.add_argument("--skip_training", action="store_true", help="Skip training and evaluate existing checkpoint")
    parser.add_argument("--force_retrain", action="store_true", help="Always retrain even if checkpoints exist")
    parser.add_argument("--auto_checkpoint", action="store_true", help="Use checkpoint automatically when available")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode (short runs)")
    parser.add_argument("--code_optimization", type=int, default=1, choices=[0, 1, 2, 3], help="CUDA optimisation level")
    parser.add_argument("--disable_cache", action="store_true", help="Disable Einstellung dataset caching")
    parser.add_argument("--enable_cache", action="store_true", default=True, help="Enable Einstellung dataset caching")
    parser.add_argument("--results_root", default="einstellung_results", help="Root directory for outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args(argv)


def execution_mode_from_args(args: argparse.Namespace) -> ExecutionMode:
    return ExecutionMode.from_flags(
        skip_training=args.skip_training,
        force_retrain=args.force_retrain,
        auto_checkpoint=args.auto_checkpoint,
    )


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        strategy=args.model,
        backbone=args.backbone,
        seed=args.seed,
        epochs=args.epochs,
        debug=args.debug,
        enable_cache=args.enable_cache and not args.disable_cache,
        code_optimization=args.code_optimization,
        execution_mode=execution_mode_from_args(args),
        results_root=Path(args.results_root),
    )


def run_single(args: argparse.Namespace) -> Dict[str, any]:
    config = build_config(args)
    runner = EinstellungRunner(project_root=Path.cwd())
    result = runner.run(config)

    if result.get("success"):
        summary_df = pd.read_csv(result["summary_path"]) if result.get("summary_path") else pd.DataFrame()
        report_dir = Path(result["results_dir"]) / "reports"
        write_single_run_report(result=result, summary_df=summary_df, output_dir=report_dir)

    return result


def run_comparative(args: argparse.Namespace) -> List[Dict[str, any]]:
    config = build_config(args)
    runner = EinstellungRunner(project_root=Path.cwd())

    plan = ComparativeExperimentPlan(
        baselines=["scratch_t2", "interleaved"],
        continual_methods=["sgd", "derpp", "ewc_on", "gpm", "dgr"],
        backbone=args.backbone,
        seed=args.seed,
        epochs=args.epochs,
    )

    output_root = Path("comparative_results")
    results, report_path = run_comparative_suite(runner, config, plan, output_root)

    LOGGER.info("Comparative report: %s", report_path)
    return results


def run_einstellung_experiment(
    strategy: str = "derpp",
    backbone: str = "resnet18",
    seed: int = 42,
    skip_training: bool = False,
    force_retrain: bool = False,
    auto_checkpoint: bool = True,
    debug: bool = False,
    enable_cache: bool = True,
    code_optimization: int = 1,
    epochs: Optional[int] = None,
) -> Dict[str, any]:
    args = argparse.Namespace(
        comparative=False,
        model=strategy,
        backbone=backbone,
        seed=seed,
        epochs=epochs,
        skip_training=skip_training,
        force_retrain=force_retrain,
        auto_checkpoint=auto_checkpoint,
        debug=debug,
        enable_cache=enable_cache,
        disable_cache=not enable_cache,
        code_optimization=code_optimization,
        results_root="einstellung_results",
        verbose=False,
    )
    return run_single(args)


def run_comparative_experiment(
    skip_training: bool = False,
    force_retrain: bool = False,
    auto_checkpoint: bool = True,
    debug: bool = False,
    enable_cache: bool = True,
    code_optimization: int = 1,
    epochs: Optional[int] = None,
) -> List[Dict[str, any]]:
    args = argparse.Namespace(
        comparative=True,
        model="derpp",
        backbone="resnet18",
        seed=42,
        epochs=epochs,
        skip_training=skip_training,
        force_retrain=force_retrain,
        auto_checkpoint=auto_checkpoint,
        debug=debug,
        enable_cache=enable_cache,
        disable_cache=not enable_cache,
        code_optimization=code_optimization,
        results_root="einstellung_results",
        verbose=False,
    )
    return run_comparative(args)


def create_einstellung_args(strategy: str, backbone: str, seed: int, debug: bool = False, epochs: Optional[int] = None) -> List[str]:
    """Legacy helper retained for tests – returns CLI args for main.py."""
    config = ExperimentConfig(
        strategy=strategy,
        backbone=backbone,
        seed=seed,
        debug=debug,
        epochs=epochs,
    )
    return build_mammoth_args(config, results_path=Path("/tmp"), evaluation_only=False, checkpoint_path=None)


def extract_accuracy_from_output(output: str) -> Optional[float]:
    pattern = r"Accuracy for \d+ task\(s\):\s*\[Class-IL\]:\s*([\d.]+)\s*%"
    match = re.search(pattern, output or "")
    return float(match.group(1)) if match else None


def find_csv_file(output_dir: str) -> Optional[str]:
    root = Path(output_dir)
    if not root.exists():
        return None
    for candidate in [root / "timeline.csv", root / "eri_sc_metrics.csv", root / "summary.csv"]:
        if candidate.exists():
            return str(candidate)
    csv_files = sorted(root.glob("**/*.csv"))
    return str(csv_files[0]) if csv_files else None


def aggregate_comparative_results(results_list: List[Dict[str, any]], output_dir: str) -> str:
    frames: List[pd.DataFrame] = []

    for result in results_list:
        if not result or not result.get('success', False):
            continue

        method = result.get('strategy', 'unknown')
        backbone = result.get('backbone', 'unknown')
        seed = result.get('seed', 42)

        # Create method-specific directory
        method_dir = os.path.join(individual_dir, f"{method}_{backbone}_seed{seed}")
        os.makedirs(method_dir, exist_ok=True)

        # Copy/link original results
        original_output_dir = result.get('output_dir')
        if original_output_dir and os.path.exists(original_output_dir):
            # Copy key files to organized structure
            files_to_copy = [
                'eri_sc_metrics.csv',
                'eri_dynamics.pdf',
                'terminal_log.txt'
            ]

            for filename in files_to_copy:
                src_path = os.path.join(original_output_dir, filename)
                if os.path.exists(src_path):
                    dst_path = os.path.join(method_dir, filename)
                    import shutil
                    shutil.copy2(src_path, dst_path)

            # Copy reports directory if it exists
            reports_src = os.path.join(original_output_dir, 'reports')
            if os.path.exists(reports_src):
                reports_dst = os.path.join(method_dir, 'reports')
                import shutil
                shutil.copytree(reports_src, reports_dst, dirs_exist_ok=True)

        organized_results[method] = method_dir

    return organized_results


def generate_master_comparative_report(results_list: List[Dict],
                                     comparative_metrics: Dict[str, Dict],
                                     statistical_results: Dict[str, Any],
                                     output_structure: Dict[str, str]) -> str:
    """
    Generate master comparative report combining individual and cross-method analysis.

    Args:
        results_list: List of experiment result dictionaries
        comparative_metrics: Comparative metrics from compute_comparative_metrics_from_aggregated_data
        statistical_results: Statistical analysis results
        output_structure: Directory structure from create_enhanced_output_structure

    Returns:
        Path to generated master report
    """
    reports_dir = output_structure['reports']
    report_path = os.path.join(reports_dir, "master_comparative_report.html")

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Organize results by type
    baseline_methods = []
    cl_methods = []

    for result in results_list:
        if result and result.get('success', False):
            method = result.get('strategy', 'unknown')
            if method in ['scratch_t2', 'interleaved']:
                baseline_methods.append(result)
            else:
                cl_methods.append(result)

    # Generate HTML report
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Master Comparative Einstellung Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            line-height: 1.6;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .section {{
            background: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .method-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .method-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .baseline-card {{ border-left: 4px solid #28a745; }}
        .cl-card {{ border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .toc {{
            background: #e9ecef;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .toc ul {{ list-style-type: none; padding-left: 0; }}
        .toc li {{ margin: 5px 0; }}
        .toc a {{ text-decoration: none; color: #007bff; }}
        .toc a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Master Comparative Einstellung Analysis Report</h1>
        <h2>Comprehensive Cross-Method Cognitive Rigidity Assessment</h2>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Total Methods:</strong> {len([r for r in results_list if r and r.get('success', False)])}
           | <strong>Baselines:</strong> {len(baseline_methods)}
           | <strong>Continual Learning:</strong> {len(cl_methods)}</p>
    </div>

    <div class="toc">
        <h3>📋 Table of Contents</h3>
        <ul>
            <li><a href="#executive-summary">Executive Summary</a></li>
            <li><a href="#baseline-methods">Baseline Methods Analysis</a></li>
            <li><a href="#continual-learning-methods">Continual Learning Methods Analysis</a></li>
            <li><a href="#comparative-metrics">Comparative Metrics Analysis</a></li>
            <li><a href="#statistical-significance">Statistical Significance Testing</a></li>
            <li><a href="#method-rankings">Method Rankings and Recommendations</a></li>
            <li><a href="#experimental-metadata">Experimental Metadata</a></li>
        </ul>
    </div>

    <div class="section" id="executive-summary">
        <h2>📊 Executive Summary</h2>
        <p>This report presents a comprehensive comparative analysis of {len(cl_methods)} continual learning methods
           against {len(baseline_methods)} baseline methods for Einstellung Effect assessment.</p>

        <div class="method-grid">"""

    # Add method summary cards
    for result in baseline_methods + cl_methods:
        method = result.get('strategy', 'unknown')
        backbone = result.get('backbone', 'unknown')
        final_acc = result.get('final_accuracy', 0)
        card_class = "baseline-card" if method in ['scratch_t2', 'interleaved'] else "cl-card"
        method_type = "Baseline" if method in ['scratch_t2', 'interleaved'] else "Continual Learning"

        # Get comparative metrics if available
        method_metrics = comparative_metrics.get(method, {})
        pd_t = method_metrics.get('pd_t_final', None)
        sfr_rel = method_metrics.get('sfr_rel_final', None)
        ad = method_metrics.get('adaptation_delay', None)

        html_content += f"""
            <div class="method-card {card_class}">
                <h4>{method.upper()} ({backbone})</h4>
                <p><strong>Type:</strong> {method_type}</p>
                <p><strong>Final Accuracy:</strong> <span class="metric-value">{final_acc:.2f}%</span></p>"""

        if pd_t is not None:
            html_content += f'<p><strong>Performance Deficit:</strong> {pd_t:.3f}</p>'
        if sfr_rel is not None:
            html_content += f'<p><strong>SFR Relative:</strong> {sfr_rel:.3f}</p>'
        if ad is not None:
            html_content += f'<p><strong>Adaptation Delay:</strong> {ad:.1f} epochs</p>'

        html_content += "</div>"

    html_content += """
        </div>
    </div>"""

    # Baseline Methods Analysis
    html_content += f"""
    <div class="section" id="baseline-methods">
        <h2>🎯 Baseline Methods Analysis</h2>
        <p>Baseline methods provide reference points for measuring continual learning effectiveness:</p>
        <table>
            <tr><th>Method</th><th>Description</th><th>Final Accuracy</th><th>Purpose</th></tr>"""

    for result in baseline_methods:
        method = result.get('strategy', 'unknown')
        final_acc = result.get('final_accuracy', 0)

        if method == 'scratch_t2':
            description = "Task 2 only training"
            purpose = "Optimal performance reference (no catastrophic forgetting)"
        elif method == 'interleaved':
            description = "Mixed Task 1 + Task 2 training"
            purpose = "Joint training reference (no task boundaries)"
        else:
            description = "Unknown baseline"
            purpose = "Reference method"

        html_content += f"""
            <tr>
                <td><strong>{method.upper()}</strong></td>
                <td>{description}</td>
                <td class="success">{final_acc:.2f}%</td>
                <td>{purpose}</td>
            </tr>"""

    html_content += """
        </table>
    </div>"""

    # Continual Learning Methods Analysis
    html_content += f"""
    <div class="section" id="continual-learning-methods">
        <h2>🔄 Continual Learning Methods Analysis</h2>
        <p>Continual learning methods tested for cognitive rigidity and adaptation capabilities:</p>
        <table>
            <tr><th>Method</th><th>Final Accuracy</th><th>PD_t</th><th>SFR_rel</th><th>AD</th><th>Assessment</th></tr>"""

    for result in cl_methods:
        method = result.get('strategy', 'unknown')
        final_acc = result.get('final_accuracy', 0)

        method_metrics = comparative_metrics.get(method, {})
        pd_t = method_metrics.get('pd_t_final', None)
        sfr_rel = method_metrics.get('sfr_rel_final', None)
        ad = method_metrics.get('adaptation_delay', None)

        # Determine assessment based on metrics
        if pd_t is not None and sfr_rel is not None:
            if pd_t < 0.1 and sfr_rel < 0.1:
                assessment = "Excellent adaptation"
                assessment_class = "success"
            elif pd_t < 0.3 and sfr_rel < 0.3:
                assessment = "Good adaptation"
                assessment_class = "success"
            elif pd_t < 0.5 and sfr_rel < 0.5:
                assessment = "Moderate rigidity"
                assessment_class = "warning"
            else:
                assessment = "High rigidity"
                assessment_class = "danger"
        else:
            assessment = "Insufficient data"
            assessment_class = ""

        pd_str = f"{pd_t:.3f}" if pd_t is not None else "N/A"
        sfr_str = f"{sfr_rel:.3f}" if sfr_rel is not None else "N/A"
        ad_str = f"{ad:.1f}" if ad is not None else "N/A"

        html_content += f"""
            <tr>
                <td><strong>{method.upper()}</strong></td>
                <td>{final_acc:.2f}%</td>
                <td>{pd_str}</td>
                <td>{sfr_str}</td>
                <td>{ad_str}</td>
                <td class="{assessment_class}">{assessment}</td>
            </tr>"""

    html_content += """
        </table>
    </div>"""

    # Statistical Significance Section
    if statistical_results:
        html_content += f"""
        <div class="section" id="statistical-significance">
            <h2>📈 Statistical Significance Testing</h2>
            <p>Statistical analysis of performance differences between methods:</p>"""

        if 'interpretation' in statistical_results:
            interpretation = statistical_results['interpretation']
            html_content += f"""
            <h4>Key Findings:</h4>
            <ul>"""

            if 'significant_differences' in interpretation:
                html_content += f"<li><strong>Significant Differences:</strong> {interpretation['significant_differences']}</li>"
            if 'large_effects' in interpretation:
                html_content += f"<li><strong>Effect Sizes:</strong> {interpretation['large_effects']}</li>"
            if 'best_method' in interpretation:
                html_content += f"<li><strong>Best Performing Method:</strong> {interpretation['best_method']}</li>"

            html_content += """
            </ul>"""

        html_content += """
        </div>"""

    # Experimental Metadata
    html_content += f"""
    <div class="section" id="experimental-metadata">
        <h2>🔬 Experimental Metadata</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Dataset</td><td>CIFAR-100 Einstellung</td></tr>
            <tr><td>Task Structure</td><td>2 tasks (T1: base classes, T2: shortcut classes)</td></tr>
            <tr><td>Evaluation Protocol</td><td>T1_all, T2_shortcut_normal, T2_shortcut_masked, T2_nonshortcut_normal</td></tr>
            <tr><td>Random Seed</td><td>42</td></tr>
            <tr><td>Backbone Architecture</td><td>ResNet-18</td></tr>
            <tr><td>Total Experiments</td><td>{len([r for r in results_list if r and r.get('success', False)])}</td></tr>
            <tr><td>Report Generated</td><td>{timestamp}</td></tr>
        </table>

        <h4>Method Configurations:</h4>
        <ul>"""

    for result in results_list:
        if result and result.get('success', False):
            method = result.get('strategy', 'unknown')
            backbone = result.get('backbone', 'unknown')
            used_checkpoint = result.get('used_checkpoint', False)
            source = "Checkpoint" if used_checkpoint else "Training"
            html_content += f"<li><strong>{method.upper()}</strong> with {backbone} (Source: {source})</li>"

    html_content += """
        </ul>
    </div>

    <hr>
    <p><em>Master Comparative Report generated by Mammoth Einstellung Experiment Runner</em></p>
</body>
</html>"""

    # Write report
    with open(report_path, 'w') as f:
        f.write(html_content)

    return report_path


def create_publication_ready_outputs(output_structure: Dict[str, str],
                                   comparative_visualizations_dir: str) -> Dict[str, str]:
    """
    Create publication-ready figures and organize them in publication_ready directory.

    Args:
        output_structure: Directory structure from create_enhanced_output_structure
        comparative_visualizations_dir: Directory containing comparative visualizations

    Returns:
        Dictionary mapping output types to their publication-ready paths
    """
    pub_dir = output_structure['publication_ready']
    publication_outputs = {}

    # Copy and rename visualizations for publication
    visualization_mappings = {
        'eri_dynamics.pdf': 'figure_1_comparative_dynamics.pdf',
        'eri_heatmap.pdf': 'figure_2_adaptation_heatmap.pdf',
        'eri_dynamics.png': 'figure_1_comparative_dynamics.png',
        'eri_heatmap.png': 'figure_2_adaptation_heatmap.png'
    }

    for original_name, pub_name in visualization_mappings.items():
        src_path = os.path.join(comparative_visualizations_dir, original_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(pub_dir, pub_name)
            import shutil
            shutil.copy2(src_path, dst_path)
            publication_outputs[pub_name] = dst_path

    # Create publication metadata file
    metadata_path = os.path.join(pub_dir, 'publication_metadata.json')
    metadata = {
        'title': 'Comparative Einstellung Effect Analysis in Continual Learning',
        'figures': {
            'figure_1': {
                'title': 'Comparative Learning Dynamics Across Methods',
                'description': 'Timeline showing accuracy evolution and performance deficits',
                'files': ['figure_1_comparative_dynamics.pdf', 'figure_1_comparative_dynamics.png']
            },
            'figure_2': {
                'title': 'Adaptation Delay Sensitivity Analysis',
                'description': 'Heatmap showing robustness across threshold values',
                'files': ['figure_2_adaptation_heatmap.pdf', 'figure_2_adaptation_heatmap.png']
            }
        },
        'generated': datetime.now().isoformat()
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    publication_outputs['metadata'] = metadata_path

    return publication_outputs


def generate_experiment_metadata(results_list: List[Dict],
                               comparative_metrics: Dict[str, Dict],
                               statistical_results: Dict[str, Any],
                               output_structure: Dict[str, str]) -> str:
    """
    Generate comprehensive experimental metadata and configuration summary.

    Args:
        results_list: List of experiment result dictionaries
        comparative_metrics: Comparative metrics from compute_comparative_metrics_from_aggregated_data
        statistical_results: Statistical analysis results
        output_structure: Directory structure from create_enhanced_output_structure

    Returns:
        Path to generated metadata file
    """
    metadata_dir = output_structure['metadata']
    metadata_path = os.path.join(metadata_dir, 'experiment_metadata.json')

    # Collect experiment metadata
    metadata = {
        'experiment_info': {
            'title': 'Comparative Einstellung Effect Analysis',
            'description': 'Cross-method assessment of cognitive rigidity in continual learning',
            'generated': datetime.now().isoformat(),
            'total_methods': len([r for r in results_list if r and r.get('success', False)]),
            'baseline_methods': len([r for r in results_list if r and r.get('success', False) and r.get('strategy') in ['scratch_t2', 'interleaved']]),
            'continual_learning_methods': len([r for r in results_list if r and r.get('success', False) and r.get('strategy') not in ['scratch_t2', 'interleaved']])
        },
        'dataset_config': {
            'name': 'CIFAR-100 Einstellung',
            'task_structure': '2 tasks (T1: base classes, T2: shortcut classes)',
            'evaluation_protocol': ['T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal'],
            'shortcut_type': 'Visual shortcuts in Task 2 classes'
        },
        'method_configurations': {},
        'comparative_metrics_summary': {},
        'statistical_analysis_summary': statistical_results,
        'output_structure': output_structure
    }

    # Add method configurations
    for result in results_list:
        if result and result.get('success', False):
            method = result.get('strategy', 'unknown')
            metadata['method_configurations'][method] = {
                'backbone': result.get('backbone', 'unknown'),
                'seed': result.get('seed', 42),
                'final_accuracy': result.get('final_accuracy', 0),
                'used_checkpoint': result.get('used_checkpoint', False),
                'training_time': result.get('training_time', 0),
                'evaluation_time': result.get('evaluation_time', 0),
                'output_directory': result.get('output_dir', 'unknown')
            }

    # Add comparative metrics summary
    for method, metrics in comparative_metrics.items():
        metadata['comparative_metrics_summary'][method] = {
            'performance_deficit_final': metrics.get('pd_t_final', None),
            'shortcut_forgetting_rate_relative_final': metrics.get('sfr_rel_final', None),
            'adaptation_delay': metrics.get('adaptation_delay', None)
        }

    # Write metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def aggregate_comparative_results(results_list: List[Dict], output_dir: str) -> str:
    """
    Aggregate CSV results from multiple experiments into single comparative dataset.

    Args:
        results_list: List of experiment result dictionaries
        output_dir: Directory to save aggregated results

    Returns:
        Path to aggregated CSV file

    Raises:
        ValueError: If no valid CSV files found
        IOError: If aggregation fails
    """
    try:
        from eri_vis.data_loader import ERIDataLoader
        import pandas as pd
    except ImportError as e:
        raise ImportError(f"ERI visualization components not available: {e}")

    # Collect all CSV files from individual experiments
    csv_files = []
    for result in results_list:
        if result and result.get('success', False):
            csv_path = find_csv_file(result['output_dir'])
            if csv_path:
                csv_files.append(csv_path)

    if not csv_files:
        raise ValueError("No valid CSV files found in experiment results")

    print(f"🔄 Aggregating {len(csv_files)} CSV files for comparative analysis...")

    # Use existing ERIDataLoader to load and validate datasets
    loader = ERIDataLoader()
    datasets = []

    for csv_file in csv_files:
        try:
            dataset = loader.load_csv(csv_file)
            datasets.append(dataset)
            print(f"   ✅ Loaded {len(dataset)} rows from {os.path.basename(csv_file)}")
        except Exception as e:
            print(f"   ⚠️  Failed to load {os.path.basename(csv_file)}: {e}")
            continue

    if not datasets:
        raise ValueError("No datasets could be loaded successfully")

    # Merge all datasets into single DataFrame
    merged_data = pd.concat([dataset.data for dataset in datasets], ignore_index=True)

    # Remove duplicates (in case same experiment was run multiple times)
    merged_data = merged_data.drop_duplicates(
        subset=['method', 'seed', 'epoch_eff', 'split'],
        keep='last'
    ).reset_index(drop=True)

    # Validate baseline methods and provide warnings
    validation_result = validate_baseline_methods(merged_data)

    # Print validation warnings using existing logging system
    for warning in validation_result['warnings']:
        print(warning)

    # Log available methods and baseline status
    print(f"📊 Available methods: {validation_result['available_methods']}")
    if validation_result['missing_baselines']:
        print(f"❌ Missing baseline methods: {validation_result['missing_baselines']}")
        print(f"📈 Comparative metrics available: PD_t={validation_result['can_compute_pd_t']}, SFR_rel={validation_result['can_compute_sfr_rel']}")
    else:
        print(f"✅ All baseline methods present - full comparative analysis available")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export aggregated CSV
    aggregated_csv_path = output_path / "comparative_eri_metrics.csv"
    merged_data.to_csv(aggregated_csv_path, index=False, float_format='%.6f')

    # Save validation results for later use
    validation_path = output_path / "baseline_validation.json"
    import json
    with open(validation_path, 'w') as f:
        json.dump(validation_result, f, indent=2)

    print(f"📊 Aggregated {len(merged_data)} rows from {len(datasets)} experiments")
    print(f"   Methods: {sorted(merged_data['method'].unique())}")
    print(f"   Seeds: {sorted(merged_data['seed'].unique())}")
    print(f"   Splits: {sorted(merged_data['split'].unique())}")
    print(f"💾 Saved aggregated results to: {aggregated_csv_path}")
    print(f"💾 Saved baseline validation to: {validation_path}")

    return str(aggregated_csv_path)


def find_csv_file(output_dir: str) -> Optional[str]:
    """
    Find and validate ERI CSV file in experiment output directory.

    Uses existing file path patterns and naming conventions from current experiment system.
    Implements validation to ensure CSV files contain required columns and valid data.

    Args:
        output_dir: Directory to search for CSV files

    Returns:
        Path to valid CSV file if found, None otherwise
    """
    if not output_dir or not os.path.exists(output_dir):
        return None

    # Look for ERI-specific CSV files first (following existing naming conventions)
    eri_patterns = [
        "**/eri_sc_metrics.csv",
        "**/eri_metrics.csv",
        "**/einstellung_metrics.csv",
        "**/comparative_eri_metrics.csv"  # For aggregated results
    ]

    for pattern in eri_patterns:
        csv_files = list(Path(output_dir).glob(pattern))
        if csv_files:
            # Validate and return the most recent valid file
            for csv_file in sorted(csv_files, key=lambda p: p.stat().st_mtime, reverse=True):
                if validate_csv_file(str(csv_file)):
                    return str(csv_file)

    # Fallback to any CSV file with validation
    csv_files = list(Path(output_dir).glob("**/*.csv"))
    if csv_files:
        # Filter out obviously unrelated files
        relevant_files = [
            f for f in csv_files
            if not any(exclude in f.name.lower()
                      for exclude in ['log', 'debug', 'temp', 'backup', 'checkpoint'])
        ]

        # Validate files and return first valid one
        for csv_file in sorted(relevant_files, key=lambda p: p.stat().st_mtime, reverse=True):
            if validate_csv_file(str(csv_file)):
                return str(csv_file)

    return None


def validate_csv_file(csv_path: str) -> bool:
    """
    Validate CSV file contains required columns and valid ERI data.

    Args:
        csv_path: Path to CSV file to validate

    Returns:
        True if file is valid ERI CSV, False otherwise
    """
    try:
        import pandas as pd

        # Check file exists and is readable
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return False

        # Try to load CSV
        df = pd.read_csv(csv_path)

        # Check required columns for ERI format
        required_cols = ["method", "seed", "epoch_eff", "split", "acc"]
        if not all(col in df.columns for col in required_cols):
            return False

        # Check for minimum data
        if len(df) == 0:
            return False

        # Check valid split names
        valid_splits = {"T1_all", "T2_shortcut_normal", "T2_shortcut_masked", "T2_nonshortcut_normal"}
        if not any(split in valid_splits for split in df['split'].unique()):
            return False

        # Check accuracy values are in valid range
        if df['acc'].min() < 0 or df['acc'].max() > 1:
            return False

        return True

    except Exception:
        # Any error during validation means invalid file
        return False


def generate_synthetic_eri_data(csv_path: Path, experiment_results: Dict[str, Any]) -> None:
    """Generate synthetic ERI data for demonstration purposes."""
    import pandas as pd
    import numpy as np

    method = experiment_results.get('model', 'sgd')
    seed = experiment_results.get('seed', 42)
    final_acc = experiment_results.get('final_accuracy', 50.0) / 100.0  # Convert to 0-1 range

    # Generate synthetic timeline data
    epochs = np.linspace(0.1, 2.0, 20)  # Effective epochs

    # Synthetic accuracy curves with realistic patterns
    base_acc = final_acc * 0.8  # Start lower

    data_rows = []
    for epoch in epochs:
        # T1_all: Slight decline due to forgetting
        t1_acc = max(0.1, base_acc * (1.0 - 0.1 * epoch))

        # T2_shortcut_normal: Improves with training
        t2_sc_normal = min(0.9, base_acc + 0.3 * (1 - np.exp(-2 * epoch)))

        # T2_shortcut_masked: Lower performance, slower improvement
        t2_sc_masked = min(0.8, base_acc + 0.2 * (1 - np.exp(-1.5 * epoch)))

        # T2_nonshortcut_normal: Baseline performance
        t2_nonsc = min(0.85, base_acc + 0.25 * (1 - np.exp(-1.8 * epoch)))

        # Add some noise
        noise = np.random.normal(0, 0.02)

        data_rows.extend([
            [method, seed, epoch, "T1_all", max(0, t1_acc + noise)],
            [method, seed, epoch, "T2_shortcut_normal", max(0, t2_sc_normal + noise)],
            [method, seed, epoch, "T2_shortcut_masked", max(0, t2_sc_masked + noise)],
            [method, seed, epoch, "T2_nonshortcut_normal", max(0, t2_nonsc + noise)]
        ])

    # Create DataFrame and save
    df = pd.DataFrame(data_rows, columns=["method", "seed", "epoch_eff", "split", "acc"])
    df.to_csv(csv_path, index=False)
    print(f"Generated synthetic ERI data: {csv_path}")


def run_experiment(model: str, backbone: str, seed: int,
                  results_path: str, execution_mode: str = "interactive",
                  **kwargs) -> Dict[str, Any]:
    """
    Enhanced experiment runner with comprehensive reporting and Einstellung integration
    """
    # Initialize logging and reporting
    logger = TerminalLogger(results_path)
    reporter = ResultsReporter(results_path)

    start_time = time.time()
    training_time = 0
    evaluation_time = 0

    logger.log("=" * 60)
    logger.log("Running Einstellung Effect Experiment")
    logger.log(f"Strategy: {model}")
    logger.log(f"Backbone: {backbone}")
    logger.log(f"Seed: {seed}")
    logger.log(f"Mode: {execution_mode}")
    logger.log("=" * 60)

    # DEBUG: Log experiment configuration details
    logger.log("🔍 EXPERIMENT CONFIGURATION DEBUG:")
    logger.log(f"   - Model: {model}")
    logger.log(f"   - Backbone: {backbone}")
    logger.log(f"   - Execution mode: {execution_mode}")
    logger.log(f"   - Kwargs: {kwargs}")

    # Find existing checkpoints
    checkpoints = find_existing_checkpoints(model, backbone, seed)
    logger.log(f"   - Found {len(checkpoints)} existing checkpoints")

    used_checkpoint = False
    checkpoint_path = None
    evaluation_only = False

    # Handle checkpoint discovery and execution mode
    if checkpoints and execution_mode != "force_retrain":
        if execution_mode in ["auto_checkpoint", "skip_training"]:
            used_checkpoint = True
            checkpoint_path = checkpoints[-1]
            evaluation_only = execution_mode == "skip_training"
            logger.log(f"✓ Using existing checkpoint: {checkpoint_path}")
            if evaluation_only:
                logger.log("⚠️  EVALUATION-ONLY mode detected")
                logger.log("   This will limit available Einstellung metrics")
                logger.log("   For comprehensive analysis, use --force_retrain")
        elif execution_mode == "interactive":
            # Simplified interactive prompt for now
            logger.log("📁 Found existing checkpoints:")
            for i, ckpt in enumerate(checkpoints[:3]):  # Show first 3
                logger.log(f"   {i+1}. {os.path.basename(ckpt)}")
            logger.log("\n🤔 Recommendation: Use --force_retrain for full Einstellung analysis")
            logger.log("   Checkpoint mode provides limited metrics")

            # For automated testing, default to retraining for full analysis
            logger.log("✓ Defaulting to retrain for comprehensive Einstellung metrics")
            used_checkpoint = False
    else:
        logger.log("✓ No checkpoints found or forced retraining")

    # CRITICAL: Override for Einstellung experiments - always prefer full training
    if not used_checkpoint or execution_mode == "force_retrain":
        logger.log("🧠 EINSTELLUNG COMPREHENSIVE MODE:")
        logger.log("   - Full training will be performed")
        logger.log("   - Complete ERI metrics will be available")

        # Check if attention analysis is supported for this backbone
        backbone_type = backbone.lower()
        if 'vit' in backbone_type or 'vision' in backbone_type or 'transformer' in backbone_type:
            logger.log("   - Attention analysis will be conducted (ViT model)")
        else:
            logger.log("   - Attention analysis not available (CNN backbone)")
            logger.log("     Use --backbone vit for attention analysis")

        used_checkpoint = False
        evaluation_only = False
    elif used_checkpoint:
        logger.log("⚠️  EINSTELLUNG LIMITED MODE:")
        logger.log("   - Using existing checkpoint")
        logger.log("   - Limited Einstellung metrics available")
        logger.log("   - No attention analysis")
        logger.log("   - Consider --force_retrain for full analysis")

    # Dataset selection - CRITICAL FOR INTEGRATION
    def determine_dataset_name(backbone: str) -> str:
        """Determine the appropriate Einstellung dataset name based on backbone."""
        if backbone.lower() == 'vit' or 'vit' in backbone.lower():
            return 'seq-cifar100-einstellung-224'
        else:
            return 'seq-cifar100-einstellung'

    dataset_name = determine_dataset_name(backbone)
    logger.log("🧠 DATASET SELECTION:")
    logger.log(f"   - Backbone: {backbone}")
    logger.log(f"   - Selected dataset: {dataset_name}")
    logger.log(f"   - Contains 'einstellung': {'einstellung' in dataset_name.lower()}")

    if 'einstellung' not in dataset_name.lower():
        logger.log("   ❌ WARNING: Dataset name does not contain 'einstellung'")
        logger.log("   This will prevent Einstellung integration from activating!")

    # Honour overrides for core training hyper-parameters while keeping sensible defaults
    debug = kwargs.get('debug', False)
    base_debug_epochs = 5 if 'vit' in backbone.lower() else 10
    base_full_epochs = 20 if 'vit' in backbone.lower() else 50

    default_epochs = base_debug_epochs if debug else base_full_epochs

    if model == 'interleaved':
        default_epochs *= 2
        base_debug_epochs *= 2
        base_full_epochs *= 2

    if debug:
        logger.log("🐛 DEBUG MODE: Using shorter training epochs for faster testing")
        logger.log(f"   - Debug epochs: {default_epochs} (full training baseline: {base_full_epochs})")

    n_epochs = kwargs.get('epochs')
    if n_epochs is None:
        n_epochs = kwargs.get('n_epochs')
    if n_epochs is None:
        n_epochs = default_epochs

    default_batch_size = 32 if 'vit' in backbone.lower() else 32
    batch_size = kwargs.get('batch_size')
    if batch_size is None:
        batch_size = default_batch_size

    num_workers = kwargs.get('num_workers')
    if num_workers is None:
        num_workers = 4

    lr = kwargs.get('lr')
    if lr is None:
        lr = 0.01

    # Build command arguments
    cmd_args = [
        '--dataset', dataset_name,
        '--model', model,
        '--backbone', backbone,
        '--seed', str(seed),
        '--n_epochs', str(n_epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--num_workers', str(num_workers),
        '--non_verbose', '0',  # Enable verbose progress bars
        '--results_path', results_path,
        '--savecheck', 'last',
        '--base_path', './data',
        '--code_optimization', str(kwargs.get('code_optimization', 1))  # Enable automatic CUDA performance optimizations
    ]

    # Add model-specific arguments
    if model == 'derpp':
        cmd_args.extend(['--buffer_size', '500', '--alpha', '0.1', '--beta', '0.5'])
    elif model == 'ewc_on':
        cmd_args.extend(['--e_lambda', '1000', '--gamma', '1.0'])
    elif model == 'gpm':
        cmd_args.extend([
            '--gpm-threshold-base', '0.97',
            '--gpm-threshold-increment', '0.003',
            '--gpm-activation-samples', '512'
        ])
    elif model == 'dgr':
        cmd_args.extend([
            '--dgr-z-dim', '100',
            '--dgr-vae-lr', '0.001',
            '--dgr-replay-ratio', '0.5',
            '--dgr-temperature', '2.0'
        ])

    # CRITICAL: Ensure Einstellung integration is properly activated
    if not evaluation_only:
        logger.log("🧠 Activating Einstellung Effect evaluation integration...")
        # These flags ensure the Einstellung evaluator plugin is activated
        # and subset evaluation happens during training
        if '--einstellung_evaluation_subsets' not in cmd_args:
            cmd_args.extend(['--einstellung_evaluation_subsets', '1'])
        if '--einstellung_extract_attention' not in cmd_args:
            cmd_args.extend(['--einstellung_extract_attention', '1'])
        if '--einstellung_apply_shortcut' not in cmd_args:
            cmd_args.extend(['--einstellung_apply_shortcut', '1'])
        if '--einstellung_adaptation_threshold' not in cmd_args:
            cmd_args.extend(['--einstellung_adaptation_threshold', '0.8'])

        # Add cache enable/disable parameter
        enable_cache = kwargs.get('enable_cache', True)
        if '--einstellung_enable_cache' not in cmd_args:
            cmd_args.extend(['--einstellung_enable_cache', '1' if enable_cache else '0'])

        logger.log("   ✓ Einstellung flags added to command")
        logger.log(f"   ✓ Dataset caching: {'enabled' if enable_cache else 'disabled'}")
    else:
        logger.log("⚠️  Skipping Einstellung flags due to evaluation-only mode")

    # Add checkpoint arguments
    if used_checkpoint:
        logger.log("Starting evaluation...")
        logger.log("⚠️  Note: Evaluation-only mode provides basic metrics only")
        logger.log("    For full ERI analysis, use --force_retrain")
        eval_start = time.time()
    else:
        logger.log("Starting training with comprehensive Einstellung evaluation...")
        train_start = time.time()

    # Build full command
    cmd = [sys.executable, 'main.py'] + cmd_args

    # DEBUG: Log the full command
    logger.log("🚀 COMMAND EXECUTION DEBUG:")
    logger.log(f"   Command: {' '.join(cmd)}")
    logger.log(f"   Working directory: {os.getcwd()}")
    logger.log(f"   Dataset argument: --dataset {dataset_name}")
    logger.log(f"   Einstellung detection test: {'einstellung' in dataset_name.lower()}")

    # Run the experiment
    logger.log(f"Running command: {' '.join(cmd)}")
    logger.log("🚀 Starting training - you should see progress output below:")
    logger.log("")
    sys.stdout.flush()  # Ensure output is shown immediately

    output_lines: List[str] = []
    timeout_seconds = 7200

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd(),
            bufsize=1,
            universal_newlines=True,
        )

        assert process.stdout is not None
        for raw_line in iter(process.stdout.readline, ''):
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds and process.poll() is None:
                process.kill()
                raise subprocess.TimeoutExpired(cmd, timeout_seconds)

            line = raw_line.rstrip('\n')
            logger.log(line)
            output_lines.append(raw_line)

        process_stdout = process.stdout
        process_stdout.close()
        returncode = process.wait()

    except subprocess.TimeoutExpired:
        logger.log("❌ Experiment timed out after 2 hours")
        logger.log("   This suggests a performance issue, likely with ViT attention extraction")
        return {"success": False, "message": "Experiment timed out"}
    except Exception as e:
        logger.log(f"❌ Unexpected error: {str(e)}")
        return {"success": False, "message": f"Unexpected error: {str(e)}"}
    finally:
        if 'process' in locals() and process.stdout:
            process.stdout.close()

    logger.log("✓ Command execution completed")

    if returncode != 0:
        logger.log("❌ Experiment failed!")
        return {"success": False, "message": f"Experiment failed with return code {returncode}"}

    # Calculate timings
    end_time = time.time()
    total_time = end_time - start_time

    if used_checkpoint:
        evaluation_time = end_time - eval_start
    else:
        training_time = end_time - train_start

    # Extract comprehensive metrics from output
    full_output = ''.join(output_lines)
    comprehensive_metrics = reporter.extract_comprehensive_metrics_from_output(full_output)

    logger.log("=" * 40)
    logger.log("EXPERIMENT RESULTS")
    logger.log("=" * 40)
    logger.log(f"Final accuracy: {comprehensive_metrics['final_accuracy']:.1f}%")

    # Log Einstellung-specific metrics if available
    if comprehensive_metrics.get('einstellung_metrics'):
        em = comprehensive_metrics['einstellung_metrics']
        logger.log("🧠 EINSTELLUNG EFFECT ANALYSIS")
        logger.log(f"   ERI Score: {em.get('eri_score', 0):.3f}")
        logger.log(f"   Performance Deficit: {em.get('performance_deficit', 0):.3f}")
        logger.log(f"   Shortcut Feature Reliance: {em.get('shortcut_feature_reliance', 0):.3f}")
    else:
        if used_checkpoint:
            logger.log("⚠️  Limited metrics available in evaluation-only mode")
            logger.log("   Use --force_retrain for full ERI analysis")

    # Log subset accuracies if available
    if comprehensive_metrics.get('subset_accuracies'):
        logger.log("📊 SUBSET ACCURACIES")
        for subset, acc in comprehensive_metrics['subset_accuracies'].items():
            logger.log(f"   {subset}: {acc:.1f}%")

    # Log raw accuracies
    if comprehensive_metrics.get('raw_accuracies'):
        raw = comprehensive_metrics['raw_accuracies']
        logger.log("📈 RAW ACCURACIES")
        logger.log(f"   Class-IL: {raw.get('class_il', [])}")
        logger.log(f"   Task-IL: {raw.get('task_il', [])}")

    logger.log(f"Output directory: {results_path}")
    logger.log(f"Used checkpoint: {used_checkpoint}")

    # Prepare enhanced results dictionary
    experiment_results = {
        "success": True,
        "model": model,
        "backbone": backbone,
        "seed": seed,
        "final_accuracy": comprehensive_metrics['final_accuracy'],
        "raw_accuracies": comprehensive_metrics.get('raw_accuracies', {}),
        "subset_accuracies": comprehensive_metrics.get('subset_accuracies', {}),
        "einstellung_metrics": comprehensive_metrics.get('einstellung_metrics', {}),
        "used_checkpoint": used_checkpoint,
        "checkpoint_path": checkpoint_path,
        "training_time": training_time,
        "evaluation_time": evaluation_time,
        "total_time": total_time,
        "output_dir": results_path,
        "config": {
            "model": model,
            "backbone": backbone,
            "seed": seed,
            "execution_mode": execution_mode
        }
    }

    # Generate comprehensive Einstellung report
    try:
        comprehensive_report_file = reporter.generate_comprehensive_einstellung_report(
            experiment_results,
            comprehensive_metrics
        )
        logger.log(f"📄 Comprehensive Einstellung report: {comprehensive_report_file}")
    except Exception as e:
        logger.log(f"⚠️  Could not generate comprehensive report: {str(e)}")

    # Generate standard report
    report_file = reporter.generate_comprehensive_report(
        experiment_results,
        logger.get_captured_output()
    )

    logger.log(f"📄 Standard report generated: {report_file}")

    # NEW: Generate ERI visualization system outputs
    try:
        logger.log("🎨 Generating ERI visualization system outputs...")
        generate_eri_visualizations(results_path, experiment_results)
        logger.log("✅ ERI visualization system outputs generated successfully")
    except Exception as e:
        logger.log(f"⚠️  Could not generate ERI visualizations: {str(e)}")

    return experiment_results


def run_einstellung_experiment(strategy='derpp', backbone='resnet18', seed=42,
                             skip_training=False, force_retrain=False, auto_checkpoint=True,
                             debug=False, enable_cache=True, code_optimization=1):
    """
    Run a single Einstellung Effect experiment with enhanced checkpoint management.

    Args:
        strategy: Continual learning strategy
        backbone: Model backbone
        seed: Random seed
        skip_training: Skip training, only evaluate existing checkpoints
        force_retrain: Force retraining even if checkpoints exist
        auto_checkpoint: Automatically use existing checkpoints if found
        debug: If True, use shorter training epochs for faster testing
        enable_cache: If True, enable dataset caching for improved performance
        code_optimization: CUDA optimization level (0-3, default 1)

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running Einstellung Effect Experiment")
    print(f"Strategy: {strategy}")
    print(f"Backbone: {backbone}")
    print(f"Seed: {seed}")
    print(f"Mode: {'Evaluation-only' if skip_training else 'Training + Evaluation'}")
    print(f"{'='*60}")

    # Check for existing checkpoints
    existing_checkpoints = find_existing_checkpoints(strategy, backbone, seed)

    evaluation_only = False
    checkpoint_to_load = None

    if existing_checkpoints:
        print(f"📁 Found {len(existing_checkpoints)} existing checkpoint(s)")

        if skip_training:
            # Skip training mode: always use existing checkpoint
            checkpoint_to_load = existing_checkpoints[0]  # Use most recent
            evaluation_only = True
            print(f"✅ Using existing checkpoint: {os.path.basename(checkpoint_to_load)}")

        elif force_retrain:
            # Force retrain mode: ignore existing checkpoints
            print("🔄 Force retraining: ignoring existing checkpoints")

        elif auto_checkpoint:
            # Auto mode: use existing checkpoint without prompting
            checkpoint_to_load = existing_checkpoints[0]
            evaluation_only = True
            print(f"✅ Auto-using existing checkpoint: {os.path.basename(checkpoint_to_load)}")

        else:
            # Interactive mode: ask user what to do
            action = prompt_checkpoint_action(existing_checkpoints, strategy, backbone, seed)

            if action == 'use':
                checkpoint_to_load = existing_checkpoints[0]
                evaluation_only = True
                print(f"✅ Using existing checkpoint: {os.path.basename(checkpoint_to_load)}")
            elif action == 'retrain':
                print("🔄 Retraining from scratch")
            else:  # cancel
                print("❌ Cancelled by user")
                return {
                    'strategy': strategy,
                    'backbone': backbone,
                    'seed': seed,
                    'success': False,
                    'error': 'Cancelled by user'
                }
    else:
        if skip_training:
            print("❌ No existing checkpoints found for evaluation-only mode")
            return {
                'strategy': strategy,
                'backbone': backbone,
                'seed': seed,
                'success': False,
                'error': 'No checkpoints found for evaluation-only mode'
            }
        else:
            print("🆕 No existing checkpoints found. Starting training from scratch.")

    # Create experiment arguments
    cmd_args = create_einstellung_args(
        strategy, backbone, seed, evaluation_only, checkpoint_to_load,
        debug=debug, enable_cache=enable_cache, code_optimization=code_optimization
    )

    # Create output directory
    output_dir = f"./einstellung_results/{strategy}_{backbone}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    # Add results path to command
    cmd_args.extend(['--results_path', output_dir])

    try:
        # Run the experiment using subprocess
        action_desc = "evaluation" if evaluation_only else "training"
        print(f"Starting {action_desc}...")

        # Build full command
        cmd = [sys.executable, 'main.py'] + cmd_args

        print(f"Running command: {' '.join(cmd)}")

        # Run the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )

        # Stream output in real time
        output_lines = []
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
                output_lines.append(line)

        process.wait()

        if process.returncode != 0:
            print(f"ERROR: Process failed with return code {process.returncode}")
            return {
                'strategy': strategy,
                'backbone': backbone,
                'seed': seed,
                'success': False,
                'error': f'Process failed with return code {process.returncode}'
            }

        # Try to parse results from output
        full_output = ''.join(output_lines)

        # Look for final accuracy in output
        import re
        acc_pattern = r"Accuracy for \d+ task\(s\):\s+\[Class-IL\]:\s+([\d.]+) %"
        match = re.search(acc_pattern, full_output)

        final_acc = None
        if match:
            final_acc = float(match.group(1))

        print(f"\n{'='*40}")
        print("EXPERIMENT RESULTS")
        print(f"{'='*40}")
        print(f"Final accuracy: {final_acc}%")
        print(f"Output directory: {output_dir}")
        print(f"Used checkpoint: {checkpoint_to_load is not None}")

        return {
            'strategy': strategy,
            'backbone': backbone,
            'seed': seed,
            'final_accuracy': final_acc,
            'output_dir': output_dir,
            'used_checkpoint': checkpoint_to_load is not None,
            'checkpoint_path': checkpoint_to_load,
            'evaluation_only': evaluation_only,
            'success': True
        }

    except Exception as e:
        print(f"ERROR: Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'strategy': strategy,
            'backbone': backbone,
            'seed': seed,
            'success': False,
            'error': str(e)
        }


def run_comparative_experiment(skip_training=False, force_retrain=False, auto_checkpoint=True,
                              debug=False, enable_cache=True, code_optimization=1):
    """Run comparative experiments across different strategies."""

    print("🔬 Running Comparative Einstellung Effect Experiments")
    print("Testing cognitive rigidity across different continual learning strategies")

    # Experiment configurations
    # Baseline methods first for proper dependency management
    baseline_configs = [
        ('scratch_t2', 'resnet18'),
        ('interleaved', 'resnet18'),
    ]

    # Continual learning methods
    cl_configs = [
        ('sgd', 'resnet18'),
        ('derpp', 'resnet18'),
        ('ewc_on', 'resnet18'),
        ('gpm', 'resnet18'),
        ('dgr', 'resnet18'),
    ]

    # Combine baseline and CL methods (baselines first for dependency management)
    configs = baseline_configs + cl_configs

    print(f"📋 Running {len(configs)} experiments: {len(baseline_configs)} baselines + {len(cl_configs)} continual learning methods")

    results = []

    # Run individual experiments
    for i, (strategy, backbone) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running {strategy} with {backbone}...")

        result = run_einstellung_experiment(
            strategy, backbone, seed=42,
            skip_training=skip_training,
            force_retrain=force_retrain,
            auto_checkpoint=auto_checkpoint,
            debug=debug,
            enable_cache=enable_cache,
            code_optimization=code_optimization
        )

        if result:
            results.append(result)
            status = "✅" if result.get('success', False) else "❌"
            acc = f"{result.get('final_accuracy', 0):.2f}%" if result.get('success', False) else "FAILED"
            source = "Checkpoint" if result.get('used_checkpoint', False) else "Training"
            print(f"   {status} {strategy}/{backbone}: {acc} ({source})")
        else:
            print(f"   ❌ {strategy}/{backbone}: FAILED (no result)")

    # Aggregate results for comparative visualization (Task 8)
    successful_results = [r for r in results if r and r.get('success', False)]

    if len(successful_results) > 1:
        print(f"\n🔄 Aggregating results from {len(successful_results)} successful experiments...")

        try:
            # Create enhanced output structure (Task 15)
            print("📁 Creating structured output hierarchy...")
            comparative_output_dir = "./comparative_results"
            output_structure = create_enhanced_output_structure(comparative_output_dir)

            # Organize individual method results (Task 15)
            print("📋 Organizing individual method results...")
            organized_results = organize_individual_method_results(successful_results, output_structure)

            # Aggregate CSV results using existing function
            aggregated_csv = aggregate_comparative_results(successful_results, output_structure['aggregated_data'])

            # Generate comparative visualizations using existing ERI system
            print("📊 Generating comparative visualizations...")
            generate_eri_visualizations(output_structure['comparative_visualizations'], {
                'config': {'csv_path': aggregated_csv},
                'model': 'comparative_analysis'
            })

            # Generate statistical analysis report (Task 13 & 14)
            print("📈 Performing statistical significance testing...")
            statistical_results = {}
            try:
                from utils.statistical_analysis import generate_statistical_report
                statistical_report_path = generate_statistical_report(aggregated_csv, output_structure['statistical_analysis'])
                print(f"📊 Statistical analysis report: {statistical_report_path}")

                # Load statistical results for master report
                try:
                    from utils.statistical_analysis import StatisticalAnalyzer
                    analyzer = StatisticalAnalyzer(alpha=0.05, correction_method='bonferroni')
                    statistical_results = analyzer.analyze_comparative_metrics(aggregated_csv)
                except Exception as e:
                    print(f"⚠️  Statistical results loading failed: {e}")
                    statistical_results = {}

            except Exception as e:
                print(f"⚠️  Statistical analysis failed: {e}")
                print("Comparative visualizations are still available")
                statistical_results = {}

            # Compute comparative metrics for master report
            comparative_metrics = {}
            try:
                comparative_metrics = compute_comparative_metrics_from_aggregated_data(
                    output_structure['aggregated_data'], successful_results
                )
            except Exception as e:
                print(f"⚠️  Comparative metrics computation failed: {e}")
                comparative_metrics = {}

            # Generate master comparative report (Task 15)
            print("📄 Generating master comparative report...")
            try:
                master_report_path = generate_master_comparative_report(
                    successful_results, comparative_metrics, statistical_results, output_structure
                )
                print(f"📋 Master report: {master_report_path}")
            except Exception as e:
                print(f"⚠️  Master report generation failed: {e}")

            # Create publication-ready outputs (Task 15)
            print("📚 Creating publication-ready outputs...")
            try:
                publication_outputs = create_publication_ready_outputs(
                    output_structure, output_structure['comparative_visualizations']
                )
                print(f"📖 Publication-ready figures: {len(publication_outputs)} files created")
            except Exception as e:
                print(f"⚠️  Publication-ready output creation failed: {e}")

            # Generate experiment metadata (Task 15)
            print("🔬 Generating experiment metadata...")
            try:
                metadata_path = generate_experiment_metadata(
                    successful_results, comparative_metrics, statistical_results, output_structure
                )
                print(f"📊 Experiment metadata: {metadata_path}")
            except Exception as e:
                print(f"⚠️  Metadata generation failed: {e}")

            print(f"✅ Enhanced comparative analysis complete!")
            print(f"📁 Structured results saved to: {comparative_output_dir}")
            print(f"📊 Aggregated data: {aggregated_csv}")
            print(f"📋 Individual results organized in: {output_structure['individual_results']}")
            print(f"📈 Visualizations in: {output_structure['comparative_visualizations']}")
            print(f"📄 Reports in: {output_structure['reports']}")
            print(f"📚 Publication-ready outputs in: {output_structure['publication_ready']}")

        except Exception as e:
            print(f"⚠️  Comparative aggregation failed: {e}")
            print("Individual experiment results are still available in their respective directories")
    else:
        print(f"\n⚠️  Only {len(successful_results)} successful experiment(s) - skipping comparative analysis")
        print("Need at least 2 successful experiments for comparative visualization")

    # Enhanced summary comparison with comparative metrics
    print(f"\n{'='*120}")
    print("COMPARATIVE ANALYSIS SUMMARY")
    print(f"{'='*120}")

    # Check if we have baseline data for comparative metrics
    has_scratch_t2 = any(r.get('strategy') == 'scratch_t2' and r.get('success', False) for r in results)
    has_interleaved = any(r.get('strategy') == 'interleaved' and r.get('success', False) for r in results)

    # Compute comparative metrics if we have aggregated data and baselines
    comparative_metrics = {}
    statistical_results = {}
    if (has_scratch_t2 or has_interleaved) and len(successful_results) > 1:
        try:
            # Use aggregated_data directory from enhanced output structure
            aggregated_data_dir = output_structure.get('aggregated_data', comparative_output_dir) if 'output_structure' in locals() else comparative_output_dir
            comparative_metrics = compute_comparative_metrics_from_aggregated_data(
                aggregated_data_dir, successful_results
            )

            # Compute statistical significance for enhanced reporting (Task 14)
            try:
                from utils.statistical_analysis import StatisticalAnalyzer
                analyzer = StatisticalAnalyzer(alpha=0.05, correction_method='bonferroni')
                # Use the aggregated CSV from the enhanced structure
                aggregated_csv_path = os.path.join(aggregated_data_dir, "comparative_eri_metrics.csv") if 'aggregated_csv' not in locals() else aggregated_csv
                statistical_results = analyzer.analyze_comparative_metrics(aggregated_csv_path)
            except Exception as e:
                print(f"⚠️  Statistical analysis for reporting failed: {e}")
                statistical_results = {}

        except Exception as e:
            print(f"⚠️  Failed to compute comparative metrics: {e}")

    # Enhanced table headers with statistical significance indicators
    if has_scratch_t2 or has_interleaved:
        print(f"{'Strategy':<15} {'Backbone':<12} {'PD_t':<10} {'SFR_rel':<10} {'AD':<8} {'Final Acc':<12} {'Sig.':<5} {'Source':<12}")
        print(f"{'-'*125}")
    else:
        print(f"{'Strategy':<15} {'Backbone':<12} {'Final Acc':<12} {'Source':<12} {'Status':<20}")
        print(f"{'-'*80}")

    for result in results:
        if result and result.get('success', False):
            strategy = result.get('strategy', 'Unknown')
            backbone = result.get('backbone', 'Unknown')
            final_acc = f"{result.get('final_accuracy', 0):.2f}%"
            source = "Checkpoint" if result.get('used_checkpoint', False) else "Training"

            if has_scratch_t2 or has_interleaved:
                # Enhanced reporting with comparative metrics when baselines available
                method_metrics = comparative_metrics.get(strategy, {})

                # Format PD_t (Performance Deficit at final epoch)
                pd_t = method_metrics.get('pd_t_final', None)
                pd_str = f"{pd_t:.3f}" if pd_t is not None else "N/A"

                # Format SFR_rel (Shortcut Forgetting Rate relative at final epoch)
                sfr_rel = method_metrics.get('sfr_rel_final', None)
                sfr_str = f"{sfr_rel:.3f}" if sfr_rel is not None else "N/A"

                # Format AD (Adaptation Delay)
                ad = method_metrics.get('adaptation_delay', None)
                ad_str = f"{ad:.1f}" if ad is not None else "N/A"

                # Statistical significance indicators (Task 14)
                sig_indicator = get_significance_indicator(strategy, statistical_results)

                print(f"{strategy:<15} {backbone:<12} {pd_str:<10} {sfr_str:<10} {ad_str:<8} {final_acc:<12} {sig_indicator:<5} {source:<12}")
            else:
                # Basic reporting when baselines not available
                print(f"{strategy:<15} {backbone:<12} {final_acc:<12} {source:<12} {'Success':<20}")
        else:
            strategy = result.get('strategy', 'Unknown') if result else 'Unknown'
            backbone = result.get('backbone', 'Unknown') if result else 'Unknown'
            if has_scratch_t2 or has_interleaved:
                print(f"{strategy:<15} {backbone:<12} {'FAILED':<10} {'FAILED':<10} {'FAILED':<8} {'FAILED':<12} {'':<5} {'FAILED':<12}")
            else:
                print(f"{strategy:<15} {backbone:<12} {'FAILED':<12} {'FAILED':<12} {'Failed':<20}")

    # Missing baseline warnings
    if not has_scratch_t2 and not has_interleaved:
        print(f"\n⚠️  Missing Baseline Methods:")
        print(f"   • No baseline methods (scratch_t2, interleaved) were run successfully")
        print(f"   • Comparative metrics (PD_t, SFR_rel, AD) cannot be computed")
        print(f"   • Run baseline methods first to enable full comparative analysis")
    elif not has_scratch_t2:
        print(f"\n⚠️  Missing Scratch_T2 Baseline:")
        print(f"   • Scratch_T2 baseline not available - some comparative metrics may be incomplete")
        print(f"   • PD_t and SFR_rel calculations require Scratch_T2 as reference")
    elif not has_interleaved:
        print(f"\n⚠️  Missing Interleaved Baseline:")
        print(f"   • Interleaved baseline not available - using Scratch_T2 as primary reference")

    # Interpretation guide with statistical significance explanation
    if has_scratch_t2 or has_interleaved:
        print(f"\n📖 Interpretation Guide:")
        print(f"   • PD_t: Performance Deficit relative to Scratch_T2 (higher = worse)")
        print(f"   • SFR_rel: Shortcut Forgetting Rate relative to Scratch_T2 (higher = more forgetting)")
        print(f"   • AD: Adaptation Delay in epochs to reach threshold (higher = slower adaptation)")
        print(f"   • Sig.: Statistical significance indicators (*** p<0.001, ** p<0.01, * p<0.05)")
        print(f"   • Baseline methods provide reference points for measuring continual learning effectiveness")

        # Statistical summary if available
        if statistical_results and 'interpretation' in statistical_results:
            print(f"\n📊 Statistical Summary:")
            interpretation = statistical_results['interpretation']
            if 'significant_differences' in interpretation:
                print(f"   • {interpretation['significant_differences']}")
            if 'large_effects' in interpretation and interpretation['large_effects'] != "No large effect sizes detected between methods.":
                print(f"   • {interpretation['large_effects'][:100]}...")  # Truncate for console

    print(f"\n📊 Experiment Summary:")
    print(f"   • Total experiments: {len(results)}")
    print(f"   • Successful: {len(successful_results)}")
    print(f"   • Failed: {len(results) - len(successful_results)}")
    print(f"   • Baseline methods available: {has_scratch_t2 or has_interleaved}")

    return results


def main():
    """Main entry point with enhanced reporting."""
    parser = argparse.ArgumentParser(description='Einstellung Effect Experiments with Comprehensive Reporting')

    # Experiment type
    parser.add_argument('--comparative', action='store_true',
                       help='Run comparative experiments across strategies')

    # Single experiment parameters
    parser.add_argument('--model', type=str, default='derpp',
                   choices=['sgd', 'derpp', 'ewc_on', 'gpm', 'dgr', 'scratch_t2', 'interleaved'],
                       help='Continual learning strategy (includes baseline methods)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'vit'],
                       help='Model backbone')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--epochs', type=int,
                       help='Override number of training epochs (defaults depend on backbone)')

    # Checkpoint management
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only evaluate existing checkpoints')
    parser.add_argument('--force_retrain', action='store_true',
                       help='Force retraining even if checkpoints exist')
    parser.add_argument('--auto_checkpoint', action='store_true',
                       help='Automatically use existing checkpoints if found')

    # Performance optimization
    parser.add_argument('--code_optimization', type=int, default=1, choices=[0, 1, 2, 3],
                       help='CUDA performance optimization level (0=none, 1=TF32+cuDNN, 2=+BF16, 3=+torch.compile)')

    # Dataset caching
    parser.add_argument('--enable_cache', action='store_true', default=True,
                       help='Enable dataset caching for improved performance (default: True)')
    parser.add_argument('--disable_cache', action='store_true',
                       help='Disable dataset caching for debugging or comparison purposes')

    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (shorter training epochs for faster testing)')

    args = parser.parse_args()

    # Handle cache enable/disable logic
    enable_cache = args.enable_cache and not args.disable_cache

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Determine execution mode
    if args.skip_training:
        execution_mode = "skip_training"
    elif args.force_retrain:
        execution_mode = "force_retrain"
    elif args.auto_checkpoint:
        execution_mode = "auto_checkpoint"
    else:
        execution_mode = "interactive"

    # Run experiments
    if args.comparative:
        # Use the enhanced comparative experiment runner with aggregation
        results = run_comparative_experiment(
            skip_training=args.skip_training,
            force_retrain=args.force_retrain,
            auto_checkpoint=args.auto_checkpoint,
            debug=args.debug,
            enable_cache=enable_cache,
            code_optimization=args.code_optimization
        )

        successful_results = [r for r in results if r and r.get('success', False)]
        print(f"\n📊 Comparative experiment completed: {len(successful_results)}/{len(results)} successful")

        return results
    else:
        # Single experiment
        results_path = f"./einstellung_results/{args.model}_{args.backbone}_seed{args.seed}"

        result = run_experiment(
            model=args.model,
            backbone=args.backbone,
            seed=args.seed,
            results_path=results_path,
            execution_mode=execution_mode,
            epochs=args.epochs,
            code_optimization=args.code_optimization,
            debug=args.debug,
            enable_cache=enable_cache
        )

        if result and result.get('success', False):
            print(f"✅ Experiment completed successfully!")
            print(f"📊 Final accuracy: {result.get('final_accuracy', 0):.2f}%")
            print(f"📄 Used checkpoint: {result.get('used_checkpoint', False)}")
            print(f"📁 Results saved in: {result.get('output_dir', 'N/A')}")
            print(f"📋 Comprehensive report: {result.get('output_dir', 'N/A')}/reports/experiment_report.html")
            return [result]
        else:
            print(f"❌ Experiment failed: {result.get('message', 'Unknown error') if result else 'Unknown error'}")
            sys.exit(1)


if __name__ == '__main__':
    main()
