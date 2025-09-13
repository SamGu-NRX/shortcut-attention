#!/usr/bin/env python3
# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple Einstellung Effect Experiment Runner

This script demonstrates how to run Einstellung Effect experiments using the
integrated system. It shows:

1. Basic usage with DER++ and EWC strategies
2. Configuration of Einstellung parameters
3. Results collection and analysis
4. Attention analysis for ViT models

Usage:
    python run_einstellung_experiment.py --model derpp --backbone resnet18
    python run_einstellung_experiment.py --model ewc_on --backbone vit_base_patch16_224
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure Mammoth modules are importable
sys.path.append(str(Path(__file__).parent))

# Import Mammoth's main training function
from main import main as mammoth_main
from utils.args import add_experiment_args, add_management_args
from utils.einstellung_integration import enable_einstellung_integration, get_einstellung_evaluator


def create_einstellung_args(strategy='derpp', backbone='resnet18', seed=42):
    """
    Create arguments namespace for Einstellung experiments directly.

    Args:
        strategy: Continual learning strategy ('derpp', 'ewc_on', 'sgd')
        backbone: Model backbone ('resnet18', 'vit')
        seed: Random seed

    Returns:
        argparse.Namespace with experiment configuration
    """
    import argparse

    # Determine dataset and parameters based on backbone
    if backbone == 'vit':
        dataset_name = 'seq-cifar100-einstellung-224'
        patch_size = 16  # Larger for 224x224 images
        batch_size = 32
        n_epochs = 20
    else:
        dataset_name = 'seq-cifar100-einstellung'
        patch_size = 4   # Smaller for 32x32 images
        batch_size = 32
        n_epochs = 50

    # Use subprocess to call main.py instead of using mammoth_main directly
    cmd_args = [
        '--dataset', dataset_name,
        '--model', strategy,
        '--backbone', backbone,
        '--n_epochs', str(n_epochs),
        '--batch_size', str(batch_size),
        '--lr', '0.01',
        '--seed', str(seed)
        # Note: csv_log doesn't exist in Mammoth, removed
    ]

    # Strategy-specific parameters
    if strategy == 'derpp':
        cmd_args.extend([
            '--buffer_size', '500',
            '--alpha', '0.1',
            '--beta', '0.5'
        ])
    elif strategy == 'ewc_on':
        cmd_args.extend([
            '--e_lambda', '1000',
            '--gamma', '1.0'
        ])

    # Einstellung parameters
    cmd_args.extend([
        '--einstellung_patch_size', str(patch_size),
        '--einstellung_patch_color', '255', '0', '255',
        '--einstellung_adaptation_threshold', '0.8',
        '--einstellung_apply_shortcut', '1',
        '--einstellung_evaluation_subsets', '1',
        '--einstellung_extract_attention', '1'
    ])

    return cmd_args


def run_einstellung_experiment(strategy='derpp', backbone='resnet18', seed=42):
    """
    Run a single Einstellung Effect experiment using subprocess.

    Args:
        strategy: Continual learning strategy
        backbone: Model backbone
        seed: Random seed

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running Einstellung Effect Experiment")
    print(f"Strategy: {strategy}")
    print(f"Backbone: {backbone}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")

    # Create experiment arguments
    cmd_args = create_einstellung_args(strategy, backbone, seed)

    # Create output directory
    output_dir = f"./einstellung_results/{strategy}_{backbone}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    # Add results path to command
    cmd_args.extend(['--results_path', output_dir])

    try:
        # Run the experiment using subprocess
        print("Starting training...")

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
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            output_lines.append(line)

        process.wait()

        if process.returncode != 0:
            print(f"ERROR: Process failed with return code {process.returncode}")
            return None

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

        return {
            'strategy': strategy,
            'backbone': backbone,
            'seed': seed,
            'final_accuracy': final_acc,
            'output_dir': output_dir,
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


def run_comparative_experiment():
    """Run comparative experiments across different strategies."""

    print("Running Comparative Einstellung Effect Experiments")
    print("Testing cognitive rigidity across different continual learning strategies")

    # Experiment configurations
    configs = [
        ('sgd', 'resnet18'),
        ('derpp', 'resnet18'),
        ('ewc_on', 'resnet18'),
    ]

    # Add ViT experiments if available
    try:
        configs.append(('derpp', 'vit_base_patch16_224'))
    except:
        print("ViT backbone not available, skipping attention analysis")

    results = []

    for strategy, backbone in configs:
        result = run_einstellung_experiment(strategy, backbone, seed=42)
        if result:
            results.append(result)

    # Summary comparison
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    print(f"{'Strategy':<15} {'Backbone':<20} {'ERI Score':<12} {'Perf. Deficit':<15} {'Ad. Delay':<12}")
    print(f"{'-'*80}")

    for result in results:
        metrics = result['metrics']
        eri = f"{metrics.eri_score:.4f}" if metrics.eri_score else "N/A"
        pd = f"{metrics.performance_deficit:.4f}" if metrics.performance_deficit else "N/A"
        ad = f"{metrics.adaptation_delay:.1f}" if metrics.adaptation_delay else "N/A"

        print(f"{result['strategy']:<15} {result['backbone']:<20} {eri:<12} {pd:<15} {ad:<12}")

    print(f"\nHigher ERI scores indicate more cognitive rigidity (worse adaptation)")
    print(f"Performance Deficit: accuracy drop when shortcuts are removed")
    print(f"Adaptation Delay: epochs required to reach 80% accuracy")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Einstellung Effect Experiments')

    # Experiment type
    parser.add_argument('--comparative', action='store_true',
                       help='Run comparative experiments across strategies')

    # Single experiment parameters
    parser.add_argument('--model', type=str, default='derpp',
                       choices=['sgd', 'derpp', 'ewc_on'],
                       help='Continual learning strategy')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'vit_base_patch16_224'],
                       help='Model backbone')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run experiments
    if args.comparative:
        results = run_comparative_experiment()
    else:
        result = run_einstellung_experiment(args.model, args.backbone, args.seed)
        results = [result] if result else []

    if results:
        print(f"\n✓ Successfully completed {len(results)} experiment(s)")
    else:
        print("\n✗ No experiments completed successfully")
        sys.exit(1)


if __name__ == '__main__':
    main()
