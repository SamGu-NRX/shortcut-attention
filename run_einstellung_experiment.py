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
    Create command line arguments for Einstellung experiments.

    Args:
        strategy: Continual learning strategy ('derpp', 'ewc_on', 'sgd')
        backbone: Model backbone ('resnet18', 'vit_base_patch16_224')
        seed: Random seed

    Returns:
        argparse.Namespace with experiment configuration
    """
    # Create argument parser like Mammoth does
    parser = argparse.ArgumentParser(description='Einstellung Effect Experiments')

    # Add Mammoth's standard arguments
    add_management_args(parser)
    add_experiment_args(parser)

    # Einstellung-specific arguments
    parser.add_argument('--einstellung_patch_size', type=int, default=4,
                       help='Size of the magenta shortcut patch')
    parser.add_argument('--einstellung_patch_color', nargs=3, type=int,
                       default=[255, 0, 255],
                       help='RGB color of the shortcut patch')
    parser.add_argument('--einstellung_adaptation_threshold', type=float, default=0.8,
                       help='Accuracy threshold for Adaptation Delay calculation')
    parser.add_argument('--einstellung_apply_shortcut', action='store_true',
                       help='Apply shortcuts during training')
    parser.add_argument('--einstellung_mask_shortcut', action='store_true',
                       help='Mask shortcuts during evaluation')
    parser.add_argument('--einstellung_evaluation_subsets', action='store_true', default=True,
                       help='Enable multi-subset evaluation')
    parser.add_argument('--einstellung_extract_attention', action='store_true', default=True,
                       help='Extract attention maps for analysis')

    # Create arguments list
    args_list = [
        '--dataset', 'seq-cifar100-einstellung',
        '--model', strategy,
        '--backbone', backbone,
        '--n_epochs', '50',
        '--batch_size', '32',
        '--lr', '0.01',
        '--seed', str(seed),
        '--csv_log',
        '--nowand',  # Disable wandb for demonstration
        '--non_verbose'
    ]

    # Strategy-specific parameters
    if strategy == 'derpp':
        args_list.extend([
            '--buffer_size', '500',
            '--alpha', '0.1',
            '--beta', '0.5'
        ])
    elif strategy == 'ewc_on':
        args_list.extend([
            '--e_lambda', '1000',
            '--gamma', '1.0'
        ])

    # Einstellung parameters
    args_list.extend([
        '--einstellung_patch_size', '4',
        '--einstellung_patch_color', '255', '0', '255',
        '--einstellung_adaptation_threshold', '0.8',
        '--einstellung_apply_shortcut',
        '--einstellung_evaluation_subsets',
        '--einstellung_extract_attention'
    ])

    # Parse arguments
    args = parser.parse_args(args_list)

    return args


def run_einstellung_experiment(strategy='derpp', backbone='resnet18', seed=42):
    """
    Run a single Einstellung Effect experiment.

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
    args = create_einstellung_args(strategy, backbone, seed)

    # Create output directory
    output_dir = f"./einstellung_results/{strategy}_{backbone}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    # Enable Einstellung integration
    integration_enabled = enable_einstellung_integration(args)

    if not integration_enabled:
        print("ERROR: Failed to enable Einstellung integration")
        return None

    print("✓ Einstellung integration enabled")

    try:
        # Run the experiment using Mammoth's main function
        print("Starting training...")
        result = mammoth_main(args)

        # Get the evaluator and final metrics
        evaluator = get_einstellung_evaluator()

        if evaluator is not None:
            final_metrics = evaluator.get_final_metrics()

            print(f"\n{'='*40}")
            print("EXPERIMENT RESULTS")
            print(f"{'='*40}")

            if final_metrics.adaptation_delay is not None:
                print(f"Adaptation Delay: {final_metrics.adaptation_delay} epochs")

            if final_metrics.performance_deficit is not None:
                print(f"Performance Deficit: {final_metrics.performance_deficit:.4f}")

            if final_metrics.shortcut_feature_reliance is not None:
                print(f"Shortcut Feature Reliance: {final_metrics.shortcut_feature_reliance:.4f}")

            if final_metrics.eri_score is not None:
                print(f"ERI Score: {final_metrics.eri_score:.4f}")

            # Export detailed results
            results_file = f"{output_dir}/detailed_results.json"
            evaluator.export_results(results_file)
            print(f"\nDetailed results saved to: {results_file}")

            return {
                'strategy': strategy,
                'backbone': backbone,
                'seed': seed,
                'metrics': final_metrics,
                'output_dir': output_dir
            }
        else:
            print("WARNING: No Einstellung evaluator found")
            return None

    except Exception as e:
        print(f"ERROR: Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


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
