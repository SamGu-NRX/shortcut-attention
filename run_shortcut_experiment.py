#!/usr/bin/env python3
"""
Simple script to run the shortcut investigation experiment.

This script provides an easy way to run the continual learning experiment
investigating shortcut features with DER++ and EWC methods.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def run_experiment(method='derpp', seed=42, epochs=10, debug=False):
    """
    Run a single experiment with the specified parameters.
    
    Args:
        method: Continual learning method ('derpp' or 'ewc_on')
        seed: Random seed for reproducibility
        epochs: Number of training epochs
        debug: Whether to run in debug mode
    """
    
    # Base command
    cmd = [
        sys.executable, 'main.py',
        '--dataset', 'seq-cifar10-custom',
        '--model', method,
        '--backbone', 'vit',
        '--seed', str(seed),
        '--n_epochs', str(epochs),
        '--batch_size', '32',
        '--lr', '0.01',
        '--nowand', '1',  # Disable wandb
    ]
    
    # Method-specific arguments
    if method == 'derpp':
        cmd.extend([
            '--buffer_size', '200',
            '--alpha', '0.1',
            '--beta', '0.5',
        ])
    elif method == 'ewc_on':
        cmd.extend([
            '--e_lambda', '0.4',
            '--gamma', '0.85',
        ])
    
    if debug:
        cmd.append('--debug_mode')
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Experiment completed successfully!")
        print("STDOUT:", result.stdout[-500:])  # Last 500 characters
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
        print("STDERR:", e.stderr[-500:])  # Last 500 characters
        return False


def run_comparison_experiment():
    """
    Run a comparison experiment with both DER++ and EWC methods.
    """
    print("=" * 60)
    print("SHORTCUT FEATURE INVESTIGATION EXPERIMENT")
    print("=" * 60)
    print()
    print("This experiment investigates how different continual learning")
    print("methods handle shortcut features using a custom CIFAR-10 setup:")
    print()
    print("Task 1: airplane, automobile (potential shortcuts: sky, road)")
    print("Task 2: bird, truck (potential shortcuts: sky, road/wheels)")
    print()
    print("Methods to compare:")
    print("- DER++ (memory-based)")
    print("- EWC (regularization-based)")
    print()
    print("Architecture: Vision Transformer (for attention analysis)")
    print("=" * 60)
    print()
    
    methods = ['derpp', 'ewc_on']
    seeds = [42, 123, 456]
    epochs = 10
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*20} Running {method.upper()} {'='*20}")
        results[method] = {}
        
        for seed in seeds:
            print(f"\nRunning {method} with seed {seed}...")
            success = run_experiment(method=method, seed=seed, epochs=epochs)
            results[method][seed] = success
            
            if success:
                print(f"✓ {method} with seed {seed} completed successfully")
            else:
                print(f"✗ {method} with seed {seed} failed")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for method in methods:
        successful_runs = sum(results[method].values())
        total_runs = len(results[method])
        print(f"{method.upper()}: {successful_runs}/{total_runs} successful runs")
        
        for seed, success in results[method].items():
            status = "✓" if success else "✗"
            print(f"  Seed {seed}: {status}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. Check the results/ directory for experiment outputs")
    print("2. Run attention and activation analysis on successful experiments")
    print("3. Compare attention patterns between methods")
    print("4. Investigate shortcut feature usage")
    print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run shortcut investigation experiment')
    parser.add_argument('--method', type=str, choices=['derpp', 'ewc_on', 'both'], 
                       default='both', help='Method to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--comparison', action='store_true', 
                       help='Run full comparison experiment')
    
    args = parser.parse_args()
    
    if args.comparison:
        run_comparison_experiment()
    elif args.method == 'both':
        print("Running both methods...")
        for method in ['derpp', 'ewc_on']:
            print(f"\nRunning {method}...")
            run_experiment(method=method, seed=args.seed, epochs=args.epochs, debug=args.debug)
    else:
        print(f"Running {args.method}...")
        run_experiment(method=args.method, seed=args.seed, epochs=args.epochs, debug=args.debug)


if __name__ == "__main__":
    main()
