# run_shortcut_experiment.py
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

def run_experiment(method='derpp', seed=42, epochs=10, debug_val=0): # Changed debug to debug_val
    """
    Run a single experiment with the specified parameters.
    
    Args:
        method: Continual learning method ('derpp' or 'ewc_on')
        seed: Random seed for reproducibility
        epochs: Number of training epochs
        debug_val: Value for Mammoth's debug_mode (e.g., 0 or 1)
    """
    
    # Base command
    cmd = [
        sys.executable, 'main.py',
        '--dataset', 'seq-cifar10-224-custom',
        '--model', method,
        '--backbone', 'vit',
        '--seed', str(seed),
        '--n_epochs', str(epochs),
        '--batch_size', '32', 
        '--lr', '0.01',
        # Removed --wandb_mode disabled. Wandb should not init if project/entity are not set.
        '--optimizer', 'sgd',
        '--optim_wd', '0.0',
        '--optim_mom', '0.0',
        '--num_workers', '0', 
        '--drop_last', '0', 
        '--debug_mode', str(debug_val), # Standard Mammoth debug mode
        # The following are often defaulted by the dataset or main.py's arg parser,
        # but providing them can sometimes avoid issues if defaults are unexpected.
        # '--validation_mode', 'current', # Defaulted by our dataset
        # '--custom_class_order', '0,1,2,9', # Defaulted by our dataset
        # '--permute_classes', '0', # Defaulted by our dataset (to False)
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
    
    # Note: debug_mode is now passed as a value (0 or 1)
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600) 
        print("Experiment completed successfully!")
        print("STDOUT (last 500 chars):", result.stdout[-500:])
        if result.stderr:
             print("STDERR (last 500 chars):", result.stderr[-500:])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error code {e.returncode}: {e}")
        print("STDOUT (last 500 chars):", e.stdout[-500:])
        print("STDERR (last 500 chars):", e.stderr[-500:])
        return False
    except subprocess.TimeoutExpired as e:
        print(f"Experiment timed out after {e.timeout} seconds.")
        if e.stdout:
            print("STDOUT (last 500 chars):", e.stdout.decode(errors='ignore')[-500:])
        if e.stderr:
            print("STDERR (last 500 chars):", e.stderr.decode(errors='ignore')[-500:])
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
    seeds = [42] # Start with one seed for faster testing
    epochs = 1 # Make epochs very short for the first successful run test
    debug_value_for_main = 0 # 0 for no extensive debug, 1 for debug
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*20} Running {method.upper()} {'='*20}")
        results[method] = {}
        
        for seed in seeds:
            print(f"\nRunning {method} with seed {seed}...")
            success = run_experiment(method=method, seed=seed, epochs=epochs, debug_val=debug_value_for_main) 
            results[method][seed] = success
            
            if success:
                print(f"✓ {method} with seed {seed} completed successfully")
            else:
                print(f"✗ {method} with seed {seed} failed")
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for method in methods:
        successful_runs = sum(results[method].values())
        total_runs = len(results[method])
        print(f"{method.upper()}: {successful_runs}/{total_runs} successful runs")
        
        for seed, success_flag in results[method].items():
            status = "✓" if success_flag else "✗"
            print(f"  Seed {seed}: {status}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. Check the results/ directory for experiment outputs.")
    print("2. If experiments ran, proceed with attention/activation analysis.")
    print("3. If still failing, carefully examine the full STDOUT/STDERR from main.py.")
    print("   Look for missing arguments or other setup issues reported by main.py.")
    print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run shortcut investigation experiment')
    parser.add_argument('--method', type=str, choices=['derpp', 'ewc_on', 'both'], 
                       default='both', help='Method to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs (short for testing)') # Default to 1
    # Changed debug flag to control debug_val for main.py
    parser.add_argument('--debug_main', type=int, choices=[0, 1], default=0, 
                        help='Set debug_mode for Mammoth main.py (0 or 1)')
    parser.add_argument('--comparison', action='store_true', 
                       help='Run full comparison experiment')
    
    script_args = parser.parse_args()
    
    if script_args.comparison:
        run_comparison_experiment() # debug_main is handled inside run_comparison_experiment
    elif script_args.method == 'both':
        print("Running both methods...")
        for method_name in ['derpp', 'ewc_on']:
            print(f"\nRunning {method_name}...")
            run_experiment(method=method_name, seed=script_args.seed, 
                           epochs=script_args.epochs, debug_val=script_args.debug_main)
    else:
        print(f"Running {script_args.method}...")
        run_experiment(method=script_args.method, seed=script_args.seed, 
                       epochs=script_args.epochs, debug_val=script_args.debug_main)


if __name__ == "__main__":
    main()