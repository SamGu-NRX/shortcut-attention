# run_shortcut_experiment.py
#!/usr/bin/env python3
"""
Simple script to run the shortcut investigation experiment.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def run_experiment(method='derpp', seed=42, epochs=10, debug_val=0, use_gpu=True, num_data_workers=2):
    """
    Run a single experiment with the specified parameters.
    
    Args:
        method: Continual learning method ('derpp' or 'ewc_on')
        seed: Random seed for reproducibility
        epochs: Number of training epochs
        debug_val: Value for Mammoth's debug_mode (e.g., 0 or 1)
        use_gpu: Boolean, whether to attempt running on GPU (adds --device cuda if True)
        num_data_workers: Number of workers for DataLoader
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
        '--optimizer', 'sgd',
        '--optim_wd', '0.0',
        '--optim_mom', '0.0',
        '--num_workers', str(num_data_workers), # Use parameter
        '--drop_last', '0', 
        '--debug_mode', str(debug_val),
    ]

    if use_gpu:
        # Assuming main.py will pick up CUDA if available.
        # If explicit setting is needed and main.py supports --device:
        # cmd.extend(['--device', 'cuda'])
        pass # Rely on main.py's auto-detection or its own --device flag
    else:
        # If you want to force CPU for some reason (and main.py supports --device)
        # cmd.extend(['--device', 'cpu'])
        pass
    
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
        
    print(f"Running command: {' '.join(cmd)}")
    
    # Adjust timeout: 10 minutes might be too short for ViT on CPU for 2 tasks, 1 epoch each.
    # Let's try 30 minutes (1800 seconds) for CPU, or keep 10 min if expecting GPU.
    # If on GPU, 1 epoch should be much faster.
    timeout_seconds = 18000 if not use_gpu else 6000 
    if epochs > 1: # Increase timeout for more epochs
        timeout_seconds = timeout_seconds * epochs

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_seconds) 
        print("Experiment completed successfully!")
        print("STDOUT (last 500 chars):", result.stdout[-500:])
        if result.stderr:
             print("STDERR (last 500 chars):", result.stderr[-500:])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error code {e.returncode}: {e}")
        print("STDOUT:", e.stdout) # Print full STDOUT on error
        print("STDERR:", e.stderr) # Print full STDERR on error
        return False
    except subprocess.TimeoutExpired as e:
        print(f"Experiment timed out after {e.timeout} seconds.")
        stdout_decoded = e.stdout.decode(errors='ignore') if e.stdout else "No STDOUT"
        stderr_decoded = e.stderr.decode(errors='ignore') if e.stderr else "No STDERR"
        print("STDOUT (last 1000 chars):", stdout_decoded[-1000:])
        print("STDERR (last 1000 chars):", stderr_decoded[-1000:])
        return False


def run_comparison_experiment(attempt_gpu=True, workers=2):
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
            print(f"\nRunning {method} with seed {seed} for {epochs} epoch(s)...")
            success = run_experiment(method=method, seed=seed, epochs=epochs, 
                                     debug_val=debug_value_for_main, 
                                     use_gpu=attempt_gpu,
                                     num_data_workers=workers) 
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
    print("3. If still failing/timing out on CPU, consider reducing workload further for tests or ensure GPU access.")
    print("="*60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run shortcut investigation experiment')
    parser.add_argument('--method', type=str, choices=['derpp', 'ewc_on', 'both'], 
                       default='both', help='Method to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--debug_main', type=int, choices=[0, 1], default=0, 
                        help='Set debug_mode for Mammoth main.py (0 or 1)')
    parser.add_argument('--cpu_only', action='store_true', help='Force CPU execution for the experiment run')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--comparison', action='store_true', 
                       help='Run full comparison experiment')
    
    script_args = parser.parse_args()
    
    use_gpu_flag = not script_args.cpu_only

    if script_args.comparison:
        run_comparison_experiment(attempt_gpu=use_gpu_flag, workers=script_args.workers)
    elif script_args.method == 'both':
        print("Running both methods...")
        for method_name in ['derpp', 'ewc_on']:
            print(f"\nRunning {method_name}...")
            run_experiment(method=method_name, seed=script_args.seed, 
                           epochs=script_args.epochs, debug_val=script_args.debug_main,
                           use_gpu=use_gpu_flag, num_data_workers=script_args.workers)
    else:
        print(f"Running {script_args.method}...")
        run_experiment(method=script_args.method, seed=script_args.seed, 
                       epochs=script_args.epochs, debug_val=script_args.debug_main,
                       use_gpu=use_gpu_flag, num_data_workers=script_args.workers)


if __name__ == "__main__":
    main()