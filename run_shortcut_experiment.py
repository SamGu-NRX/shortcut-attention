# run_shortcut_experiment.py
#!/usr/bin/env python3
"""
Experiment manager for investigating shortcut features in continual learning.

This script runs experiments, parses results from stdout, aggregates them,
and saves them to a JSON file. It also provides a clean summary of results.
"""

import os
import sys
import argparse
import subprocess
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple

import pandas as pd
from tqdm import tqdm

# --- Configuration ---
# Centralize experiment parameters for easy modification.
EXPERIMENT_CONFIG = {
    "methods": ['derpp', 'ewc_on'],
    "seeds": [42, 123, 456],
    "epochs_per_task": 10, # Set the desired number of epochs for the actual run
    "batch_size": 32,
    "lr": 0.01,
    "optimizer": "sgd",
    "num_workers": 4, # Increase workers if your machine has multiple cores
}

def parse_results(output: str) -> Dict[str, float]:
    """
    Parses the stdout of a Mammoth experiment to extract key metrics.

    Args:
        output: The captured stdout string from the subprocess.

    Returns:
        A dictionary containing parsed accuracy values.
    """
    results = {}
    try:
        # Regex to find the final accuracy line
        # Example: "Accuracy for 2 task(s): [Class-IL]: 97.8 % [Task-IL]: 99.38 %"
        final_acc_pattern = re.compile(
            r"Accuracy for \d+ task\(s\):\s+\[Class-IL\]:\s+([\d.]+) %\s+\[Task-IL\]:\s+([\d.]+) %"
        )
        match = final_acc_pattern.search(output)
        if match:
            results['final_acc_cil'] = float(match.group(1))
            results['final_acc_til'] = float(match.group(2))

        # Regex to find the raw accuracy values for more detailed analysis
        # Example: "Raw accuracy values: Class-IL [96.55, 99.05] | Task-IL [99.7, 99.05]"
        raw_acc_pattern = re.compile(
            r"Raw accuracy values: Class-IL \[(.*?)\] \| Task-IL \[(.*?)\]"
        )
        raw_match = raw_acc_pattern.search(output)
        if raw_match:
            results['raw_acc_cil'] = [float(x.strip()) for x in raw_match.group(1).split(',')]
            results['raw_acc_til'] = [float(x.strip()) for x in raw_match.group(2).split(',')]

    except Exception as e:
        print(f"Warning: Could not parse results from output. Error: {e}")

    return results


def run_single_experiment(
    method: str, seed: int, config: Dict, output_dir: str, use_gpu: bool
) -> Dict[str, Any]:
    """
    Runs a single experiment trial and returns its results.

    Args:
        method: The continual learning method to use.
        seed: The random seed for the trial.
        config: A dictionary with experiment parameters.
        output_dir: The directory to save logs and models to.
        use_gpu: Boolean flag to use GPU.

    Returns:
        A dictionary containing the status and parsed results of the trial.
    """
    # Define the results path for this specific run
    run_results_path = os.path.join(
        output_dir, 'mammoth_outputs', f"{method}_seed_{seed}"
    )

    cmd = [
        sys.executable, 'main.py',
        '--dataset', 'seq-cifar10-224-custom',
        '--model', method,
        '--backbone', 'vit',
        '--results_path', run_results_path, # Tell Mammoth where to save its files
        '--seed', str(seed),
        '--n_epochs', str(config['epochs_per_task']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--optimizer', config['optimizer'],
        '--optim_wd', '0.0',
        '--optim_mom', '0.0',
        '--num_workers', str(config['num_workers']),
        '--drop_last', '0',
        '--debug_mode', '0',
    ]

    if not use_gpu:
        cmd.extend(['--device', 'cpu'])

    if method == 'derpp':
        cmd.extend(['--buffer_size', '200', '--alpha', '0.1', '--beta', '0.5'])
    elif method == 'ewc_on':
        cmd.extend(['--e_lambda', '0.4', '--gamma', '0.85'])

    timeout_seconds = 1800 * config['epochs_per_task'] # 30 mins per epoch

    try:
        # Use Popen to stream output in real-time for a better user experience
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        
        full_output = []
        # The tqdm bar for the subprocess itself
        with tqdm(total=1, desc=f"  Running {method} (seed {seed})", leave=False, bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            for line in iter(process.stdout.readline, ''):
                # Print the line to show live progress from main.py
                sys.stdout.write(line)
                full_output.append(line)
            process.wait(timeout=timeout_seconds)
            pbar.update(1) # Mark as complete

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, output=''.join(full_output))

        parsed_data = parse_results(''.join(full_output))
        return {'status': 'success', 'results': parsed_data}

    except subprocess.CalledProcessError as e:
        print(f"\nExperiment failed with error code {e.returncode}.")
        return {'status': 'failed', 'error': f"Exit code {e.returncode}"}
    except subprocess.TimeoutExpired:
        print(f"\nExperiment timed out after {timeout_seconds} seconds.")
        return {'status': 'timeout', 'error': f"Timeout after {timeout_seconds}s"}
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return {'status': 'crash', 'error': str(e)}


def run_full_comparison(config: Dict, output_dir: str, use_gpu: bool):
    """
    Runs the full comparison experiment across all methods and seeds.
    """
    print("=" * 60)
    print("Starting Shortcut Feature Investigation Experiment")
    print(f"Results will be saved in: {output_dir}")
    print("=" * 60)

    all_results = {
        'experiment_info': {
            'name': 'shortcut_investigation',
            'timestamp': datetime.now().isoformat(),
            'config': config
        },
        'runs': []
    }

    # Outer progress bar for methods
    for method in tqdm(config['methods'], desc="Overall Progress"):
        # Inner progress bar for seeds
        for seed in tqdm(config['seeds'], desc=f"  {method} seeds", leave=False):
            run_data = {
                'method': method,
                'seed': seed,
            }
            trial_result = run_single_experiment(method, seed, config, output_dir, use_gpu)
            run_data.update(trial_result)
            all_results['runs'].append(run_data)

            # Save results incrementally after each run
            with open(os.path.join(output_dir, 'experiment_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Use pandas to create a clean summary table
    summary_data = []
    for run in all_results['runs']:
        row = {
            'Method': run['method'],
            'Seed': run['seed'],
            'Status': run['status'],
            'Final CIL Acc (%)': run.get('results', {}).get('final_acc_cil', 'N/A'),
            'Final TIL Acc (%)': run.get('results', {}).get('final_acc_til', 'N/A'),
        }
        summary_data.append(row)
    
    if not summary_data:
        print("No runs were completed to summarize.")
        return

    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))

    # Calculate and print aggregate statistics
    print("\n--- Aggregate Statistics (Successful Runs) ---")
    success_df = df[df['Status'] == 'success'].copy()
    # Ensure numeric conversion for aggregation
    success_df['Final CIL Acc (%)'] = pd.to_numeric(success_df['Final CIL Acc (%)'], errors='coerce')
    success_df['Final TIL Acc (%)'] = pd.to_numeric(success_df['Final TIL Acc (%)'], errors='coerce')

    agg_df = success_df.groupby('Method').agg(
        mean_cil_acc=('Final CIL Acc (%)', 'mean'),
        std_cil_acc=('Final CIL Acc (%)', 'std'),
        mean_til_acc=('Final TIL Acc (%)', 'mean'),
        std_til_acc=('Final TIL Acc (%)', 'std'),
        runs=('Seed', 'count')
    ).reset_index()
    
    # Format for printing
    agg_df['mean_cil_acc'] = agg_df['mean_cil_acc'].map('{:.2f}'.format)
    agg_df['std_cil_acc'] = agg_df['std_cil_acc'].map('{:.2f}'.format)
    agg_df['mean_til_acc'] = agg_df['mean_til_acc'].map('{:.2f}'.format)
    agg_df['std_til_acc'] = agg_df['std_til_acc'].map('{:.2f}'.format)
    
    print(agg_df.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"Full results saved to: {os.path.join(output_dir, 'experiment_results.json')}")
    print("Next steps: Analyze the saved models and attention maps.")
    print("=" * 60)


def main():
    """Main function to parse arguments and start the experiment."""
    parser = argparse.ArgumentParser(description='Run Shortcut Investigation Experiment Manager')
    parser.add_argument('--epochs', type=int, default=EXPERIMENT_CONFIG['epochs_per_task'],
                        help='Number of epochs per task.')
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force CPU execution for the experiment run.')
    parser.add_argument('--workers', type=int, default=EXPERIMENT_CONFIG['num_workers'],
                        help='Number of data loading workers.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results. A timestamped subdir will be created.')
    
    args = parser.parse_args()

    # Update config with command-line arguments
    config = EXPERIMENT_CONFIG.copy()
    config['epochs_per_task'] = args.epochs
    config['num_workers'] = args.workers

    # Create a unique directory for this experiment run
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.path.join(os.path.dirname(__file__), 'results')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_output_dir, f"shortcut_exp_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    # Create subdir for mammoth's own outputs
    os.makedirs(os.path.join(experiment_dir, 'mammoth_outputs'), exist_ok=True)

    use_gpu_flag = not args.cpu_only
    run_full_comparison(config, experiment_dir, use_gpu_flag)


if __name__ == "__main__":
    main()