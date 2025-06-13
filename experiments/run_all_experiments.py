"""
PHASE 1: EXPERIMENT RUNNER

This script's only job is to run all training experiments for the shortcut
investigation. It uses subprocess to call Mammoth's main.py, ensuring
robustness and correct argument parsing.

It generates:
1. Model checkpoints in the `checkpoints/` directory.
2. A manifest file (`experiment_manifest.json`) in the results directory,
   which lists all runs and their parameters for the analysis script.
"""
import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
import torch

# Add mammoth path to system path
mammoth_path = os.path.dirname(os.path.abspath(__file__))
if mammoth_path not in sys.path:
    sys.path.insert(0, mammoth_path)

# --- Experiment Configuration ---
BASE_ARGS = {
    "dataset": "seq-cifar10-224-custom",
    "backbone": "vit",
    "n_epochs": 10,
    "batch_size": 32,
    "lr": 0.001,
    "base_path": "./data/",
}
METHODS = ["derpp", "ewc_on"]
SEEDS = [42, 123]
# --- End Configuration ---


def run_single_experiment(
    method: str, seed: int, results_dir: str, device: str
) -> Dict:
    """Runs a single experiment trial using subprocess."""
    print(f"\n--- Running Method: {method}, Seed: {seed} ---")

    # Define a unique, predictable prefix for this run's checkpoints
    ckpt_name = f"shortcut_exp_{method}_seed_{seed}"

    # Define where Mammoth should save its own log files (if any)
    mammoth_results_path = os.path.join(results_dir, f"{method}_seed_{seed}")
    os.makedirs(mammoth_results_path, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(mammoth_path, "main.py"),
        "--dataset", BASE_ARGS["dataset"],
        "--model", method,
        "--backbone", BASE_ARGS["backbone"],
        "--seed", str(seed),
        "--n_epochs", str(BASE_ARGS["n_epochs"]),
        "--batch_size", str(BASE_ARGS["batch_size"]),
        "--lr", str(BASE_ARGS["lr"]),
        "--device", device,
        "--base_path", BASE_ARGS["base_path"],
        "--results_path", mammoth_results_path,
        "--num_workers", "0",
        "--savecheck", "task",
        "--nowand",
        "--ckpt_name", ckpt_name,
    ]

    if method == "derpp":
        cmd.extend(["--buffer_size", "200", "--alpha", "0.1", "--beta", "0.5"])
    elif method == "ewc_on":
        cmd.extend(["--e_lambda", "1000", "--gamma", "1.0"])

    print(f"Executing command: {' '.join(cmd)}")
    log_file_path = os.path.join(mammoth_results_path, "training_log.txt")

    try:
        with open(log_file_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=mammoth_path,
            )
            for line in iter(process.stdout.readline, ""):
                sys.stdout.write(line)
                log_file.write(line)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        print(f"✓ Training successful. Log saved to {log_file_path}")
        return {"status": "success", "ckpt_name": ckpt_name}

    except Exception as e:
        print(f"✗ Training failed for {method} seed {seed}. Error: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Runs all configured experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"shortcut_investigation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Starting experiment run. Results will be stored in: {results_dir}")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    experiment_manifest = {
        "experiment_name": f"shortcut_investigation_{timestamp}",
        "results_dir": results_dir,
        "base_args": BASE_ARGS,
        "runs": [],
    }

    for method in tqdm(METHODS, desc="Overall Progress"):
        for seed in SEEDS:
            run_result = run_single_experiment(method, seed, results_dir, device)
            run_info = {"method": method, "seed": seed, **run_result}
            experiment_manifest["runs"].append(run_info)

            # Save manifest incrementally
            manifest_path = os.path.join(results_dir, "experiment_manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(experiment_manifest, f, indent=4)

    print("\n" + "=" * 50)
    print("All training runs complete.")
    print(f"Manifest file saved to: {manifest_path}")
    print("You can now run the analysis script:")
    print(f"python analyze_all_results.py --results_dir {results_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()