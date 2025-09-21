# experiments/run_training.py

"""
Experiment Training Runner for Shortcut Feature Investigation.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List

import torch
from tqdm import tqdm

# Add mammoth path to system path
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if mammoth_path not in sys.path:
    sys.path.insert(0, mammoth_path)

# Import the central configuration
from experiments.default_args import get_base_args


class TrainingRunner:
    """Orchestrates the training phase of the experiment."""

    def __init__(
        self, base_args: Dict, experiment_name: str = "shortcut_investigation"
    ):
        self.base_args = base_args
        self.experiment_name = experiment_name
        self.results_dir = os.path.join("results", self.experiment_name)
        os.makedirs(self.results_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(self.results_dir, "training_runner.log")
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Training Run: {self.experiment_name}")
        self.logger.info(
            f"Using device: {self.base_args['device']} "
            f"({torch.cuda.get_device_name(0) if self.base_args['device'] != 'cpu' else 'CPU'})"
        )

    def run_all_methods(self, methods: List[str], seeds: List[int]) -> None:
        """Runs the training for all specified methods and seeds."""
        for method in tqdm(methods, desc="Overall Progress"):
            for seed in tqdm(seeds, desc=f"  Method: {method}", leave=False):
                self.logger.info(
                    f"\n{'='*20} Training Method: {method}, Seed: {seed} {'='*20}"
                )
                self._run_single_experiment(method, seed)

    def _run_single_experiment(self, method: str, seed: int) -> None:
        """Executes a single training trial using a subprocess."""
        run_results_path = os.path.join(self.results_dir, f"{method}_seed_{seed}")
        os.makedirs(run_results_path, exist_ok=True)

        ckpt_name = f"shortcut_exp_{method}_seed_{seed}"

        cmd = [
            sys.executable,
            os.path.join(mammoth_path, "main.py"),
            "--dataset", self.base_args["dataset"],
            "--model", method,
            "--backbone", self.base_args["backbone"],
            "--seed", str(seed),
            "--n_epochs", str(self.base_args["n_epochs"]),
            "--batch_size", str(self.base_args["batch_size"]),
            "--lr", str(self.base_args["lr"]),
            "--device", self.base_args["device"],
            "--base_path", self.base_args["base_path"],
            "--results_path", run_results_path,
            "--num_workers", str(self.base_args["num_workers"]),
            "--savecheck", self.base_args["savecheck"],
            "--ckpt_name", ckpt_name,
        ]

        if method == "derpp":
            cmd.extend(["--buffer_size", "200", "--alpha", "0.1", "--beta", "0.5"])
        elif method == "ewc_on":
            cmd.extend(["--e_lambda", "1000", "--gamma", "1.0"])

        try:
            self.logger.info(f"Executing command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=mammoth_path,
            )

            output_log = []
            print("-" * 60 + f"\nLive output for {method} (seed {seed}):")
            for line in iter(process.stdout.readline, ""):
                if any(kw in line for kw in ["Epoch", "Task", "Accuracy", "Loss"]):
                    print(line.strip())
                output_log.append(line)
            print("-" * 60)

            process.wait()
            with open(os.path.join(run_results_path, "training_log.txt"), "w") as f:
                f.writelines(output_log)

            if process.returncode != 0:
                self.logger.error(
                    f"Training failed for {method} seed {seed}. See log for details."
                )
            else:
                self.logger.info(
                    f"Training completed successfully for {method} seed {seed}."
                )

        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}", exc_info=True)


def main():
    """Configures and starts the training runs."""
    # Get the complete, consistent base arguments
    base_args = get_base_args()

    runner = TrainingRunner(
        base_args=base_args,
        experiment_name=f"shortcut_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    runner.run_all_methods(methods=["derpp", "ewc_on"], seeds=[42, 123])

    print("\nAll training runs are complete!")
    print(f"Results and checkpoints saved in: {runner.results_dir}")
    print(f"To analyze, run: python experiments/analyze_attention.py {runner.results_dir}")


if __name__ == "__main__":
    main()