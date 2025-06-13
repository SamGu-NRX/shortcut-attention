# deprecated

# experiments/shortcut_investigation.py

"""
Main experiment script for investigating shortcut features in continual learning.
This script runs experiments via subprocess with real-time progress tracking
and then performs post-hoc analysis on the saved checkpoints.
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from datetime import datetime
from typing import Dict, List

import torch
from tqdm import tqdm

# Add mammoth path to system path
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if mammoth_path not in sys.path:
    sys.path.insert(0, mammoth_path)

# Imports for the analysis part, placed here to ensure path is set
from backbone import get_backbone
from datasets import get_dataset
from models import get_model
from utils.conf import set_random_seed
from utils.attention_visualization import analyze_task_attention
from utils.network_flow_visualization import (
    ActivationExtractor,
    visualize_activations,
)


class ShortcutInvestigationExperiment:
    """Main class for running shortcut feature investigation experiments."""

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
                    os.path.join(self.results_dir, "experiment.log")
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized experiment: {self.experiment_name}")
        self.logger.info(
            f"Using device: {self.base_args['device']} "
            f"({torch.cuda.get_device_name(0) if self.base_args['device'] != 'cpu' else 'CPU'})"
        )

    def run_and_analyze(
        self, methods: List[str], seeds: List[int]
    ) -> None:
        """Run experiments and perform analysis for all methods and seeds."""
        # Use tqdm for an overall progress bar
        for method in tqdm(methods, desc="Overall Progress"):
            for seed in tqdm(seeds, desc=f"  Method: {method}", leave=False):
                self.logger.info(
                    f"\n{'='*20} Running Method: {method}, Seed: {seed} {'='*20}"
                )
                # 1. Run the full continual learning experiment
                results_path = self._run_single_experiment(method, seed)
                if not results_path:
                    self.logger.error(
                        "Experiment run failed. Skipping analysis."
                    )
                    continue

                # 2. Perform post-hoc analysis on the saved checkpoint
                self.logger.info(
                    f"\n{'='*20} Analyzing Method: {method}, Seed: {seed} {'='*20}"
                )
                self._analyze_checkpoints(method, seed, results_path)

    def _run_single_experiment(self, method: str, seed: int) -> str:
        """Run a single experiment using subprocess with real-time output."""
        self.logger.info(
            f"Running Mammoth for {method} with seed {seed} via subprocess."
        )

        results_path = os.path.join(self.results_dir, f"{method}_seed_{seed}")
        os.makedirs(results_path, exist_ok=True)

        # Create a predictable name for the checkpoint directory
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
            "--results_path", results_path,
            "--num_workers", "0",
            # --- Corrected Arguments ---
            "--savecheck", "task",      # FIX 1: Use 'task' instead of '1'
            "--ckpt_name", ckpt_name,   # FIX 3: Provide an explicit checkpoint name
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
            print("-" * 60)
            print(f"Live output for {method} (seed {seed}):")
            for line in iter(process.stdout.readline, ""):
                if any(
                    keyword in line
                    for keyword in ["Epoch", "Task", "Accuracy", "Loss", "Progress"]
                ):
                    print(line.strip())
                output_log.append(line)
            print("-" * 60)

            process.wait()
            
            with open(os.path.join(results_path, "training_log.txt"), "w") as f:
                f.writelines(output_log)

            if process.returncode != 0:
                self.logger.error(
                    f"Command failed with return code {process.returncode}. "
                    f"Check 'training_log.txt' in {results_path} for details."
                )
                return None

            self.logger.info("Mammoth training completed successfully.")
            # Return the predictable checkpoint name for the analysis function
            return ckpt_name

        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while running subprocess: {e}",
                exc_info=True,
            )
            return None

    def _analyze_checkpoints(self, method: str, seed: int, ckpt_name: str):
        """Load each task's checkpoint from the predictable path and run analysis."""
        # Mammoth saves all checkpoints in a single top-level 'checkpoints' directory
        global_checkpoint_dir = os.path.join(mammoth_path, "checkpoints", ckpt_name)

        if not os.path.exists(global_checkpoint_dir):
            self.logger.warning(
                f"Checkpoint directory not found: {global_checkpoint_dir}. Analysis cannot proceed."
            )
            return

        checkpoints = sorted(
            [
                f
                for f in os.listdir(global_checkpoint_dir)
                if f.startswith("task_") and f.endswith((".pth", ".pt"))
            ]
        )

        if not checkpoints:
            self.logger.warning(
                f"No per-task checkpoints found in {global_checkpoint_dir}. Analysis cannot proceed."
            )
            return

        self.logger.info(f"Found checkpoints for analysis: {checkpoints}")

        # ======================================================================
        # THE FIX: Create a complete Namespace for the analysis phase.
        # ======================================================================
        # Start with a complete copy of the base arguments
        args_dict = self.base_args.copy()
        # Update with the specific method and seed for this analysis run
        args_dict.update({"model": method, "seed": seed})
        # Add any method-specific args needed by the model constructor
        if method == "derpp":
            args_dict.update({"buffer_size": 200})
        # Convert the complete dictionary to a Namespace object
        args = argparse.Namespace(**args_dict)
        # ======================================================================

        try:
            # Now, get_dataset(args) will work because args.joint exists.
            dataset = get_dataset(args)
            backbone = get_backbone(args)
            loss = dataset.get_loss()
            transform = dataset.get_transform()
            model = get_model(args, backbone, loss, transform, dataset)
            model.to(args.device)

            for ckpt_file in checkpoints:
                task_id_str = os.path.splitext(ckpt_file)[0].split("_")[-1]
                task_id = int(task_id_str)
                
                self.logger.info(f"--- Analyzing checkpoint: {ckpt_file} (Task {task_id}) ---")

                checkpoint_path = os.path.join(global_checkpoint_dir, ckpt_file)
                state_dict = torch.load(checkpoint_path, map_location=args.device)
                model.load_state_dict(state_dict["net"])

                analysis_dir = os.path.join(
                    self.results_dir, f"{method}_seed_{seed}", "analysis", f"after_task_{task_id}"
                )
                os.makedirs(analysis_dir, exist_ok=True)

                for past_task_id in range(task_id + 1):
                    self.logger.info(
                        f"  > Visualizing attention for Task {past_task_id} samples"
                    )
                    dataset.set_task(past_task_id)
                    task_analysis_dir = os.path.join(
                        analysis_dir, f"analyzing_task_{past_task_id}"
                    )

                    analyze_task_attention(
                        model,
                        dataset,
                        device=args.device,
                        save_dir=os.path.join(task_analysis_dir, "attention"),
                    )

                    extractor = ActivationExtractor(model, device=args.device)
                    _, test_loader = dataset.get_data_loaders()
                    sample_batch = next(iter(test_loader))[0][:5]
                    activations = extractor.extract_activations(sample_batch)
                    visualize_activations(
                        activations,
                        save_path=os.path.join(
                            task_analysis_dir, "activations.png"
                        ),
                    )
                    extractor.remove_hooks()

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}", exc_info=True)


def main():
    """Main function to configure and run the experiment."""
    base_args = {
        "dataset": "seq-cifar10-224-custom",
        "backbone": "vit",
        "n_epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "device": "0" if torch.cuda.is_available() else "cpu",
        "base_path": "./data/",
    }

    experiment = ShortcutInvestigationExperiment(
        base_args=base_args,
        experiment_name=f"shortcut_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    experiment.run_and_analyze(methods=["derpp", "ewc_on"], seeds=[42, 123])

    print("\nExperiment completed successfully!")
    print(f"Results and analysis saved in: {experiment.results_dir}")


if __name__ == "__main__":
    main()