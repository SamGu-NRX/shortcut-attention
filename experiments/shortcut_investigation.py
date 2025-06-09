# experiments/shortcut_investigation.py

"""
Main experiment script for investigating shortcut features in continual learning.
This script runs experiments and then performs post-hoc analysis on the
checkpoints saved after each task.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import torch

# Add mammoth path
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, mammoth_path)

from backbone import get_backbone
from datasets import get_dataset
from main import main as mammoth_main
from models import get_model
from utils.conf import set_random_seed
from utils.attention_visualization import (
    analyze_task_attention,
)
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
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized experiment: {self.experiment_name}")

    def run_and_analyze(
        self, methods: List[str], seeds: List[int]
    ) -> None:
        """Run experiments and perform analysis for all methods and seeds."""
        for method in methods:
            for seed in seeds:
                self.logger.info(
                    f"\n{'='*20} Running Method: {method}, Seed: {seed} {'='*20}"
                )
                # 1. Run the full continual learning experiment
                model_dir = self._run_single_experiment(method, seed)
                if not model_dir:
                    self.logger.error("Experiment run failed. Skipping analysis.")
                    continue

                # 2. Perform post-hoc analysis on each task's checkpoint
                self.logger.info(
                    f"\n{'='*20} Analyzing Method: {method}, Seed: {seed} {'='*20}"
                )
                self._analyze_checkpoints(method, seed, model_dir)

    def _run_single_experiment(self, method: str, seed: int) -> str:
        """Run a single experiment and return the path to the model directory."""
        self.logger.info(f"Running Mammoth for {method} with seed {seed}")
        set_random_seed(seed)

        # Define a unique name for the checkpoint to easily find it later
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_name = f"{method}_seed{seed}_{now}"
        model_dir = os.path.join("checkpoints", ckpt_name)

        args_dict = self.base_args.copy()
        args_dict.update(
            {
                "model": method,
                "seed": seed,
                "ckpt_name": ckpt_name,
                "savecheck": 1,  # IMPORTANT: Enable saving checkpoints
            }
        )

        # Method-specific hyperparameters
        if method == "derpp":
            args_dict.update({"buffer_size": 200, "alpha": 0.1, "beta": 0.5})
        elif method == "ewc_on":
            args_dict.update({"e_lambda": 1000, "gamma": 1.0})

        args = argparse.Namespace(**args_dict)

        try:
            mammoth_main(args)
            return model_dir
        except Exception as e:
            self.logger.error(
                f"Error running {method} with seed {seed}: {e}",
                exc_info=True,
            )
            return None

    def _analyze_checkpoints(self, method: str, seed: int, model_dir: str):
        """Load each task's checkpoint and run analysis."""
        # Find all task checkpoints
        checkpoints = sorted(
            [f for f in os.listdir(model_dir) if f.startswith("task_")]
        )
        if not checkpoints:
            self.logger.warning(f"No task checkpoints found in {model_dir}")
            return

        # Prepare a base model and dataset instance
        args_dict = self.base_args.copy()
        args_dict.update({"model": method, "seed": seed})
        if method == "derpp":
            args_dict.update({"buffer_size": 200})
        args = argparse.Namespace(**args_dict)

        dataset = get_dataset(args)
        backbone = get_backbone(args)
        loss = dataset.get_loss()
        transform = dataset.get_transform()
        model = get_model(args, backbone, loss, transform, dataset)
        model.to(args.device)

        for ckpt_file in checkpoints:
            task_id = int(ckpt_file.split("_")[1])
            self.logger.info(f"--- Analyzing checkpoint: {ckpt_file} ---")

            # Load the state dict from the checkpoint
            checkpoint_path = os.path.join(model_dir, ckpt_file)
            state_dict = torch.load(checkpoint_path, map_location=args.device)
            model.load_state_dict(state_dict["net"])

            # Create analysis directory
            analysis_dir = os.path.join(
                self.results_dir, method, f"seed_{seed}", f"task_{task_id}"
            )
            os.makedirs(analysis_dir, exist_ok=True)

            # Analyze all tasks seen so far
            for past_task_id in range(task_id + 1):
                self.logger.info(
                    f"Analyzing performance on Task {past_task_id}"
                )
                dataset.set_task(past_task_id)
                task_analysis_dir = os.path.join(
                    analysis_dir, f"analyzing_task_{past_task_id}"
                )

                # 1. Attention Analysis
                analyze_task_attention(
                    model,
                    dataset,
                    device=args.device,
                    save_dir=os.path.join(task_analysis_dir, "attention"),
                )

                # 2. Activation Analysis
                extractor = ActivationExtractor(model, device=args.device)
                _, test_loader = dataset.get_data_loaders()
                sample_batch = next(iter(test_loader))[0][:5]  # 5 samples
                activations = extractor.extract_activations(sample_batch)
                visualize_activations(
                    activations,
                    save_path=os.path.join(
                        task_analysis_dir, "activations.png"
                    ),
                )
        # Reset dataset to its original state if needed
        dataset.set_task(0)


def main():
    """Main function to run the shortcut investigation experiment."""
    base_args = {
        "dataset": "seq-cifar10-224-custom",
        "backbone": "vit",
        "n_epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "nowand": 1,
        "non_verbose": 1,
        "debug_mode": 0,
    }

    experiment = ShortcutInvestigationExperiment(
        base_args=base_args,
        experiment_name=f"shortcut_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    experiment.run_and_analyze(
        methods=["derpp", "ewc_on"], seeds=[42, 123]
    )

    print("Experiment completed successfully!")
    print(f"Results and analysis saved in: {experiment.results_dir}")


if __name__ == "__main__":
    main()