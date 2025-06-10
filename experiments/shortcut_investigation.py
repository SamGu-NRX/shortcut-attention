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
        
        # Create results directory for this specific run
        results_path = os.path.join(self.results_dir, f"{method}_seed_{seed}")
        os.makedirs(results_path, exist_ok=True)

        args_dict = self.base_args.copy()
        args_dict.update(
            {
                "model": method,
                "seed": seed,
                "ckpt_name": ckpt_name,
                "results_path": results_path,  # Set the results path
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
            return results_path  # Return the results directory
        except Exception as e:
            self.logger.error(
                f"Error running {method} with seed {seed}: {e}",
                exc_info=True,
            )
            return None

    def _analyze_checkpoints(self, method: str, seed: int, results_path: str):
        """Load each task's checkpoint and run analysis."""
        # Look for checkpoints in the results directory or subdirectories
        checkpoint_paths = []
        
        # Check for checkpoints in main directory
        if os.path.exists(results_path):
            for file in os.listdir(results_path):
                if file.endswith('.pth') or file.endswith('_last'):
                    checkpoint_paths.append(os.path.join(results_path, file))
        
        # Check for checkpoints in 'checkpoints' subdirectory
        checkpoints_dir = os.path.join(results_path, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            for file in os.listdir(checkpoints_dir):
                if file.endswith('.pth') or file.endswith('_last'):
                    checkpoint_paths.append(os.path.join(checkpoints_dir, file))
        
        if not checkpoint_paths:
            self.logger.warning(f"No checkpoints found in {results_path}")
            return

        # For now, just analyze the final checkpoint
        # (Mammoth typically saves the final model, not per-task checkpoints)
        final_checkpoint = checkpoint_paths[0]  # Use the first (and likely only) checkpoint
        
        self.logger.info(f"--- Analyzing final checkpoint: {os.path.basename(final_checkpoint)} ---")

        # Prepare a base model and dataset instance
        args_dict = self.base_args.copy()
        args_dict.update({"model": method, "seed": seed})
        if method == "derpp":
            args_dict.update({"buffer_size": 200})
        args = argparse.Namespace(**args_dict)

        try:
            dataset = get_dataset(args)
            backbone = get_backbone(args)
            loss = dataset.get_loss()
            transform = dataset.get_transform()
            model = get_model(args, backbone, loss, transform, dataset)
            model.to(args.device if args.device != "cpu" else "cpu")

            # Load the state dict from the checkpoint
            state_dict = torch.load(final_checkpoint, map_location=args.device if args.device != "cpu" else "cpu")
            
            # Handle different checkpoint formats
            if isinstance(state_dict, dict) and 'net' in state_dict:
                model.load_state_dict(state_dict['net'])
            elif isinstance(state_dict, dict) and 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
            else:
                model.load_state_dict(state_dict)

            # Create analysis directory
            analysis_dir = os.path.join(
                self.results_dir, method, f"seed_{seed}", "final_analysis"
            )
            os.makedirs(analysis_dir, exist_ok=True)

            # For simplicity, analyze both tasks with the final model
            for task_id in range(dataset.N_TASKS):
                self.logger.info(f"Analyzing performance on Task {task_id}")
                
                # Set the current task for the dataset
                if hasattr(dataset, 'set_task'):
                    dataset.set_task(task_id)
                
                task_analysis_dir = os.path.join(analysis_dir, f"task_{task_id}")

                # 1. Attention Analysis
                analyze_task_attention(
                    model,
                    dataset,
                    device=args.device if args.device != "cpu" else "cpu",
                    save_dir=os.path.join(task_analysis_dir, "attention"),
                )

                # 2. Activation Analysis
                extractor = ActivationExtractor(model, device=args.device if args.device != "cpu" else "cpu")
                _, test_loader = dataset.get_data_loaders()
                sample_batch = next(iter(test_loader))[0][:5]  # 5 samples
                activations = extractor.extract_activations(sample_batch)
                visualize_activations(
                    activations,
                    save_path=os.path.join(task_analysis_dir, "activations.png"),
                )
                extractor.remove_hooks()
                
        except Exception as e:
            self.logger.error(f"Error in checkpoint analysis: {e}", exc_info=True)


def main():
    """Main function to configure and run the experiment."""
    # This dictionary now contains a comprehensive set of default arguments
    # that the Mammoth framework expects.
    base_args = {
        # ==========================================
        # Core Experiment Settings (What we care about)
        # ==========================================
        "dataset": "seq-cifar10-224-custom",
        "model": "derpp",  # Will be overridden
        "backbone": "vit",
        "n_epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "device": "0" if torch.cuda.is_available() else "cpu",

        # ==========================================
        # Required Mammoth Arguments (To prevent AttributeErrors)
        # ==========================================
        "base_path": "./data/",
        "results_path": None,  # Will be set per-run
        "savecheck": 1,
        "ckpt_name": None,  # Will be set per-run
        "debug_mode": 0,
        "non_verbose": 1,
        "code_optimization": 0,

        # --- Logging and Reporting ---
        "nowand": 1,  # Disables Weights & Biases
        "wandb_entity": None,
        "wandb_project": None,
        "tensorboard": False,
        "csv_log": False,
        "notes": None,

        # --- Dataset & Tasking ---
        "seed": 42,  # Will be overridden
        "start_from": None,
        "stop_after": None,
        "joint": False,
        "label_perc": 1.0,
        "label_perc_by_class": 1.0,
        "validation": None,
        "validation_mode": "current",
        "noise_rate": 0.0,
        "transform_type": "weak",
        "custom_class_order": None,
        "permute_classes": False,
        "custom_task_order": None,
        "drop_last": False,
        "num_workers": 0,

        # --- Model & Training ---
        "fitting_mode": "epochs",
        "n_iters": None,
        "minibatch_size": 32, # Set to batch_size by default
        "optimizer": "sgd",
        "optim_wd": 0.0,
        "optim_mom": 0.0,
        "optim_nesterov": False,
        "lr_scheduler": None,
        "distributed": "no",
        "load_best_args": False,
        "inference_only": 0,

        # --- Metrics ---
        "enable_other_metrics": False,
        "eval_future": False,
        "ignore_other_metrics": 0,

        # --- Config Files (Not used in this setup) ---
        "dataset_config": None,
        "model_config": None,

        # --- ViT Specific (Passed to backbone) ---
        "pretrained": False,
        "pretrain_type": "in21k-ft-in1k",
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
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