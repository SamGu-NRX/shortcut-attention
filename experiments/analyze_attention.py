# experiments/analyze_attention.py

"""
Experiment Analysis Script for Shortcut Feature Investigation.
"""

import argparse
import logging
import os
import sys
from typing import Dict

import torch

# Add mammoth path to system path
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if mammoth_path not in sys.path:
    sys.path.insert(0, mammoth_path)

from backbone import get_backbone
from datasets import get_dataset
from models import get_model
from utils.attention_visualization import analyze_task_attention
from utils.network_flow_visualization import (
    ActivationExtractor,
    visualize_activations,
)
# Import the central configuration
from experiments.default_args import get_base_args


class AnalysisRunner:
    """Orchestrates the analysis phase of the experiment."""

    def __init__(self, experiment_dir: str, base_args: Dict):
        self.experiment_dir = experiment_dir
        self.base_args = base_args
        if not os.path.exists(experiment_dir):
            raise FileNotFoundError(
                f"Experiment directory not found: {experiment_dir}"
            )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(self.experiment_dir, "analysis.log")
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Analyzing results from: {self.experiment_dir}")

    def run_full_analysis(self):
        """Finds all experiment runs and analyzes their checkpoints."""
        run_dirs = [
            d
            for d in os.listdir(self.experiment_dir)
            if os.path.isdir(os.path.join(self.experiment_dir, d))
        ]
        for run_dir in run_dirs:
            try:
                method, _, seed_str = run_dir.partition("_seed_")
                seed = int(seed_str)
                self.logger.info(
                    f"\n{'='*20} Analyzing Method: {method}, Seed: {seed} {'='*20}"
                )
                ckpt_name_prefix = f"shortcut_exp_{method}_seed_{seed}"
                self._analyze_checkpoints(method, seed, ckpt_name_prefix)
            except (ValueError, IndexError):
                self.logger.warning(
                    f"Could not parse method/seed from directory '{run_dir}'. Skipping."
                )

    def _analyze_checkpoints(
        self, method: str, seed: int, ckpt_name_prefix: str
    ):
        """Loads the MOST RECENT checkpoints for a single run and generates visualizations."""
        global_checkpoint_dir = os.path.join(mammoth_path, "checkpoints")
        if not os.path.exists(global_checkpoint_dir):
            self.logger.error(
                f"Global checkpoint directory not found: {global_checkpoint_dir}"
            )
            return

        all_run_files = [
            f
            for f in os.listdir(global_checkpoint_dir)
            if f.startswith(ckpt_name_prefix) and f.endswith(".pt")
        ]
        if not all_run_files:
            self.logger.warning(
                f"No checkpoints with prefix '{ckpt_name_prefix}' found."
            )
            return

        runs = {}
        for f in all_run_files:
            parts = f.split("_")
            run_id = f"{parts[-3]}_{parts[-2]}"
            if run_id not in runs:
                runs[run_id] = []
            runs[run_id].append(f)

        latest_run_id = sorted(runs.keys())[-1]
        checkpoints_to_analyze = sorted(runs[latest_run_id])

        self.logger.info(
            f"Found {len(runs)} run(s). Analyzing latest run '{latest_run_id}'."
        )
        self.logger.info(
            f"Checkpoints for analysis: {checkpoints_to_analyze}"
        )

        args_dict = self.base_args.copy()
        args_dict.update({"model": method, "seed": seed})
        if method == "derpp":
            args_dict.update({"buffer_size": 200})
        args = argparse.Namespace(**args_dict)

        try:
            dataset_factory = get_dataset(args)
            backbone = get_backbone(args)
            loss = dataset_factory.get_loss()
            transform = dataset_factory.get_transform()
            model = get_model(args, backbone, loss, transform, dataset_factory)
            model.to(args.device)

            # ======================================================================
            # THE FINAL FIX: Generate all task loaders sequentially, as intended
            # by the Mammoth framework's design.
            # ======================================================================
            self.logger.info("Generating all task data loaders sequentially...")
            all_task_test_loaders = []
            for _ in range(dataset_factory.N_TASKS):
                # Each call to get_data_loaders advances the internal task counter
                _, test_loader = dataset_factory.get_data_loaders()
                all_task_test_loaders.append(test_loader)
            self.logger.info("Data loader generation complete.")
            # ======================================================================

            for ckpt_file in checkpoints_to_analyze:
                task_id = int(os.path.splitext(ckpt_file)[0].split("_")[-1])
                self.logger.info(
                    f"--- Analyzing checkpoint: {ckpt_file} (Task {task_id}) ---"
                )

                checkpoint_path = os.path.join(
                    global_checkpoint_dir, ckpt_file
                )
                state_dict = torch.load(
                    checkpoint_path, map_location=args.device
                )

                if isinstance(state_dict, dict) and "model" in state_dict:
                    model_state_dict = state_dict["model"]
                    model.load_state_dict(model_state_dict)
                else:
                    self.logger.warning(
                        "Checkpoint does not contain a 'model' key. Assuming the file is the state_dict itself."
                    )
                    model.load_state_dict(state_dict)

                analysis_dir = os.path.join(
                    self.experiment_dir,
                    f"{method}_seed_{seed}",
                    "analysis",
                    f"after_task_{task_id}",
                )
                os.makedirs(analysis_dir, exist_ok=True)

                for past_task_id in range(task_id + 1):
                    self.logger.info(
                        f"  > Visualizing attention for Task {past_task_id} samples"
                    )
                    # Retrieve the correct, pre-generated loader for the task.
                    current_test_loader = all_task_test_loaders[past_task_id]

                    task_analysis_dir = os.path.join(
                        analysis_dir, f"analyzing_task_{past_task_id}"
                    )
                    os.makedirs(task_analysis_dir, exist_ok=True)

                    analyze_task_attention(
                        model,
                        current_test_loader.dataset,
                        device=args.device,
                        save_dir=os.path.join(
                            task_analysis_dir, "attention"
                        ),
                    )

                    extractor = ActivationExtractor(model, device=args.device)
                    sample_batch = next(iter(current_test_loader))[0][:5]
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
    """Parses arguments and starts the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze attention experiment results."
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to the experiment results directory created by run_training.py",
    )
    cli_args = parser.parse_args()

    # Get the complete, consistent base arguments
    base_args = get_base_args()

    analyzer = AnalysisRunner(
        experiment_dir=cli_args.experiment_dir, base_args=base_args
    )
    analyzer.run_full_analysis()

    print("\nAnalysis complete!")
    print(f"Visualizations saved within: {cli_args.experiment_dir}")


if __name__ == "__main__":
    main()