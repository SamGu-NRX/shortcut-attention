#!/usr/bin/env python3
# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Einstellung Effect Experiment Runner

This module orchestrates comprehensive Einstellung Effect experiments across
multiple continual learning strategies, training scenarios, and random seeds.

Scenarios:
- Sequential: Standard continual learning (T1 → T2)
- Scratch-T2: Train only on T2 with shortcuts
- Interleaved: Mixed training with T1 and T2 samples

Strategies:
- DER++: Dark Experience Replay with knowledge distillation
- EWC: Elastic Weight Consolidation
- SGD: Standard fine-tuning baseline

Analysis:
- Multi-seed statistical analysis
- Comprehensive ERI metrics
- Attention pattern analysis (ViT only)
- Comparative visualization
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Add Mammoth to path
sys.path.append(str(Path(__file__).parent.parent))

from main import main as mammoth_main
from utils.args import add_experiment_args, add_management_args
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.einstellung_evaluator import EinstellungEvaluator, create_einstellung_evaluator
from utils.einstellung_metrics import EinstellungMetrics, calculate_cross_experiment_eri_statistics


class EinstellungExperimentConfig:
    """Configuration for Einstellung Effect experiments."""

    def __init__(self):
        # Dataset configuration
        self.dataset = 'seq-cifar100-einstellung'
        self.backbone = 'resnet18'  # or 'vit_base_patch16_224'

        # Training scenarios
        self.scenarios = ['sequential', 'scratch_t2', 'interleaved']

        # Continual learning strategies
        self.strategies = ['sgd', 'ewc_on', 'derpp']

        # Random seeds for statistical analysis
        self.seeds = [42, 123, 456, 789, 1011]

        # Training configuration
        self.n_epochs = 50
        self.batch_size = 32
        self.lr = 0.01

        # Einstellung-specific parameters
        self.patch_size = 4
        self.patch_color = [255, 0, 255]  # Magenta
        self.adaptation_threshold = 0.8

        # Output configuration
        self.base_output_dir = './einstellung_results'
        self.save_checkpoints = True
        self.export_attention = True


class EinstellungScenarioManager:
    """Manages different training scenarios for Einstellung experiments."""

    def __init__(self, config: EinstellungExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_scenario_args(self,
                           scenario: str,
                           strategy: str,
                           seed: int,
                           output_dir: str) -> argparse.Namespace:
        """
        Create command line arguments for a specific scenario.

        Args:
            scenario: Training scenario ('sequential', 'scratch_t2', 'interleaved')
            strategy: Continual learning strategy
            seed: Random seed
            output_dir: Output directory for results

        Returns:
            Namespace with experiment arguments
        """
        # Base arguments
        args = argparse.Namespace()

        # Dataset and model
        args.dataset = self.config.dataset
        args.model = strategy
        args.backbone = self.config.backbone

        # Training configuration
        args.n_epochs = self.config.n_epochs
        args.batch_size = self.config.batch_size
        args.lr = self.config.lr
        args.seed = seed

        # Einstellung parameters
        args.einstellung_patch_size = self.config.patch_size
        args.einstellung_patch_color = self.config.patch_color
        args.einstellung_adaptation_threshold = self.config.adaptation_threshold
        args.einstellung_evaluation_subsets = True
        args.einstellung_extract_attention = self.config.export_attention

        # Scenario-specific configurations
        if scenario == 'sequential':
            # Standard continual learning: T1 → T2
            args.einstellung_apply_shortcut = False  # Start without shortcuts
            args.scenario_type = 'sequential'

        elif scenario == 'scratch_t2':
            # Train only on T2 with shortcuts
            args.einstellung_apply_shortcut = True
            args.scenario_type = 'scratch_t2'
            # Additional args to skip T1
            args.start_from_task = 1

        elif scenario == 'interleaved':
            # Mixed training with T1 and T2
            args.einstellung_apply_shortcut = True
            args.scenario_type = 'interleaved'
            # Additional interleaving configuration would go here

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Strategy-specific parameters
        if strategy == 'derpp':
            args.buffer_size = 500
            args.alpha = 0.1
            args.beta = 0.5

        elif strategy == 'ewc_on':
            args.e_lambda = 1000
            args.gamma = 1.0

        # Output and logging
        args.wandb = False  # Disable wandb for batch experiments
        args.csv_log = True
        args.tensorboard = False
        args.savecheck = self.config.save_checkpoints

        # Create output directory structure
        scenario_dir = os.path.join(output_dir, f"{scenario}_{strategy}_seed{seed}")
        os.makedirs(scenario_dir, exist_ok=True)
        args.output_dir = scenario_dir

        # Device configuration
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Additional standard Mammoth args
        args.nowand = True
        args.non_verbose = False
        args.disable_log = False
        args.eval_epochs = None
        args.validation = False

        return args

    def run_scenario(self,
                    scenario: str,
                    strategy: str,
                    seed: int,
                    output_dir: str) -> Optional[EinstellungMetrics]:
        """
        Run a single scenario experiment.

        Args:
            scenario: Training scenario
            strategy: Continual learning strategy
            seed: Random seed
            output_dir: Output directory

        Returns:
            EinstellungMetrics if successful, None otherwise
        """
        self.logger.info(f"Running {scenario} with {strategy}, seed {seed}")

        try:
            # Create arguments for this scenario
            args = self.create_scenario_args(scenario, strategy, seed, output_dir)

            # Set random seed
            set_random_seed(seed)

            # Create evaluator
            evaluator = create_einstellung_evaluator(args)

            # Modify Mammoth's main to accept external evaluator
            # This would require integration with Mammoth's training loop

            # For now, simulate the experiment structure
            self.logger.info(f"Would run: {strategy} on {scenario} scenario with seed {seed}")

            # TODO: Integrate with actual Mammoth training
            # result_metrics = self._run_mammoth_experiment(args, evaluator)

            # Placeholder return for now
            return None

        except Exception as e:
            self.logger.error(f"Error in scenario {scenario}-{strategy}-{seed}: {e}")
            return None

    def _run_mammoth_experiment(self, args, evaluator) -> Optional[EinstellungMetrics]:
        """
        Run the actual Mammoth experiment with Einstellung evaluation.

        This would integrate with Mammoth's training pipeline.
        """
        # This is where we would call Mammoth's main training loop
        # with the evaluator plugin integrated

        # Placeholder implementation
        pass


class EinstellungExperimentRunner:
    """Main experiment runner for comprehensive Einstellung Effect analysis."""

    def __init__(self, config: EinstellungExperimentConfig):
        self.config = config
        self.scenario_manager = EinstellungScenarioManager(config)
        self.logger = logging.getLogger(__name__)

        # Results storage
        self.results = {}

        # Setup output directory
        self.output_dir = config.base_output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_comprehensive_experiment(self):
        """
        Run comprehensive Einstellung Effect experiments across all scenarios,
        strategies, and seeds.
        """
        self.logger.info("Starting comprehensive Einstellung Effect experiments")
        self.logger.info(f"Scenarios: {self.config.scenarios}")
        self.logger.info(f"Strategies: {self.config.strategies}")
        self.logger.info(f"Seeds: {self.config.seeds}")

        total_experiments = len(self.config.scenarios) * len(self.config.strategies) * len(self.config.seeds)
        self.logger.info(f"Total experiments: {total_experiments}")

        experiment_count = 0

        for scenario in self.config.scenarios:
            for strategy in self.config.strategies:
                strategy_results = []

                for seed in self.config.seeds:
                    experiment_count += 1
                    self.logger.info(f"Experiment {experiment_count}/{total_experiments}")

                    # Run single experiment
                    metrics = self.scenario_manager.run_scenario(
                        scenario, strategy, seed, self.output_dir
                    )

                    if metrics:
                        strategy_results.append(metrics)

                # Store results for this strategy-scenario combination
                key = f"{scenario}_{strategy}"
                self.results[key] = strategy_results

        # Generate comprehensive analysis
        self._generate_comprehensive_analysis()

    def run_single_experiment(self,
                            scenario: str,
                            strategy: str,
                            seed: int = 42) -> Optional[EinstellungMetrics]:
        """
        Run a single Einstellung experiment.

        Args:
            scenario: Training scenario
            strategy: Continual learning strategy
            seed: Random seed

        Returns:
            EinstellungMetrics if successful
        """
        return self.scenario_manager.run_scenario(scenario, strategy, seed, self.output_dir)

    def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis across all experiments."""
        self.logger.info("Generating comprehensive analysis")

        # Statistical analysis across seeds
        statistical_results = {}
        for key, metrics_list in self.results.items():
            if metrics_list:
                stats = calculate_cross_experiment_eri_statistics(metrics_list)
                statistical_results[key] = stats

        # Export results
        results_file = os.path.join(self.output_dir, 'comprehensive_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'configuration': self._config_to_dict(),
                'individual_results': {k: [self._metrics_to_dict(m) for m in v]
                                     for k, v in self.results.items()},
                'statistical_analysis': statistical_results
            }, f, indent=2, default=str)

        self.logger.info(f"Results exported to {results_file}")

        # Generate visualizations
        self._generate_visualizations()

    def _config_to_dict(self) -> Dict:
        """Convert configuration to dictionary for export."""
        return {
            'dataset': self.config.dataset,
            'backbone': self.config.backbone,
            'scenarios': self.config.scenarios,
            'strategies': self.config.strategies,
            'seeds': self.config.seeds,
            'n_epochs': self.config.n_epochs,
            'batch_size': self.config.batch_size,
            'lr': self.config.lr,
            'patch_size': self.config.patch_size,
            'patch_color': self.config.patch_color,
            'adaptation_threshold': self.config.adaptation_threshold
        }

    def _metrics_to_dict(self, metrics: EinstellungMetrics) -> Dict:
        """Convert EinstellungMetrics to dictionary."""
        return {
            'adaptation_delay': metrics.adaptation_delay,
            'performance_deficit': metrics.performance_deficit,
            'shortcut_feature_reliance': metrics.shortcut_feature_reliance,
            'eri_score': metrics.eri_score,
            'task1_accuracy_final': metrics.task1_accuracy_final,
            'task2_shortcut_accuracy': metrics.task2_shortcut_accuracy,
            'task2_masked_accuracy': metrics.task2_masked_accuracy,
            'task2_nonshortcut_accuracy': metrics.task2_nonshortcut_accuracy
        }

    def _generate_visualizations(self):
        """Generate comprehensive visualizations of results."""
        # This would create plots comparing:
        # - ERI scores across strategies
        # - Adaptation delays by scenario
        # - Performance deficits comparison
        # - Attention pattern analysis

        self.logger.info("Visualization generation would be implemented here")


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('einstellung_experiments.log')
        ]
    )


def main():
    """Main entry point for Einstellung experiments."""
    parser = argparse.ArgumentParser(description='Einstellung Effect Experiments')
    parser.add_argument('--scenario', type=str, choices=['sequential', 'scratch_t2', 'interleaved'],
                       help='Single scenario to run (if not specified, runs all)')
    parser.add_argument('--strategy', type=str, choices=['sgd', 'ewc_on', 'derpp'],
                       help='Single strategy to run (if not specified, runs all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for single experiment')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive experiments across all configurations')
    parser.add_argument('--output_dir', type=str, default='./einstellung_results',
                       help='Output directory for results')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'vit_base_patch16_224'],
                       help='Backbone architecture')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # Create configuration
    config = EinstellungExperimentConfig()
    config.base_output_dir = args.output_dir
    config.backbone = args.backbone

    # Create experiment runner
    runner = EinstellungExperimentRunner(config)

    if args.comprehensive:
        # Run comprehensive experiments
        runner.run_comprehensive_experiment()
    else:
        # Run single experiment
        scenario = args.scenario or 'sequential'
        strategy = args.strategy or 'sgd'

        logger = logging.getLogger(__name__)
        logger.info(f"Running single experiment: {scenario} with {strategy}")

        metrics = runner.run_single_experiment(scenario, strategy, args.seed)

        if metrics:
            logger.info("Experiment completed successfully")
            logger.info(f"ERI Score: {metrics.eri_score}")
        else:
            logger.error("Experiment failed")


if __name__ == '__main__':
    main()
