#!/usr/bin/env python3
"""
ERI Visualization System - End-to-End Pipeline Runner

This runner extends the existing Mammoth Einstellung infrastructure to provide
comprehensive ERI visualization capabilities. It integrates with:

- datasets/seq_cifar100_einstellung_224.py: ViT-compatible dataset
- utils/einstellung_evaluator.py: Plugin-based evaluation system
- Existing Mammoth training pipeline and checkpoint management

CRITICAL: This extends existing infrastructure rather than recreating it.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

# Add Mammoth to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import existing Mammoth infrastructure
from main import main as mammoth_main
from utils.args import add_experiment_args, add_management_args
from utils.conf import set_random_seed
from utils.einstellung_evaluator import EinstellungEvaluator
from utils.einstellung_integration import enable_einstellung_integration, get_einstellung_evaluator

# Import ERI visualization components
from eri_vis.integration.mammoth_integration import MammothERIIntegration
from eri_vis.integration.hooks import ERIExperimentHooks
from eri_vis.styles import PlotStyleConfig


class ERIExperimentRunner:
    """
    End-to-end pipeline runner for ERI visualization experiments.

    Integrates with existing Mammoth infrastructure to provide:
    - Configuration parsing and validation
    - Mammoth initialization with existing dataset
    - Hook registration with EinstellungEvaluator
    - Automatic CSV export and PDF generation
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ERI experiment runner.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config_path:
            self.load_config(config_path)

        # Initialize components
        self.mammoth_integration = None
        self.experiment_hooks = None

    def load_config(self, config_path: str) -> None:
        """
        Load experiment configuration from YAML file.

        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def create_mammoth_args(self,
                           method: str = "derpp",
                           seed: int = 42,
                           output_dir: Optional[str] = None) -> argparse.Namespace:
        """
        Create Mammoth arguments namespace from configuration.

        Args:
            method: Continual learning method
            seed: Random seed
            output_dir: Output directory override

        Returns:
            Configured arguments namespace
        """
        # Start with base configuration
        base_config = self.config.get('experiment', {})

        # Create arguments namespace
        args = argparse.Namespace()

        # Dataset and model configuration
        args.dataset = base_config.get('dataset', 'seq-cifar100-einstellung-224')
        args.model = method
        args.backbone = base_config.get('backbone', 'resnet18')

        # Training configuration
        args.n_epochs = base_config.get('n_epochs', 50)
        args.batch_size = base_config.get('batch_size', 32)
        args.lr = base_config.get('lr', 0.01)
        args.seed = seed

        # Einstellung-specific configuration
        einstellung_config = self.config.get('einstellung', {})
        args.einstellung_patch_size = einstellung_config.get('patch_size', 4)
        args.einstellung_patch_color = einstellung_config.get('patch_color', [255, 0, 255])
        args.einstellung_adaptation_threshold = einstellung_config.get('adaptation_threshold', 0.6)
        args.einstellung_evaluation_subsets = True
        args.einstellung_extract_attention = einstellung_config.get('extract_attention', True)

        # Method-specific parameters
        method_config = self.config.get('methods', {}).get(method, {})
        if method == 'derpp':
            args.buffer_size = method_config.get('buffer_size', 500)
            args.alpha = method_config.get('alpha', 0.1)
            args.beta = method_config.get('beta', 0.5)
        elif method == 'ewc_on':
            args.e_lambda = method_config.get('e_lambda', 1000)
            args.gamma = method_config.get('gamma', 1.0)

        # Output configuration
        if output_dir:
            args.output_dir = output_dir
        else:
            base_output = self.config.get('output', {}).get('base_dir', './eri_results')
            args.output_dir = os.path.join(base_output, f"{method}_seed{seed}")

        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Device configuration - fix the device/gpu_id issue
        if torch.cuda.is_available():
            args.device = '0'  # Device ID as string for get_device function
            args.gpu_id = 0
        else:
            args.device = None  # Let get_device handle CPU fallback
            args.gpu_id = -1  # Use -1 for CPU

        # Required Mammoth arguments
        args.wandb = False  # Disable wandb for ERI experiments
        args.csv_log = True
        args.tensorboard = False
        args.nowand = True
        args.non_verbose = False
        args.disable_log = False
        args.validation = False
        args.eval_epochs = None
        args.savecheck = 'last' if self.config.get('output', {}).get('save_checkpoints', True) else None
        args.loadcheck = None
        args.inference_only = False

        # Additional required arguments
        args.label_perc = 1.0
        args.label_perc_by_class = 1.0
        args.joint = False
        args.start_from = None
        args.stop_after = None
        args.enable_other_metrics = False
        args.eval_future = False
        args.noise_rate = 0.0

        # Einstellung specific flags
        args.einstellung_apply_shortcut = True

        # Optimizer settings
        args.optim_wd = 0.0
        args.optim_mom = 0.9
        args.optim_nesterov = False

        # Scheduler settings (optional)
        args.lr_scheduler = None

        # Other common settings
        args.debug_mode = False
        args.notes = f"ERI experiment: {method} seed {seed}"
        args.base_path = './data'  # Base path for datasets

        # Additional missing attributes that might be needed
        args.conf_jobnum = None
        args.conf_timestamp = None
        args.conf_host = None
        args.code_optimization = False

        # More potential missing attributes
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

        return args

    def _args_to_cmd_list(self, args: argparse.Namespace) -> List[str]:
        """
        Convert arguments namespace to command line argument list.

        Args:
            args: Arguments namespace

        Returns:
            List of command line arguments
        """
        cmd_args = []

        # Core arguments
        cmd_args.extend(['--dataset', args.dataset])
        cmd_args.extend(['--model', args.model])
        cmd_args.extend(['--backbone', args.backbone])
        cmd_args.extend(['--n_epochs', str(args.n_epochs)])
        cmd_args.extend(['--batch_size', str(args.batch_size)])
        cmd_args.extend(['--lr', str(args.lr)])
        cmd_args.extend(['--seed', str(args.seed)])

        # Method-specific parameters
        if hasattr(args, 'buffer_size') and args.buffer_size:
            cmd_args.extend(['--buffer_size', str(args.buffer_size)])
        if hasattr(args, 'alpha') and args.alpha is not None:
            cmd_args.extend(['--alpha', str(args.alpha)])
        if hasattr(args, 'beta') and args.beta is not None:
            cmd_args.extend(['--beta', str(args.beta)])
        if hasattr(args, 'e_lambda') and args.e_lambda is not None:
            cmd_args.extend(['--e_lambda', str(args.e_lambda)])
        if hasattr(args, 'gamma') and args.gamma is not None:
            cmd_args.extend(['--gamma', str(args.gamma)])

        # Einstellung parameters
        if hasattr(args, 'einstellung_patch_size'):
            cmd_args.extend(['--einstellung_patch_size', str(args.einstellung_patch_size)])
        if hasattr(args, 'einstellung_adaptation_threshold'):
            cmd_args.extend(['--einstellung_adaptation_threshold', str(args.einstellung_adaptation_threshold)])
        if hasattr(args, 'einstellung_evaluation_subsets') and args.einstellung_evaluation_subsets:
            cmd_args.extend(['--einstellung_evaluation_subsets', '1'])
        if hasattr(args, 'einstellung_extract_attention') and args.einstellung_extract_attention:
            cmd_args.extend(['--einstellung_extract_attention', '1'])
        if hasattr(args, 'einstellung_apply_shortcut') and args.einstellung_apply_shortcut:
            cmd_args.extend(['--einstellung_apply_shortcut', '1'])

        # Patch color (special handling for list)
        if hasattr(args, 'einstellung_patch_color') and args.einstellung_patch_color:
            cmd_args.extend(['--einstellung_patch_color'] + [str(c) for c in args.einstellung_patch_color])

        # Logging and output (only include valid arguments)
        if args.savecheck:
            cmd_args.extend(['--savecheck', str(args.savecheck)])

        # Device (only if not None)
        if args.device is not None:
            cmd_args.extend(['--device', str(args.device)])

        return cmd_args

    def setup_eri_integration(self, args: argparse.Namespace) -> None:
        """
        Setup ERI visualization integration with existing Mammoth infrastructure.

        Args:
            args: Mammoth arguments namespace
        """
        # Create output directories
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create visualization style configuration
        style_config = PlotStyleConfig()

        # Initialize experiment hooks
        self.experiment_hooks = ERIExperimentHooks(
            output_dir=str(output_dir),
            style_config=style_config,
            export_frequency=1,
            auto_visualize=True
        )

        # Initialize Mammoth integration (will be connected to evaluator during training)
        self.mammoth_integration = MammothERIIntegration(
            output_dir=str(output_dir),
            style_config=style_config
        )

        self.logger.info(f"ERI integration setup complete. Output: {output_dir}")

    def register_hooks_with_evaluator(self, evaluator) -> None:
        """
        Register ERI visualization hooks with the existing EinstellungEvaluator.

        Args:
            evaluator: EinstellungEvaluator instance
        """
        if not self.experiment_hooks:
            raise RuntimeError("ERI integration not setup. Call setup_eri_integration first.")

        # Connect the integration to the evaluator
        if self.mammoth_integration:
            self.mammoth_integration.evaluator = evaluator

        # Register hooks with evaluator
        # Note: This assumes the evaluator has hook registration methods
        # If not, we'll need to modify the evaluator or use a different approach

        original_meta_begin_task = evaluator.meta_begin_task
        original_after_training_epoch = getattr(evaluator, 'after_training_epoch', None)
        original_meta_end_task = evaluator.meta_end_task

        def enhanced_meta_begin_task(model, dataset):
            result = original_meta_begin_task(model, dataset)
            # No specific task begin hook in our implementation
            return result

        def enhanced_after_training_epoch(model, dataset, epoch):
            if original_after_training_epoch:
                result = original_after_training_epoch(model, dataset, epoch)
            else:
                result = None
            self.experiment_hooks.on_epoch_end(epoch, evaluator)
            return result

        def enhanced_meta_end_task(model, dataset):
            result = original_meta_end_task(model, dataset)
            self.experiment_hooks.on_task_end(dataset.i if hasattr(dataset, 'i') else 0, evaluator)
            return result

        # Replace methods
        evaluator.meta_begin_task = enhanced_meta_begin_task
        if original_after_training_epoch:
            evaluator.after_training_epoch = enhanced_after_training_epoch
        evaluator.meta_end_task = enhanced_meta_end_task

        self.logger.info("ERI visualization hooks registered with evaluator")

    def run_single_experiment(self,
                            method: str = "derpp",
                            seed: int = 42,
                            output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a single ERI experiment with the specified configuration.

        Args:
            method: Continual learning method
            seed: Random seed
            output_dir: Output directory override

        Returns:
            Dictionary with experiment results
        """
        self.logger.info(f"Starting ERI experiment: {method} with seed {seed}")

        try:
            # Create Mammoth arguments
            args = self.create_mammoth_args(method, seed, output_dir)

            # Set random seed
            set_random_seed(seed)

            # Setup ERI integration
            self.setup_eri_integration(args)

            # Enable Einstellung integration in Mammoth
            enable_einstellung_integration(args)

            # Get the evaluator that will be created by Mammoth
            # Note: This assumes get_einstellung_evaluator creates/returns the evaluator
            evaluator = get_einstellung_evaluator()

            # Register our visualization hooks
            self.register_hooks_with_evaluator(evaluator)

            # Run Mammoth training with integrated evaluation
            self.logger.info("Starting Mammoth training with ERI integration")
            start_time = time.time()

            # Convert args namespace to command line arguments
            cmd_args = self._args_to_cmd_list(args)

            # Call Mammoth via subprocess (like the existing runner)
            import subprocess
            cmd = [sys.executable, 'main.py'] + cmd_args

            self.logger.info(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=Path(__file__).parent.parent.parent  # Run from mammoth root
            )

            if result.returncode != 0:
                raise RuntimeError(f"Mammoth training failed: {result.stderr}")

            mammoth_result = result.stdout

            training_time = time.time() - start_time

            # Generate final visualizations
            self.logger.info("Generating final ERI visualizations")
            self.experiment_hooks.on_experiment_end(evaluator)

            # Collect results
            results = {
                'method': method,
                'seed': seed,
                'training_time': training_time,
                'output_dir': args.output_dir,
                'mammoth_result': mammoth_result,
                'success': True
            }

            self.logger.info(f"ERI experiment completed successfully in {training_time:.1f}s")
            return results

        except Exception as e:
            self.logger.error(f"ERI experiment failed: {e}")
            return {
                'method': method,
                'seed': seed,
                'success': False,
                'error': str(e)
            }

    def run_batch_experiments(self,
                            methods: Optional[List[str]] = None,
                            seeds: Optional[List[int]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run batch experiments across multiple methods and seeds.

        Args:
            methods: List of methods to run (defaults to config)
            seeds: List of seeds to run (defaults to config)

        Returns:
            Dictionary mapping method names to lists of results
        """
        if methods is None:
            methods = self.config.get('methods', {}).keys()
        if seeds is None:
            seeds = self.config.get('experiment', {}).get('seeds', [42])

        self.logger.info(f"Running batch experiments: {len(methods)} methods × {len(seeds)} seeds")

        all_results = {}

        for method in methods:
            method_results = []

            for seed in seeds:
                result = self.run_single_experiment(method, seed)
                method_results.append(result)

            all_results[method] = method_results

        # Export batch results
        self._export_batch_results(all_results)

        return all_results

    def _export_batch_results(self, results: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Export batch experiment results to JSON.

        Args:
            results: Batch experiment results
        """
        output_dir = Path(self.config.get('output', {}).get('base_dir', './eri_results'))
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / 'batch_experiment_results.json'

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Batch results exported to {results_file}")


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('eri_experiment.log')
        ]
    )


def main():
    """Main entry point for ERI visualization experiments."""
    parser = argparse.ArgumentParser(description='ERI Visualization System - End-to-End Pipeline')

    # Configuration
    parser.add_argument('--config', type=str,
                       default='experiments/configs/cifar100_einstellung224.yaml',
                       help='Path to configuration file')

    # Single experiment options
    parser.add_argument('--method', type=str, default='derpp',
                       choices=['sgd', 'ewc_on', 'derpp'],
                       help='Continual learning method')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory (overrides config)')

    # Batch experiment options
    parser.add_argument('--batch', action='store_true',
                       help='Run batch experiments from config')

    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize runner
        runner = ERIExperimentRunner(args.config)

        if args.batch:
            # Run batch experiments
            logger.info("Running batch experiments")
            results = runner.run_batch_experiments()

            # Print summary
            total_experiments = sum(len(method_results) for method_results in results.values())
            successful = sum(1 for method_results in results.values()
                           for result in method_results if result.get('success', False))

            logger.info(f"Batch experiments completed: {successful}/{total_experiments} successful")

        else:
            # Run single experiment
            logger.info(f"Running single experiment: {args.method} with seed {args.seed}")
            result = runner.run_single_experiment(args.method, args.seed, args.output_dir)

            if result.get('success', False):
                logger.info("Experiment completed successfully")
                logger.info(f"Output directory: {result['output_dir']}")
            else:
                logger.error(f"Experiment failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Runner failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
