"""
Main experiment script for investigating shortcut features in continual learning.

This script runs experiments with DER++ and EWC on custom CIFAR-10 tasks to investigate
how different continual learning methods handle shortcut features.

Tasks:
- Task 1: airplane, automobile (potential shortcuts: sky, road)
- Task 2: bird, truck (potential shortcuts: sky, road/wheels)
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add mammoth path
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, mammoth_path)

from main import main as mammoth_main
from utils.attention_visualization import AttentionExtractor, visualize_attention_map, analyze_attention_patterns, compare_attention_across_tasks
from utils.network_flow_visualization import ActivationExtractor, visualize_activation_flow, analyze_feature_representations, compare_activation_patterns
from utils.conf import set_random_seed
from datasets import get_dataset
from models import get_model
from backbone import get_backbone


class ShortcutInvestigationExperiment:
    """
    Main class for running shortcut feature investigation experiments.
    """
    
    def __init__(self, base_args: Dict, experiment_name: str = "shortcut_investigation"):
        """
        Initialize the experiment.
        
        Args:
            base_args: Base arguments for the experiment
            experiment_name: Name of the experiment
        """
        self.base_args = base_args
        self.experiment_name = experiment_name
        self.results_dir = os.path.join("results", experiment_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.results_dir, 'experiment.log')),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized experiment: {experiment_name}")
        
        # Store results
        self.results = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'base_args': base_args,
            'methods': {}
        }
    
    def run_single_experiment(self, method: str, seed: int) -> Tuple[object, Dict]:
        """
        Run a single experiment with a specific method and seed.
        
        Args:
            method: Continual learning method ('derpp' or 'ewc_on')
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (trained_model, experiment_results)
        """
        self.logger.info(f"Running {method} with seed {seed}")
        
        # Set random seed
        set_random_seed(seed)
        
        # Create arguments for this experiment
        args_dict = self.base_args.copy()
        args_dict.update({
            'model': method,
            'seed': seed,
            'dataset': 'seq-cifar10-custom',
            'backbone': 'vit',
            'nowand': 1,  # Disable wandb for this experiment
            'results_path': os.path.join(self.results_dir, f"{method}_seed_{seed}"),
        })
        
        # Method-specific arguments
        if method == 'derpp':
            args_dict.update({
                'buffer_size': 200,
                'alpha': 0.1,
                'beta': 0.5,
            })
        elif method == 'ewc_on':
            args_dict.update({
                'e_lambda': 0.4,
                'gamma': 0.85,
            })
        
        # Convert to argparse.Namespace
        from argparse import Namespace
        args = Namespace(**args_dict)
        
        # Run the experiment
        try:
            # This will train the model and return results
            mammoth_main(args)
            
            # Load the trained model for analysis
            model_path = os.path.join(args.results_path, 'model.pth')
            if os.path.exists(model_path):
                # Load model for analysis
                dataset = get_dataset(args)
                backbone = get_backbone(args)
                loss = dataset.get_loss()
                model = get_model(args, backbone, loss, dataset.get_transform(), dataset=dataset)
                
                # Load trained weights
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint)
                
                return model, {'status': 'success', 'model_path': model_path}
            else:
                return None, {'status': 'failed', 'error': 'Model file not found'}
                
        except Exception as e:
            self.logger.error(f"Error running {method} with seed {seed}: {str(e)}")
            return None, {'status': 'failed', 'error': str(e)}
    
    def analyze_model_after_task(self, model, dataset, task_id: int, method: str, seed: int) -> Dict:
        """
        Analyze model attention and activations after completing a task.
        
        Args:
            model: Trained model
            dataset: Dataset object
            task_id: Task ID (0 or 1)
            method: Method name
            seed: Random seed
            
        Returns:
            Dictionary with analysis results
        """
        self.logger.info(f"Analyzing {method} after task {task_id}")
        
        # Create analysis directory
        analysis_dir = os.path.join(self.results_dir, f"{method}_seed_{seed}", f"task_{task_id}_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Get test samples for both tasks
        test_samples = self._get_test_samples(dataset, task_id)
        
        # Extract attention maps
        attention_extractor = AttentionExtractor(model)
        attention_maps = {}
        
        for class_name, samples in test_samples.items():
            class_attention_maps = {}
            for i, (sample, label) in enumerate(samples[:5]):  # Analyze first 5 samples
                sample_attention = attention_extractor.extract_attention(sample.unsqueeze(0))
                class_attention_maps[f'sample_{i}'] = sample_attention
                
                # Visualize attention for this sample
                for layer_name, attn in sample_attention.items():
                    if 'block' in layer_name:  # Focus on transformer blocks
                        for head_idx in range(min(4, attn.shape[1])):  # First 4 heads
                            save_path = os.path.join(analysis_dir, 
                                                   f'attention_{class_name}_sample_{i}_{layer_name}_head_{head_idx}.png')
                            visualize_attention_map(attn, sample.unsqueeze(0), head_idx, 
                                                   f"{layer_name}", save_path)
            
            attention_maps[class_name] = class_attention_maps
        
        # Extract network activations
        activation_extractor = ActivationExtractor(model)
        activation_maps = {}
        
        for class_name, samples in test_samples.items():
            class_activations = []
            for sample, label in samples[:10]:  # Analyze first 10 samples
                sample_activations = activation_extractor.extract_activations(sample.unsqueeze(0))
                class_activations.append(sample_activations)
            activation_maps[class_name] = class_activations
        
        # Analyze attention patterns
        all_attention_maps = {}
        for class_name, class_attention in attention_maps.items():
            for sample_id, sample_attention in class_attention.items():
                for layer_name, attn in sample_attention.items():
                    if layer_name not in all_attention_maps:
                        all_attention_maps[layer_name] = []
                    all_attention_maps[layer_name].append(attn)
        
        # Concatenate attention maps for analysis
        concatenated_attention = {}
        for layer_name, attn_list in all_attention_maps.items():
            concatenated_attention[layer_name] = torch.cat(attn_list, dim=0)
        
        attention_stats = analyze_attention_patterns(
            concatenated_attention, 
            list(test_samples.keys()), 
            analysis_dir
        )
        
        # Analyze activation flow
        sample_activations = activation_maps[list(activation_maps.keys())[0]][0]
        visualize_activation_flow(sample_activations, 
                                os.path.join(analysis_dir, 'activation_flow.png'))
        
        # Clean up extractors
        attention_extractor.remove_hooks()
        activation_extractor.remove_hooks()
        
        return {
            'attention_stats': attention_stats,
            'analysis_dir': analysis_dir,
            'num_samples_analyzed': sum(len(samples) for samples in test_samples.values())
        }
    
    def _get_test_samples(self, dataset, task_id: int) -> Dict[str, List[Tuple[torch.Tensor, int]]]:
        """
        Get test samples for analysis.
        
        Args:
            dataset: Dataset object
            task_id: Task ID
            
        Returns:
            Dictionary mapping class names to list of (sample, label) tuples
        """
        # Get class names for the current task
        task_labels = dataset.get_task_labels()
        current_task_classes = task_labels[task_id]
        
        # Get test loader
        _, test_loader = dataset.get_data_loaders()
        
        # Collect samples for each class
        test_samples = {class_name: [] for class_name in current_task_classes}
        
        for batch_idx, (data, targets, _) in enumerate(test_loader):
            for i, (sample, target) in enumerate(zip(data, targets)):
                class_idx = target.item()
                if class_idx < len(current_task_classes):
                    class_name = current_task_classes[class_idx]
                    if len(test_samples[class_name]) < 10:  # Collect up to 10 samples per class
                        test_samples[class_name].append((sample, target))
            
            # Stop if we have enough samples
            if all(len(samples) >= 10 for samples in test_samples.values()):
                break
        
        return test_samples
    
    def run_full_experiment(self, methods: List[str] = ['derpp', 'ewc_on'], 
                          seeds: List[int] = [42, 123, 456]) -> Dict:
        """
        Run the full experiment with multiple methods and seeds.
        
        Args:
            methods: List of continual learning methods to test
            seeds: List of random seeds for reproducibility
            
        Returns:
            Dictionary with all experiment results
        """
        self.logger.info(f"Starting full experiment with methods: {methods}, seeds: {seeds}")
        
        for method in methods:
            self.results['methods'][method] = {}
            
            for seed in seeds:
                self.logger.info(f"Running {method} with seed {seed}")
                
                # Run the experiment
                model, exp_results = self.run_single_experiment(method, seed)
                
                if model is not None and exp_results['status'] == 'success':
                    # Analyze after each task
                    task_analyses = {}
                    
                    # Note: In a real implementation, you would need to modify the training loop
                    # to save models after each task and then load them for analysis
                    # For now, we'll analyze the final model
                    
                    # Create a dummy dataset for analysis
                    args_dict = self.base_args.copy()
                    args_dict.update({'dataset': 'seq-cifar10-custom', 'seed': seed})
                    from argparse import Namespace
                    args = Namespace(**args_dict)
                    dataset = get_dataset(args)
                    
                    # Analyze final model (after both tasks)
                    final_analysis = self.analyze_model_after_task(model, dataset, 1, method, seed)
                    task_analyses['final'] = final_analysis
                    
                    self.results['methods'][method][seed] = {
                        'experiment_results': exp_results,
                        'task_analyses': task_analyses
                    }
                else:
                    self.results['methods'][method][seed] = {
                        'experiment_results': exp_results,
                        'task_analyses': {}
                    }
        
        # Save results
        results_file = os.path.join(self.results_dir, 'experiment_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Experiment completed. Results saved to {results_file}")
        return self.results


def main():
    """Main function to run the shortcut investigation experiment."""
    
    # Base arguments for all experiments
    base_args = {
        'n_epochs': 10,  # Reduced for faster experimentation
        'batch_size': 32,
        'lr': 0.01,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'debug_mode': 0,
        'validation': None,
        'savecheck': 0,
        'inference_only': 0,
        'ignore_other_metrics': 0,
        'eval_future': 0,
        'enable_other_metrics': 1,
    }
    
    # Create and run experiment
    experiment = ShortcutInvestigationExperiment(
        base_args=base_args,
        experiment_name=f"shortcut_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run with both methods and multiple seeds for reproducibility
    results = experiment.run_full_experiment(
        methods=['derpp', 'ewc_on'],
        seeds=[42, 123, 456]
    )
    
    print("Experiment completed successfully!")
    print(f"Results saved in: {experiment.results_dir}")


if __name__ == "__main__":
    main()
