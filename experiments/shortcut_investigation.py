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
from utils.attention_visualization import (
    AttentionAnalyzer, visualize_attention_map,
    analyze_attention_patterns, compare_attention_across_tasks
)
from utils.network_flow_visualization import (
    ActivationExtractor, visualize_activation_flow,
    analyze_feature_representations, compare_activation_patterns
)
from utils.conf import set_random_seed
from datasets import get_dataset
from models import get_model
from backbone import get_backbone

class ShortcutInvestigationExperiment:
    """Main class for running shortcut feature investigation experiments."""
    
    def __init__(self, base_args: Dict, experiment_name: str = "shortcut_investigation"):
        """Initialize the experiment."""
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
        """Run a single experiment with a specific method and seed."""
        self.logger.info(f"Running {method} with seed {seed}")
        
        set_random_seed(seed)
        
        args_dict = self.base_args.copy()
        args_dict.update({
            'model': method,
            'seed': seed,
            'dataset': 'seq-cifar10-224-custom',
            'backbone': 'vit',
            'nowand': 1,
            'results_path': os.path.join(self.results_dir, f"{method}_seed_{seed}"),
        })
        
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
        
        args = argparse.Namespace(**args_dict)
        
        try:
            mammoth_main(args)
            model_path = os.path.join(args.results_path, 'model.pth')
            
            if os.path.exists(model_path):
                dataset = get_dataset(args)
                args.num_classes = dataset.N_CLASSES
                backbone = get_backbone(args)
                loss = dataset.get_loss()
                transform = dataset.get_transform()
                model = get_model(args, backbone, loss, transform, dataset=dataset)
                checkpoint = torch.load(model_path, map_location=args.device)
                model.load_state_dict(checkpoint)
                return model, {'status': 'success', 'model_path': model_path}
            else:
                return None, {'status': 'failed', 'error': 'Model file not found'}

        except Exception as e:
            self.logger.error(f"Error running {method} with seed {seed}: {str(e)}")
            return None, {'status': 'failed', 'error': str(e)}
    
    def analyze_model_after_task(self, model, dataset, task_id: int, method: str, seed: int) -> Dict:
        """Analyze model attention and activations after completing a task."""
        self.logger.info(f"Analyzing {method} after task {task_id}")
        
        # Create analysis directory
        analysis_dir = os.path.join(self.results_dir, f"{method}_seed_{seed}", f"task_{task_id}_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Get test samples
        test_samples = self._get_test_samples(dataset, task_id)
        
        # Initialize attention analyzer
        attention_analyzer = AttentionAnalyzer(model)
        attention_maps = {}
        
        # Process each class
        for class_name, samples in test_samples.items():
            class_attention_maps = {}
            
            for i, (sample, label) in enumerate(samples[:5]):  # Analyze first 5 samples
                # Get attention maps using the model's native attention computation
                attn_maps = attention_analyzer.extract_attention_maps(sample.unsqueeze(0))
                class_attention_maps[f'sample_{i}'] = attn_maps
                
                # Visualize attention for each block
                for layer_name, attention_map in attn_maps.items():
                    for head_idx in range(min(4, attention_map.shape[1])):  # First 4 heads
                        save_path = os.path.join(analysis_dir, 
                                               f'attention_{class_name}_sample_{i}_{layer_name}_head_{head_idx}.png')
                        visualize_attention_map(
                            attention_map, 
                            sample.unsqueeze(0),
                            head_idx=head_idx,
                            layer_name=layer_name,
                            save_path=save_path
                        )
            
            attention_maps[class_name] = class_attention_maps
        
        # Extract and analyze network activations
        activation_extractor = ActivationExtractor(model)
        activation_maps = {}
        
        for class_name, samples in test_samples.items():
            class_activations = []
            for sample, label in samples[:10]:  # Analyze first 10 samples
                sample_activations = activation_extractor.extract_activations(sample.unsqueeze(0))
                class_activations.append(sample_activations)
            activation_maps[class_name] = class_activations
        
        # Aggregate attention maps for analysis
        all_attention_maps = {}
        for class_name, class_attention in attention_maps.items():
            for sample_id, sample_attention in class_attention.items():
                for layer_name, attn in sample_attention.items():
                    if layer_name not in all_attention_maps:
                        all_attention_maps[layer_name] = []
                    all_attention_maps[layer_name].append(attn)
        
        # Analyze attention patterns
        attention_stats = analyze_attention_patterns(
            all_attention_maps,
            list(test_samples.keys()),
            analysis_dir
        )
        
        # Analyze activation flow
        sample_activations = activation_maps[list(activation_maps.keys())[0]][0]
        visualize_activation_flow(
            sample_activations,
            os.path.join(analysis_dir, 'activation_flow.png')
        )
        
        # Clean up activation extractor
        activation_extractor.remove_hooks()
        
        return {
            'attention_stats': attention_stats,
            'analysis_dir': analysis_dir,
            'num_samples_analyzed': sum(len(samples) for samples in test_samples.values())
        }
    
    def _get_test_samples(self, dataset, task_id: int) -> Dict[str, List[Tuple[torch.Tensor, int]]]:
        """Get test samples for analysis."""
        task_labels = dataset.get_task_labels()
        current_task_classes = task_labels[task_id]
        
        _, test_loader = dataset.get_data_loaders()
        test_samples = {class_name: [] for class_name in current_task_classes}
        
        for batch_idx, (data, targets, _) in enumerate(test_loader):
            for i, (sample, target) in enumerate(zip(data, targets)):
                class_idx = target.item()
                if class_idx < len(current_task_classes):
                    class_name = current_task_classes[class_idx]
                    if len(test_samples[class_name]) < 10:  # Collect up to 10 samples per class
                        test_samples[class_name].append((sample, target))
            
            if all(len(samples) >= 10 for samples in test_samples.values()):
                break
        
        return test_samples
    
    def run_full_experiment(self, methods: List[str] = ['derpp', 'ewc_on'], 
                          seeds: List[int] = [42, 123, 456]) -> Dict:
        """Run the full experiment with multiple methods and seeds."""
        self.logger.info(f"Starting full experiment with methods: {methods}, seeds: {seeds}")
        
        for method in methods:
            self.results['methods'][method] = {}
            
            for seed in seeds:
                self.logger.info(f"Running {method} with seed {seed}")
                model, exp_results = self.run_single_experiment(method, seed)
                
                if model is not None and exp_results['status'] == 'success':
                    task_analyses = {}
                    
                    # Configure dataset for analysis
                    args_dict = self.base_args.copy()
                    args_dict.update({'dataset': 'seq-cifar10-224-custom', 'seed': seed})
                    args = argparse.Namespace(**args_dict)
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
    base_args = {
        'n_epochs': 10,
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
    
    experiment = ShortcutInvestigationExperiment(
        base_args=base_args,
        experiment_name=f"shortcut_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    results = experiment.run_full_experiment(
        methods=['derpp', 'ewc_on'],
        seeds=[42, 123, 456]
    )
    
    print("Experiment completed successfully!")
    print(f"Results saved in: {experiment.results_dir}")

if __name__ == "__main__":
    main()
