#!/usr/bin/env python3
"""
Enhanced experiment runner for investigating shortcut features with attention visualization.

This script focuses on:
1. Reproducible random initialization across all experiments
2. Comprehensive attention map visualization after each task
3. Comparison between DER++ and EWC methods
4. Saving model weights and performance metrics after each task

Tasks:
- Task 1: airplane (0), automobile (1) - potential shortcuts: sky, road
- Task 2: bird (2), truck (9) - potential shortcuts: sky, road/wheels
"""

import os
import sys
import argparse
import subprocess
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experiment configuration
EXPERIMENT_CONFIG = {
    "methods": ['derpp', 'ewc_on'],
    "seeds": [42, 123, 456],  # Fixed seeds for reproducibility
    "epochs_per_task": 15,  # Sufficient for learning but not too long
    "batch_size": 32,
    "lr": 0.01,
    "optimizer": "sgd",
    "num_workers": 4,
    "dataset": "seq-cifar10-224-custom",
    "backbone": "vit",
}

def ensure_reproducible_initialization(seed: int):
    """
    Ensure reproducible random initialization across all experiments.
    
    Args:
        seed: Random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set reproducible initialization with seed {seed}")

def run_single_experiment_with_analysis(
    method: str, 
    seed: int, 
    config: Dict, 
    output_dir: str, 
    use_gpu: bool
) -> Dict[str, Any]:
    """
    Run a single experiment with comprehensive attention analysis.
    
    Args:
        method: The continual learning method ('derpp' or 'ewc_on')
        seed: Random seed for reproducibility
        config: Experiment configuration
        output_dir: Directory to save results
        use_gpu: Whether to use GPU
        
    Returns:
        Dictionary with experiment results and analysis paths
    """
    logger.info(f"Running {method} with seed {seed}")
    
    # Ensure reproducible initialization
    ensure_reproducible_initialization(seed)
    
    # Create experiment-specific directory
    exp_dir = os.path.join(output_dir, f"{method}_seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Build command for mammoth
    cmd = [
        sys.executable, 'main.py',
        '--dataset', config['dataset'],
        '--model', method,
        '--backbone', config['backbone'],
        '--results_path', exp_dir,
        '--seed', str(seed),
        '--n_epochs', str(config['epochs_per_task']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--optimizer', config['optimizer'],
        '--optim_wd', '0.0',
        '--optim_mom', '0.0',
        '--num_workers', str(config['num_workers']),
        '--drop_last', '0',
        '--debug_mode', '0',
        '--savecheck', 'last',  # Save checkpoints for analysis
    ]
    
    # Note: Don't set --device cpu as it expects GPU device numbers
    # The framework will automatically use CPU if no GPU is available
    
    # Method-specific parameters
    if method == 'derpp':
        cmd.extend(['--buffer_size', '200', '--alpha', '0.1', '--beta', '0.5'])
    elif method == 'ewc_on':
        cmd.extend(['--e_lambda', '0.4', '--gamma', '0.85'])
    
    try:
        # Run the experiment with real-time output
        logger.info(f"Executing command: {' '.join(cmd)}")

        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Collect output and show progress
        output_lines = []
        logger.info(f"Starting {method} training with seed {seed}...")

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Show important progress lines
                if any(keyword in output.lower() for keyword in ['epoch', 'task', 'accuracy', 'loss']):
                    logger.info(f"[{method}] {output.strip()}")

        # Wait for process to complete
        return_code = process.poll()
        full_output = '\n'.join(output_lines)

        if return_code != 0:
            logger.error(f"Experiment failed with return code {return_code}")
            logger.error(f"Last few lines of output:")
            for line in output_lines[-10:]:
                logger.error(f"  {line}")
            return {
                'status': 'failed',
                'error': f"Return code {return_code}",
                'output': full_output
            }

        logger.info(f"✓ {method} training completed successfully")

        # Parse results from stdout
        final_accuracy = parse_accuracy_from_output(full_output)
        logger.info(f"Final accuracy: {final_accuracy}")

        # Run attention analysis
        logger.info(f"Starting attention analysis for {method}...")
        analysis_results = run_attention_analysis(method, seed, exp_dir, config)

        return {
            'status': 'success',
            'final_accuracy': final_accuracy,
            'analysis_results': analysis_results,
            'experiment_dir': exp_dir,
            'training_output': full_output
        }

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}

def parse_accuracy_from_output(output: str) -> Dict[str, float]:
    """Parse final accuracy from mammoth output."""
    import re
    
    results = {}
    # Look for final accuracy line
    pattern = r"Accuracy for \d+ task\(s\):\s+\[Class-IL\]:\s+([\d.]+) %\s+\[Task-IL\]:\s+([\d.]+) %"
    match = re.search(pattern, output)
    
    if match:
        results['class_il_accuracy'] = float(match.group(1))
        results['task_il_accuracy'] = float(match.group(2))
    
    return results

def run_attention_analysis(method: str, seed: int, exp_dir: str, config: Dict) -> Dict[str, Any]:
    """
    Run comprehensive attention analysis on the trained model.
    
    Args:
        method: Method name
        seed: Random seed used
        exp_dir: Experiment directory
        config: Experiment configuration
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Running attention analysis for {method} seed {seed}")
    
    try:
        # Import mammoth components
        from datasets import get_dataset
        from models import get_model
        from backbone import get_backbone
        from utils.attention_visualization import AttentionExtractor, visualize_attention_map
        from utils.network_flow_visualization import ActivationExtractor
        from argparse import Namespace
        
        # Create args for loading model
        args = Namespace(
            dataset=config['dataset'],
            backbone=config['backbone'],
            seed=seed,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            pretrained=False,
            pretrain_type='in21k-ft-in1k',
            custom_class_order=None,
            permute_classes=False,
            validation_mode='current',
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            drop_last=False
        )
        
        # Load dataset
        dataset = get_dataset(args)
        args.num_classes = dataset.N_CLASSES
        
        # Load model
        backbone = get_backbone(args)
        loss = dataset.get_loss()
        transform = dataset.get_transform()
        model = get_model(args, backbone, loss, transform, dataset=dataset)
        
        # Find the model checkpoint (mammoth saves in checkpoints/ subdirectory)
        checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        model_path = None

        if os.path.exists(checkpoint_dir):
            # Find the checkpoint file
            for file in os.listdir(checkpoint_dir):
                if file.endswith('_last'):
                    model_path = os.path.join(checkpoint_dir, file)
                    break

        # Also check for model.pth in the main directory
        if model_path is None:
            potential_model_path = os.path.join(exp_dir, 'model.pth')
            if os.path.exists(potential_model_path):
                model_path = potential_model_path

        if not model_path or not os.path.exists(model_path):
            logger.warning(f"Model file not found in {exp_dir}")
            return {'status': 'model_not_found'}

        checkpoint = torch.load(model_path, map_location=args.device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Create analysis directory
        analysis_dir = os.path.join(exp_dir, 'attention_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Get test samples for analysis
        test_samples = get_test_samples_for_analysis(dataset, analysis_dir)
        
        # Extract and visualize attention maps
        attention_results = extract_and_visualize_attention(
            model, test_samples, analysis_dir, dataset.get_class_names()
        )
        
        # Extract and analyze network activations
        activation_results = extract_and_analyze_activations(
            model, test_samples, analysis_dir
        )
        
        return {
            'status': 'success',
            'analysis_dir': analysis_dir,
            'attention_results': attention_results,
            'activation_results': activation_results,
            'num_samples_analyzed': len(test_samples)
        }
        
    except Exception as e:
        logger.error(f"Error in attention analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}

def get_test_samples_for_analysis(dataset, analysis_dir: str, samples_per_class: int = 10) -> List[Tuple]:
    """Get test samples for attention analysis."""
    logger.info("Collecting test samples for analysis")
    
    # Get test loader
    _, test_loader = dataset.get_data_loaders()
    class_names = dataset.get_class_names()
    
    # Collect samples by class
    samples_by_class = {class_name: [] for class_name in class_names}
    
    for batch_data, batch_targets, _ in test_loader:
        for sample, target in zip(batch_data, batch_targets):
            class_idx = target.item()
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
                if len(samples_by_class[class_name]) < samples_per_class:
                    samples_by_class[class_name].append((sample, target, class_name))
        
        # Stop when we have enough samples
        if all(len(samples) >= samples_per_class for samples in samples_by_class.values()):
            break
    
    # Flatten to single list
    all_samples = []
    for class_name, samples in samples_by_class.items():
        all_samples.extend(samples)
    
    logger.info(f"Collected {len(all_samples)} samples for analysis")
    return all_samples

def extract_and_visualize_attention(model, test_samples: List[Tuple], analysis_dir: str, class_names: List[str]) -> Dict:
    """Extract and visualize attention maps."""
    logger.info("Extracting attention maps")

    from utils.attention_visualization import AttentionExtractor, visualize_attention_map

    # Create attention extractor
    device = next(model.parameters()).device
    attention_extractor = AttentionExtractor(model, device=device)

    attention_dir = os.path.join(analysis_dir, 'attention_maps')
    os.makedirs(attention_dir, exist_ok=True)

    results = {'visualizations_created': 0, 'layers_analyzed': [], 'attention_data': {}}

    try:
        # Analyze samples from each class
        for class_name in class_names:
            class_samples = [s for s in test_samples if s[2] == class_name][:5]  # First 5 per class
            results['attention_data'][class_name] = {}

            for i, (sample, target, _) in enumerate(class_samples):
                # Extract attention
                attention_maps = attention_extractor.extract_attention(sample.unsqueeze(0))
                results['attention_data'][class_name][f'sample_{i}'] = {}

                # Visualize attention for key layers
                for layer_name, attention in attention_maps.items():
                    if 'block' in layer_name:  # Focus on transformer blocks
                        # Store attention statistics
                        results['attention_data'][class_name][f'sample_{i}'][layer_name] = {
                            'shape': list(attention.shape),
                            'mean': float(attention.mean()),
                            'std': float(attention.std()),
                            'max': float(attention.max()),
                            'min': float(attention.min())
                        }

                        # Visualize first few attention heads
                        num_heads = min(4, attention.shape[1])
                        for head_idx in range(num_heads):
                            save_path = os.path.join(
                                attention_dir,
                                f'{class_name}_sample_{i}_{layer_name}_head_{head_idx}.png'
                            )
                            visualize_attention_map(
                                attention, sample.unsqueeze(0), head_idx,
                                f"{class_name} - {layer_name} - Head {head_idx}", save_path
                            )
                            results['visualizations_created'] += 1

                        if layer_name not in results['layers_analyzed']:
                            results['layers_analyzed'].append(layer_name)

    finally:
        attention_extractor.remove_hooks()

    logger.info(f"Created {results['visualizations_created']} attention visualizations")
    return results

def extract_and_analyze_activations(model, test_samples: List[Tuple], analysis_dir: str) -> Dict:
    """Extract and analyze network activations."""
    logger.info("Extracting network activations")
    
    from utils.network_flow_visualization import ActivationExtractor, visualize_activation_flow
    
    # Create activation extractor
    device = next(model.parameters()).device
    activation_extractor = ActivationExtractor(model, device=device)
    
    activation_dir = os.path.join(analysis_dir, 'activations')
    os.makedirs(activation_dir, exist_ok=True)
    
    results = {'samples_analyzed': 0}
    
    try:
        # Analyze a few representative samples
        for i, (sample, target, class_name) in enumerate(test_samples[:10]):  # First 10 samples
            activations = activation_extractor.extract_activations(sample.unsqueeze(0))
            
            # Visualize activation flow
            flow_path = os.path.join(activation_dir, f'activation_flow_{class_name}_sample_{i}.png')
            visualize_activation_flow(activations, flow_path)
            results['samples_analyzed'] += 1
    
    finally:
        activation_extractor.remove_hooks()
    
    logger.info(f"Analyzed activations for {results['samples_analyzed']} samples")
    return results

def main():
    """Main function to run the attention-focused experiment."""
    parser = argparse.ArgumentParser(description='Run Attention-Focused Continual Learning Experiment')
    parser.add_argument('--epochs', type=int, default=EXPERIMENT_CONFIG['epochs_per_task'],
                        help='Number of epochs per task')
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force CPU execution')
    parser.add_argument('--output_dir', type=str, default='results/attention_experiment',
                        help='Output directory for results')
    parser.add_argument('--methods', nargs='+', default=EXPERIMENT_CONFIG['methods'],
                        help='Methods to test')
    parser.add_argument('--seeds', nargs='+', type=int, default=EXPERIMENT_CONFIG['seeds'],
                        help='Random seeds to use')
    
    args = parser.parse_args()
    
    # Update config
    config = EXPERIMENT_CONFIG.copy()
    config['epochs_per_task'] = args.epochs
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"attention_exp_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting attention-focused experiment")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Seeds: {args.seeds}")
    
    # Run experiments
    all_results = {
        'experiment_info': {
            'name': 'attention_focused_continual_learning',
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'methods': args.methods,
            'seeds': args.seeds
        },
        'results': {}
    }

    # Define results file path
    results_file = os.path.join(output_dir, 'experiment_results.json')

    use_gpu = not args.cpu_only and torch.cuda.is_available()
    logger.info(f"Using {'GPU' if use_gpu else 'CPU'}")

    for method in args.methods:
        all_results['results'][method] = {}

        for seed in args.seeds:
            logger.info(f"Running {method} with seed {seed}")

            result = run_single_experiment_with_analysis(
                method, seed, config, output_dir, use_gpu
            )

            all_results['results'][method][seed] = result

                    # Save results incrementally
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for method in args.methods:
        print(f"\n{method.upper()}:")
        for seed in args.seeds:
            result = all_results['results'][method][seed]
            status = result['status']
            if status == 'success':
                acc = result.get('final_accuracy', {})
                cil_acc = acc.get('class_il_accuracy', 'N/A')
                til_acc = acc.get('task_il_accuracy', 'N/A')
                print(f"  Seed {seed}: ✓ CIL: {cil_acc}% TIL: {til_acc}%")
            else:
                print(f"  Seed {seed}: ✗ {status}")
    
    print(f"\nResults saved to: {results_file}")
    print("="*60)

if __name__ == "__main__":
    main()
