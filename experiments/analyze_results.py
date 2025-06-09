"""Analyze and visualize results from the shortcut investigation experiments."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
import logging

from utils.attention_visualization import AttentionAnalyzer, analyze_task_attention
from utils.network_flow_visualization import ActivationExtractor
from datasets import get_dataset
from backbone import get_backbone
from models import get_model

def analyze_model_attention(args, model_name, results_dir):
    """Analyze attention patterns for a trained model."""
    # Load model checkpoint
    model_dir = os.path.join(results_dir, model_name)
    if not os.path.exists(model_dir):
        logging.error(f"No results found for model {model_name}")
        return
        
    checkpoints = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
    if not checkpoints:
        logging.error(f"No checkpoints found for model {model_name}")
        return
        
    # Get dataset and model
    dataset = get_dataset(args)
    args.num_classes = dataset.N_CLASSES
    backbone = get_backbone(args)
    loss = dataset.get_loss()
    transform = dataset.get_transform()
    
    # Create model
    model = get_model(args, backbone, loss, transform)
    
    # Analyze attention after each task
    for task_id, ckpt in enumerate(checkpoints):
        checkpoint_path = os.path.join(model_dir, ckpt)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        # Create output directory for attention maps
        attention_dir = os.path.join(model_dir, f'attention_task_{task_id}')
        os.makedirs(attention_dir, exist_ok=True)
        
        # Analyze attention patterns for all tasks seen so far
        for analyzed_task in range(task_id + 1):
            # Get test data for the task
            dataset.test_loader = dataset.get_test_loader(analyzed_task)
            task_dir = os.path.join(attention_dir, f'test_task_{analyzed_task}')
            os.makedirs(task_dir, exist_ok=True)
            
            # Analyze attention patterns
            class_samples = analyze_task_attention(
                model=model,
                dataset=dataset,
                device=args.device,
                save_dir=task_dir
            )
            
            # Check network flow
            extractor = ActivationExtractor(model, device=args.device)
            
            # Get one sample per class
            for class_idx, samples in class_samples.items():
                # Get activations
                activations = extractor.extract_activations(samples['input'])
                
                # Save activation visualization
                save_path = os.path.join(task_dir, f'activations_class_{class_idx}.png')
                extractor.visualize_activations(activations, save_path=save_path)
            
            extractor.remove_hooks()

def main():
    """Main function to analyze experiment results."""
    parser = ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing experiment results')
    parser.add_argument('--models', nargs='+', default=['derpp', 'ewc_on'],
                      help='Models to analyze')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run analysis on')
                      
    # Add dataset args
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset to analyze')
    parser.add_argument('--backbone', type=str, default='vit',
                      help='Backbone architecture')
    parser.add_argument('--transform_type', type=str, default='weak',
                      help='Type of data augmentation')
    
    # Model hyperparameters
    parser.add_argument('--buffer_size', type=int, default=500,
                      help='Buffer size for rehearsal models')
    parser.add_argument('--alpha', type=float, default=0.1,
                      help='Alpha parameter for DER++')
    parser.add_argument('--beta', type=float, default=0.5,
                      help='Beta parameter for DER++')
    
    args = parser.parse_args()
    
    # Create output directory for analysis
    analysis_dir = os.path.join(args.results_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Analyze each model
    for model_name in args.models:
        print(f"\nAnalyzing model: {model_name}")
        args.model = model_name
        analyze_model_attention(args, model_name, args.results_dir)

if __name__ == "__main__":
    main()
