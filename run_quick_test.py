#!/usr/bin/env python3
"""
Quick test script to run a minimal version of the attention experiment.
This runs with very few epochs to quickly verify the full pipeline works.
"""

import os
import sys
import subprocess
import json
import torch
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_quick_test():
    """Run a quick test with minimal epochs."""
    
    print("=" * 60)
    print("QUICK ATTENTION EXPERIMENT TEST")
    print("=" * 60)
    
    # Test configuration - minimal for speed
    config = {
        "methods": ['derpp'],  # Test with just one method
        "seeds": [42],  # Test with just one seed
        "epochs_per_task": 2,  # Very few epochs for quick test
        "batch_size": 16,  # Smaller batch for speed
        "lr": 0.01,
        "optimizer": "sgd",
        "num_workers": 0,  # No parallel loading for simplicity
        "dataset": "seq-cifar10-224-custom",
        "backbone": "vit",
    }
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"results/quick_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Running quick test with output dir: {output_dir}")
    
    method = config['methods'][0]
    seed = config['seeds'][0]
    
    # Create experiment directory
    exp_dir = os.path.join(output_dir, f"{method}_seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Build command
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
        '--savecheck', 'last',  # Save model for analysis
        # Note: Don't set --device cpu, framework will auto-detect
        # DER++ specific parameters
        '--buffer_size', '50',
        '--alpha', '0.1',
        '--beta', '0.5',
    ]
    
    try:
        logger.info("Running mammoth experiment...")
        logger.info(f"Command: {' '.join(cmd)}")

        # Start the process with real-time output
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
        print("Training progress:")

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Show important progress lines
                if any(keyword in output.lower() for keyword in ['epoch', 'task', 'accuracy', 'loss', 'progress']):
                    print(f"  {output.strip()}")

        # Wait for process to complete
        return_code = process.poll()

        if return_code != 0:
            logger.error(f"Experiment failed with return code {return_code}")
            logger.error(f"Last few lines of output:")
            for line in output_lines[-10:]:
                logger.error(f"  {line}")
            print("✗ Mammoth experiment failed")
            return False

        logger.info("Mammoth experiment completed successfully")
        print("✓ Mammoth experiment completed")
        
        # Check if model was saved
        model_path = os.path.join(exp_dir, 'model.pth')
        if os.path.exists(model_path):
            print("✓ Model saved successfully")
        else:
            print("⚠️  Model file not found")
            return False
        
        # Test attention analysis
        print("Testing attention analysis...")
        success = test_attention_analysis(exp_dir, config)
        
        if success:
            print("✓ Attention analysis completed")
            print(f"\n🎉 Quick test passed! Results in: {output_dir}")
            print("\nYou can now run the full experiment with:")
            print("python run_attention_experiment.py")
            return True
        else:
            print("✗ Attention analysis failed")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Experiment timed out")
        print("✗ Experiment timed out")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"✗ Unexpected error: {str(e)}")
        return False

def test_attention_analysis(exp_dir: str, config: dict) -> bool:
    """Test the attention analysis pipeline."""
    
    try:
        # Import required modules
        from datasets import get_dataset
        from models import get_model
        from backbone import get_backbone
        from utils.attention_visualization import AttentionExtractor
        from utils.network_flow_visualization import ActivationExtractor
        from argparse import Namespace
        
        # Create args for loading model
        args = Namespace(
            dataset=config['dataset'],
            backbone=config['backbone'],
            seed=config['seeds'][0],
            device='cpu',
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
        
        # Load trained weights
        model_path = os.path.join(exp_dir, 'model.pth')
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        
        print("  ✓ Model loaded successfully")
        
        # Create analysis directory
        analysis_dir = os.path.join(exp_dir, 'attention_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Get a few test samples
        _, test_loader = dataset.get_data_loaders()
        test_samples = []
        class_names = dataset.get_class_names()
        
        for batch_data, batch_targets, _ in test_loader:
            for sample, target in zip(batch_data[:2], batch_targets[:2]):  # Just 2 samples
                class_idx = target.item()
                if class_idx < len(class_names):
                    class_name = class_names[class_idx]
                    test_samples.append((sample, target, class_name))
            if len(test_samples) >= 2:
                break
        
        print(f"  ✓ Collected {len(test_samples)} test samples")
        
        # Test attention extraction
        attention_extractor = AttentionExtractor(model, device='cpu')
        
        for i, (sample, target, class_name) in enumerate(test_samples):
            attention_maps = attention_extractor.extract_attention(sample.unsqueeze(0))
            print(f"  ✓ Extracted attention for sample {i} ({class_name}): {len(attention_maps)} layers")
            
            # Test visualization for one layer
            if attention_maps:
                layer_name = list(attention_maps.keys())[0]
                attention = attention_maps[layer_name]
                
                # Simple test - just check shapes
                if len(attention.shape) >= 3:
                    print(f"    ✓ Attention shape: {attention.shape}")
                else:
                    print(f"    ⚠️  Unexpected attention shape: {attention.shape}")
        
        attention_extractor.remove_hooks()
        
        # Test activation extraction
        activation_extractor = ActivationExtractor(model, device='cpu')
        
        sample, _, class_name = test_samples[0]
        activations = activation_extractor.extract_activations(sample.unsqueeze(0))
        print(f"  ✓ Extracted activations: {len(activations)} layers")
        
        activation_extractor.remove_hooks()
        
        # Save a simple analysis report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'num_samples_analyzed': len(test_samples),
            'attention_layers': len(attention_maps) if 'attention_maps' in locals() else 0,
            'activation_layers': len(activations) if 'activations' in locals() else 0,
            'class_names': class_names,
            'status': 'success'
        }
        
        report_path = os.path.join(analysis_dir, 'analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✓ Analysis report saved: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in attention analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    
    # First run the setup test
    print("Running setup verification...")
    try:
        import test_attention_setup
        # Run a minimal version of the setup test
        if not test_attention_setup.test_custom_dataset():
            print("✗ Setup test failed - dataset not working")
            return
        if not test_attention_setup.test_vit_backbone():
            print("✗ Setup test failed - backbone not working")
            return
        print("✓ Basic setup verified")
    except Exception as e:
        print(f"⚠️  Could not run setup test: {e}")
        print("Proceeding with quick test anyway...")
    
    print()
    
    # Run the quick test
    success = run_quick_test()
    
    if success:
        print("\n" + "=" * 60)
        print("QUICK TEST SUMMARY: ✓ PASSED")
        print("=" * 60)
        print("The attention experiment pipeline is working correctly.")
        print("You can now run the full experiment with confidence.")
    else:
        print("\n" + "=" * 60)
        print("QUICK TEST SUMMARY: ✗ FAILED")
        print("=" * 60)
        print("Please check the errors above and fix any issues.")

if __name__ == "__main__":
    main()
