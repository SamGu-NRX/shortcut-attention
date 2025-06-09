#!/usr/bin/env python3
"""
Quick test script with enhanced progress tracking and time estimation.
"""

import os
import sys
import subprocess
import json
import torch
import time
from datetime import datetime, timedelta
import logging
import re
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track training progress and estimate completion time."""
    
    def __init__(self, total_epochs: int, total_tasks: int = 2):
        self.total_epochs = total_epochs
        self.total_tasks = total_tasks
        self.total_steps = total_epochs * total_tasks
        
        self.start_time = time.time()
        self.current_task = 0
        self.current_epoch = 0
        self.completed_steps = 0
        
        self.epoch_times = []
        self.last_epoch_start = None
        
    def update_from_line(self, line: str) -> bool:
        """Update progress from a log line. Returns True if progress was updated."""
        updated = False
        
        # Look for task information
        task_match = re.search(r'Task (\d+)', line)
        if task_match:
            new_task = int(task_match.group(1))
            if new_task != self.current_task:
                self.current_task = new_task
                self.current_epoch = 0
                updated = True
        
        # Look for epoch information
        epoch_match = re.search(r'Epoch (\d+)', line)
        if epoch_match:
            new_epoch = int(epoch_match.group(1))
            if new_epoch != self.current_epoch:
                # Record epoch completion time
                if self.last_epoch_start is not None:
                    epoch_duration = time.time() - self.last_epoch_start
                    self.epoch_times.append(epoch_duration)
                
                self.current_epoch = new_epoch
                # Calculate completed steps more accurately
                self.completed_steps = self.current_task * self.total_epochs + self.current_epoch
                # Cap at total steps to avoid >100% progress
                self.completed_steps = min(self.completed_steps, self.total_steps)
                self.last_epoch_start = time.time()
                updated = True
        
        return updated
    
    def get_progress_info(self) -> dict:
        """Get current progress information."""
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.completed_steps / self.total_steps) * 100
        
        # Estimate remaining time
        if self.completed_steps > 0 and self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_steps = self.total_steps - self.completed_steps
            estimated_remaining = remaining_steps * avg_epoch_time
            estimated_total = elapsed_time + estimated_remaining
        else:
            estimated_remaining = None
            estimated_total = None
        
        return {
            'current_task': self.current_task,
            'current_epoch': self.current_epoch,
            'completed_steps': self.completed_steps,
            'total_steps': self.total_steps,
            'progress_percent': progress_percent,
            'elapsed_time': elapsed_time,
            'estimated_remaining': estimated_remaining,
            'estimated_total': estimated_total,
            'avg_epoch_time': sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else None
        }
    
    def format_time(self, seconds: Optional[float]) -> str:
        """Format time in a human-readable way."""
        if seconds is None:
            return "Unknown"
        
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def print_progress(self):
        """Print current progress information."""
        info = self.get_progress_info()
        
        progress_bar = "█" * int(info['progress_percent'] / 5) + "░" * (20 - int(info['progress_percent'] / 5))
        
        print(f"\r[{progress_bar}] {info['progress_percent']:.1f}% | "
              f"Task {info['current_task']}/{self.total_tasks} | "
              f"Epoch {info['current_epoch']}/{self.total_epochs} | "
              f"Elapsed: {self.format_time(info['elapsed_time'])} | "
              f"ETA: {self.format_time(info['estimated_remaining'])}", end="", flush=True)

def run_quick_test_with_progress():
    """Run a quick test with enhanced progress tracking."""
    
    print("=" * 60)
    print("QUICK ATTENTION EXPERIMENT TEST WITH PROGRESS")
    print("=" * 60)
    
    # Test configuration - minimal for speed
    config = {
        "method": 'derpp',
        "seed": 42,
        "epochs_per_task": 3,  # Very few epochs for quick test
        "batch_size": 16,
        "lr": 0.01,
        "optimizer": "sgd",
        "num_workers": 0,
        "dataset": "seq-cifar10-224-custom",
        "backbone": "vit",
    }
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"results/quick_test_progress_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Running quick test with progress tracking")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Configuration: {config}")
    
    # Create experiment directory
    exp_dir = os.path.join(output_dir, f"{config['method']}_seed_{config['seed']}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, 'main.py',
        '--dataset', config['dataset'],
        '--model', config['method'],
        '--backbone', config['backbone'],
        '--results_path', exp_dir,
        '--seed', str(config['seed']),
        '--n_epochs', str(config['epochs_per_task']),
        '--batch_size', str(config['batch_size']),
        '--lr', str(config['lr']),
        '--optimizer', config['optimizer'],
        '--optim_wd', '0.0',
        '--optim_mom', '0.0',
        '--num_workers', str(config['num_workers']),
        '--drop_last', '0',
        '--debug_mode', '0',
        '--savecheck', 'last',
        # DER++ specific parameters
        '--buffer_size', '50',
        '--alpha', '0.1',
        '--beta', '0.5',
    ]
    
    try:
        logger.info(f"Starting mammoth experiment...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(config['epochs_per_task'], total_tasks=2)
        
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
        print("\nTraining started...")
        print("Progress will be shown below:")
        print()
        
        last_progress_update = time.time()
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                
                # Update progress tracker
                if progress_tracker.update_from_line(output):
                    progress_tracker.print_progress()
                
                # Show important lines
                if any(keyword in output.lower() for keyword in ['accuracy', 'loss', 'task', 'epoch']):
                    # Clear progress line and show the log
                    print(f"\n  {output.strip()}")
                    progress_tracker.print_progress()
                
                # Update progress every few seconds even without new info
                if time.time() - last_progress_update > 3:
                    progress_tracker.print_progress()
                    last_progress_update = time.time()
        
        # Final progress update
        print("\n")  # New line after progress bar
        
        # Wait for process to complete
        return_code = process.poll()
        full_output = '\n'.join(output_lines)
        
        if return_code != 0:
            logger.error(f"Experiment failed with return code {return_code}")
            logger.error(f"Last few lines of output:")
            for line in output_lines[-10:]:
                logger.error(f"  {line}")
            print("✗ Mammoth experiment failed")
            return False
        
        # Show final progress
        final_info = progress_tracker.get_progress_info()
        print("✓ Mammoth experiment completed successfully!")
        print(f"  Total time: {progress_tracker.format_time(final_info['elapsed_time'])}")
        if final_info['avg_epoch_time']:
            print(f"  Average epoch time: {progress_tracker.format_time(final_info['avg_epoch_time'])}")
        
        # Check if model was saved (mammoth saves in checkpoints/ subdirectory)
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

        if model_path and os.path.exists(model_path):
            print(f"✓ Model saved successfully: {os.path.basename(model_path)}")
        else:
            print("⚠️  Model file not found")
            print(f"  Checked: {exp_dir}")
            print(f"  Checked: {checkpoint_dir}")
            # List what files are actually there
            if os.path.exists(exp_dir):
                files = os.listdir(exp_dir)
                print(f"  Files in exp_dir: {files}")
            return False
        
        # Test attention analysis
        print("\nTesting attention analysis...")
        success = test_attention_analysis_quick(exp_dir, config)
        
        if success:
            print("✓ Attention analysis completed")
            print(f"\n🎉 Quick test with progress tracking passed!")
            print(f"Results in: {output_dir}")
            print("\nYou can now run the full experiment with:")
            print("python run_attention_experiment.py")
            return True
        else:
            print("✗ Attention analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_analysis_quick(exp_dir: str, config: dict) -> bool:
    """Quick test of attention analysis pipeline."""
    
    try:
        # Import required modules
        from datasets import get_dataset
        from models import get_model
        from backbone import get_backbone
        from utils.attention_visualization import AttentionExtractor
        from argparse import Namespace
        
        # Create args for loading model
        args = Namespace(
            dataset=config['dataset'],
            backbone=config['backbone'],
            seed=config['seed'],
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
        
        # Find the model checkpoint
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
            print(f"  ✗ Model file not found in {exp_dir}")
            return False

        # Load trained weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        
        print("  ✓ Model loaded successfully")
        
        # Get a test sample
        _, test_loader = dataset.get_data_loaders()
        data_iter = iter(test_loader)
        data, targets, _ = next(data_iter)
        sample = data[0:1]  # First sample
        
        # Test attention extraction
        attention_extractor = AttentionExtractor(model, device='cpu')
        attention_maps = attention_extractor.extract_attention(sample)
        attention_extractor.remove_hooks()
        
        print(f"  ✓ Extracted attention maps: {len(attention_maps)} layers")
        
        # Save a simple report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'attention_layers': len(attention_maps),
            'class_names': dataset.get_class_names(),
            'status': 'success'
        }
        
        report_path = os.path.join(exp_dir, 'quick_analysis_report.json')
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
    
    # First run basic setup verification
    print("Running basic setup verification...")
    try:
        import test_attention_setup
        if not test_attention_setup.test_custom_dataset():
            print("✗ Setup test failed - dataset not working")
            return
        print("✓ Basic setup verified")
    except Exception as e:
        print(f"⚠️  Could not run setup test: {e}")
        print("Proceeding with quick test anyway...")
    
    print()
    
    # Run the quick test with progress
    success = run_quick_test_with_progress()
    
    if success:
        print("\n" + "=" * 60)
        print("QUICK TEST WITH PROGRESS: ✓ PASSED")
        print("=" * 60)
        print("The attention experiment pipeline is working correctly.")
        print("You can now run the full experiment with confidence.")
    else:
        print("\n" + "=" * 60)
        print("QUICK TEST WITH PROGRESS: ✗ FAILED")
        print("=" * 60)
        print("Please check the errors above and fix any issues.")

if __name__ == "__main__":
    main()
