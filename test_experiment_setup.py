# test_experiment_setup.py
#!/usr/bin/env python3
"""
Test script to verify the shortcut investigation experiment setup.
"""

import os
import sys
import torch
import numpy as np
from argparse import Namespace
import logging

logging.basicConfig(level=logging.INFO)


def test_dataset():
    """Test if the custom dataset loads correctly."""
    print("Testing custom dataset...")

    try:
        from datasets import get_dataset

        args = Namespace(
            dataset='seq-cifar10-custom',
            seed=42,
            custom_class_order=None, # Let dataset default this
            permute_classes=False,
            custom_task_order=None,
            label_perc=1,
            label_perc_by_class=1,
            validation=None, 
            validation_mode='class-il', # Changed to a valid mode
            base_path='./data/',
            transform_type='weak',
            joint=False,
            start_from=None,
            stop_after=None,
            enable_other_metrics=False,
            eval_future=False,
            noise_rate=0.0,
        )
        
        dataset = get_dataset(args)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Name: {dataset.NAME}")
        print(f"  - Setting: {dataset.SETTING}")
        print(f"  - Classes per task: {dataset.N_CLASSES_PER_TASK}")
        print(f"  - Number of tasks: {dataset.N_TASKS}")
        print(f"  - Total classes: {dataset.N_CLASSES}")
        if hasattr(dataset, 'class_order') and dataset.class_order is not None: # Check before printing
            print(f"  - Class order (indices): {dataset.class_order}")
        else:
            print(f"  - Class order (indices): Not set by superclass")
        print(f"  - Class names: {dataset.get_class_names()}")
        if hasattr(dataset, 'get_task_labels'):
            print(f"  - Task labels: {dataset.get_task_labels()}")
        else:
            print(f"  - Task labels: Not available")
        
        train_loader, test_loader = dataset.get_data_loaders()
        print(f"  - Train loader: {len(train_loader)} batches")
        print(f"  - Test loader: {len(test_loader)} batches")
        
        for batch_idx, (data, targets, not_aug_data) in enumerate(train_loader):
            print(f"  - Batch shape: {data.shape}")
            print(f"  - Target shape: {targets.shape}")
            print(f"  - Unique targets in batch: {torch.unique(targets).tolist()}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc() 
        return False

def test_backbone():
    """Test if the ViT backbone loads correctly."""
    print("\nTesting ViT backbone...")
    try:
        from backbone import get_backbone
        args = Namespace(
            backbone='vit', num_classes=4, pretrained=False, pretrain_type='in21k-ft-in1k')
        backbone = get_backbone(args)
        print(f"✓ Backbone loaded successfully")
        print(f"  - Type: {type(backbone).__name__}")
        print(f"  - Number of parameters: {sum(p.numel() for p in backbone.parameters()):,}")
        test_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad(): output = backbone(test_input)
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Backbone test failed: {e}")
        import traceback; traceback.print_exc()
        return False

def test_models():
    """Test if the continual learning models load correctly."""
    print("\nTesting continual learning models...")
    try:
        from models import get_model
        from backbone import get_backbone
        from datasets import get_dataset
        
        args = Namespace(
            dataset='seq-cifar10-custom',
            backbone='vit',
            seed=42,
            custom_class_order=None, 
            permute_classes=False,
            custom_task_order=None,
            label_perc=1,
            label_perc_by_class=1,
            validation=None,
            validation_mode='class-il', # Changed
            base_path='./data/',
            transform_type='weak',
            pretrained=False,
            pretrain_type='in21k-ft-in1k',
            joint=False,
            start_from=None,
            stop_after=None,
            enable_other_metrics=False,
            eval_future=False,
            noise_rate=0.0,
            lr=0.01,
            optimizer='sgd', 
            optim_wd=0.0, # Added weight decay
            optim_mom=0.0, # Added momentum (often needed with optim_wd)
            optim_nesterov=False, # Added nesterov (often needed with optim_wd)
            device='cpu',
            model='derpp', 
            buffer_size=50,
            alpha=0.1,
            beta=0.5,
            e_lambda=0.4, 
            gamma=0.85,
        )
        
        print("  Testing DER++...")
        dataset = get_dataset(args) 
        args.num_classes = dataset.N_CLASSES 
        backbone_net = get_backbone(args) 
        loss = dataset.get_loss()
        transform = dataset.get_transform()
        
        args.model = 'derpp'
        model_derpp = get_model(args, backbone_net, loss, transform, dataset=dataset)
        print(f"    ✓ DER++ model created successfully")
        print(f"    - Buffer size: {model_derpp.buffer.buffer_size}")
        
        print("  Testing EWC...")
        args.model = 'ewc_on'
        backbone_net_ewc = get_backbone(args) # New instance for EWC
        model_ewc = get_model(args, backbone_net_ewc, loss, transform, dataset=dataset)
        print(f"    ✓ EWC model created successfully")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback; traceback.print_exc()
        return False

def test_visualization_utils():
    """Test if the visualization utilities work."""
    print("\nTesting visualization utilities...")
    try:
        from utils.attention_visualization import AttentionExtractor
        from utils.network_flow_visualization import ActivationExtractor
        from backbone import get_backbone 
        from datasets import get_dataset 
        from models import get_model

        args = Namespace(
            backbone='vit',
            pretrained=False,
            pretrain_type='in21k-ft-in1k',
            dataset='seq-cifar10-custom',
            seed=42,
            custom_class_order=None, 
            permute_classes=False,
            custom_task_order=None,
            label_perc=1,
            label_perc_by_class=1,
            validation=None,
            validation_mode='class-il', # Changed
            base_path='./data/',
            transform_type='weak',
            joint=False,
            start_from=None,
            stop_after=None,
            enable_other_metrics=False,
            eval_future=False,
            noise_rate=0.0,
            lr=0.01, 
            optimizer='sgd',
            optim_wd=0.0, # Added
            optim_mom=0.0, # Added
            optim_nesterov=False, # Added
            device='cpu',
            model='derpp', 
            buffer_size=50, 
            alpha=0.1,      
            beta=0.5,
        )

        dataset = get_dataset(args)
        args.num_classes = dataset.N_CLASSES 
        backbone_net = get_backbone(args) 
        loss = dataset.get_loss()
        transform = dataset.get_transform()
        model_instance = get_model(args, backbone_net, loss, transform, dataset=dataset)

        attention_extractor = AttentionExtractor(model_instance, device='cpu')
        print(f"    ✓ AttentionExtractor created")
        activation_extractor = ActivationExtractor(model_instance, device='cpu')
        print(f"    ✓ ActivationExtractor created")
        dummy_input = torch.randn(1, 3, 224, 224)
        attention_maps = attention_extractor.extract_attention(dummy_input)
        print(f"    ✓ Attention extraction successful")
        print(f"    - Number of attention maps: {len(attention_maps)}")
        activations = activation_extractor.extract_activations(dummy_input)
        print(f"    ✓ Activation extraction successful")
        print(f"    - Number of activation maps: {len(activations)}")
        attention_extractor.remove_hooks()
        activation_extractor.remove_hooks()
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        import traceback; traceback.print_exc()
        return False

def test_experiment_runner():
    """Test if the experiment runner can be imported."""
    print("\nTesting experiment runner...")
    try:
        from experiments.shortcut_investigation import ShortcutInvestigationExperiment
        base_args = {
            'n_epochs': 1, 'batch_size': 4, 'lr': 0.01, 'device': 'cpu',
            'debug_mode': 1, 'permute_classes': False, 'custom_class_order': None,
            'custom_task_order': None, 'validation_mode': 'class-il', 'optimizer': 'sgd',
            'optim_wd': 0.0, 'optim_mom': 0.0, 'optim_nesterov': False, # Added
        }
        experiment = ShortcutInvestigationExperiment(
            base_args=base_args, experiment_name="test_experiment")
        print(f"    ✓ Experiment runner created successfully")
        print(f"    - Results directory: {experiment.results_dir}")
        return True
    except Exception as e:
        print(f"✗ Experiment runner test failed: {e}")
        import traceback; traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("SHORTCUT INVESTIGATION EXPERIMENT - SETUP TEST")
    print("=" * 60)
    tests = [
        ("Dataset", test_dataset), ("Backbone", test_backbone),
        ("Models", test_models), ("Visualization", test_visualization_utils),
        ("Experiment Runner", test_experiment_runner),
    ]
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
            import traceback; traceback.print_exc()
    print("\n" + "=" * 60 + "\nTEST SUMMARY\n" + "=" * 60)
    passed_count = sum(1 for result in results.values() if result) 
    total_count = len(results) 
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    if passed_count == total_count:
        print("\n🎉 All tests passed! The experiment setup is ready.")
        print("\nNext steps:\n1. Run: python run_shortcut_experiment.py --comparison")
        print("2. Or run individual experiments with specific methods")
        print("3. Analyze results with experiments/analyze_results.py")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()