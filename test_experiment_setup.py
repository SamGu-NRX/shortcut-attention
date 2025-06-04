#!/usr/bin/env python3
"""
Test script to verify the shortcut investigation experiment setup.
"""

import os
import sys
import torch
import numpy as np
from argparse import Namespace

def test_dataset():
    """Test if the custom dataset loads correctly."""
    print("Testing custom dataset...")

    try:
        from datasets import get_dataset

        # Create test arguments
        args = Namespace(
            dataset='seq-cifar10-custom',
            seed=42,
            custom_class_order=None,
            permute_classes=False,
            label_perc=1,
            label_perc_by_class=1,
            validation=None,
            base_path='./data/',
            transform_type='weak',  # Add missing argument
            joint=False,  # Add missing argument
            start_from=None,
            stop_after=None,
            enable_other_metrics=False,
            eval_future=False,
            noise_rate=0.0,
        )
        
        # Load dataset
        dataset = get_dataset(args)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Name: {dataset.NAME}")
        print(f"  - Setting: {dataset.SETTING}")
        print(f"  - Classes per task: {dataset.N_CLASSES_PER_TASK}")
        print(f"  - Number of tasks: {dataset.N_TASKS}")
        print(f"  - Total classes: {dataset.N_CLASSES}")
        print(f"  - Class names: {dataset.get_class_names()}")
        if hasattr(dataset, 'get_task_labels'):
            print(f"  - Task labels: {dataset.get_task_labels()}")
        else:
            print(f"  - Task labels: Not available")
        
        # Test data loaders
        train_loader, test_loader = dataset.get_data_loaders()
        print(f"  - Train loader: {len(train_loader)} batches")
        print(f"  - Test loader: {len(test_loader)} batches")
        
        # Test a batch
        for batch_idx, (data, targets, not_aug_data) in enumerate(train_loader):
            print(f"  - Batch shape: {data.shape}")
            print(f"  - Target shape: {targets.shape}")
            print(f"  - Unique targets in batch: {torch.unique(targets).tolist()}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def test_backbone():
    """Test if the ViT backbone loads correctly."""
    print("\nTesting ViT backbone...")

    try:
        from backbone import get_backbone

        # Create test arguments - use standard ViT but with custom parameters
        args = Namespace(
            backbone='vit',
            num_classes=4,  # Our custom dataset has 4 classes
            pretrained=False,  # Add missing argument
            pretrain_type='in21k-ft-in1k',  # Add missing argument
        )

        # Load backbone
        backbone = get_backbone(args)

        print(f"✓ Backbone loaded successfully")
        print(f"  - Type: {type(backbone).__name__}")
        print(f"  - Number of parameters: {sum(p.numel() for p in backbone.parameters()):,}")

        # Test forward pass - resize to ViT expected size
        test_input = torch.randn(2, 3, 224, 224)  # ViT expected size
        with torch.no_grad():
            output = backbone(test_input)
        
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backbone test failed: {e}")
        return False

def test_models():
    """Test if the continual learning models load correctly."""
    print("\nTesting continual learning models...")
    
    try:
        from models import get_model
        from backbone import get_backbone
        from datasets import get_dataset
        
        # Create test arguments
        args = Namespace(
            dataset='seq-cifar10-custom',
            backbone='vit',
            num_classes=4,
            seed=42,
            custom_class_order=None,
            permute_classes=False,
            label_perc=1,
            label_perc_by_class=1,
            validation=None,
            base_path='./data/',
            transform_type='weak',  # Add missing argument
            pretrained=False,  # Add missing argument
            pretrain_type='in21k-ft-in1k',  # Add missing argument
            joint=False,  # Add missing argument
            start_from=None,
            stop_after=None,
            enable_other_metrics=False,
            eval_future=False,
            noise_rate=0.0,

            device='cpu',  # Use CPU for testing
            # DER++ specific
            buffer_size=50,
            alpha=0.1,
            beta=0.5,
        )
        
        # Test DER++
        print("  Testing DER++...")
        dataset = get_dataset(args)
        backbone = get_backbone(args)
        loss = dataset.get_loss()
        transform = dataset.get_transform()
        
        from models.derpp import Derpp
        model = Derpp(backbone, loss, args, transform, dataset=dataset)
        
        print(f"    ✓ DER++ model created successfully")
        print(f"    - Buffer size: {model.buffer.buffer_size}")
        
        # Test EWC
        print("  Testing EWC...")
        args.e_lambda = 0.4
        args.gamma = 0.85
        
        from models.ewc_on import EwcOn
        model_ewc = EwcOn(backbone, loss, args, transform, dataset=dataset)
        
        print(f"    ✓ EWC model created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_visualization_utils():
    """Test if the visualization utilities work."""
    print("\nTesting visualization utilities...")
    
    try:
        from utils.attention_visualization import AttentionExtractor
        from utils.network_flow_visualization import ActivationExtractor
        
        # Create a simple ViT model for testing
        from backbone import get_backbone
        from datasets import get_dataset
        from models.derpp import Derpp

        # Create proper model with .net attribute
        args = Namespace(
            backbone='vit',
            num_classes=4,
            pretrained=False,
            pretrain_type='in21k-ft-in1k',
            dataset='seq-cifar10-custom',
            seed=42,
            custom_class_order=None,
            permute_classes=False,
            label_perc=1,
            label_perc_by_class=1,
            validation=None,
            base_path='./data/',
            transform_type='weak',
            joint=False,
            start_from=None,
            stop_after=None,
            enable_other_metrics=False,
            eval_future=False,
            noise_rate=0.0,

            device='cpu',
            buffer_size=50,
            alpha=0.1,
            beta=0.5,
        )

        dataset = get_dataset(args)
        backbone = get_backbone(args)
        loss = dataset.get_loss()
        transform = dataset.get_transform()
        model = Derpp(backbone, loss, args, transform, dataset=dataset)

        # Test attention extractor
        attention_extractor = AttentionExtractor(model, device='cpu')
        print(f"    ✓ AttentionExtractor created")
        print(f"    - Number of hooks registered: {len(attention_extractor.hooks)}")

        # Test activation extractor
        activation_extractor = ActivationExtractor(model, device='cpu')
        print(f"    ✓ ActivationExtractor created")
        print(f"    - Number of hooks registered: {len(activation_extractor.hooks)}")

        # Test with dummy input - resize to ViT expected size
        dummy_input = torch.randn(1, 3, 224, 224)

        # Test attention extraction
        attention_maps = attention_extractor.extract_attention(dummy_input)
        print(f"    ✓ Attention extraction successful")
        print(f"    - Number of attention maps: {len(attention_maps)}")

        # Test activation extraction
        activations = activation_extractor.extract_activations(dummy_input)
        print(f"    ✓ Activation extraction successful")
        print(f"    - Number of activation maps: {len(activations)}")

        # Clean up
        attention_extractor.remove_hooks()
        activation_extractor.remove_hooks()
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def test_experiment_runner():
    """Test if the experiment runner can be imported."""
    print("\nTesting experiment runner...")
    
    try:
        from experiments.shortcut_investigation import ShortcutInvestigationExperiment
        
        # Create test experiment
        base_args = {
            'n_epochs': 1,  # Very short for testing
            'batch_size': 4,
            'lr': 0.01,
            'device': 'cpu',
            'debug_mode': 1,
        }
        
        experiment = ShortcutInvestigationExperiment(
            base_args=base_args,
            experiment_name="test_experiment"
        )
        
        print(f"    ✓ Experiment runner created successfully")
        print(f"    - Results directory: {experiment.results_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Experiment runner test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SHORTCUT INVESTIGATION EXPERIMENT - SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Dataset", test_dataset),
        ("Backbone", test_backbone),
        ("Models", test_models),
        ("Visualization", test_visualization_utils),
        ("Experiment Runner", test_experiment_runner),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The experiment setup is ready.")
        print("\nNext steps:")
        print("1. Run: python run_shortcut_experiment.py --comparison")
        print("2. Or run individual experiments with specific methods")
        print("3. Analyze results with experiments/analyze_results.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
