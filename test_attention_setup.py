#!/usr/bin/env python3
"""
Test script to verify the attention-focused experiment setup.
This script tests all components needed for the shortcut investigation experiment.
"""

import os
import sys
import torch
import numpy as np
from argparse import Namespace
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reproducible_initialization():
    """Test that reproducible initialization works correctly."""
    print("Testing reproducible initialization...")
    
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Test with same seed
    seed = 42
    set_seed(seed)
    tensor1 = torch.randn(3, 3)
    random1 = np.random.random(5)
    
    set_seed(seed)
    tensor2 = torch.randn(3, 3)
    random2 = np.random.random(5)
    
    assert torch.allclose(tensor1, tensor2), "PyTorch random state not reproducible"
    assert np.allclose(random1, random2), "NumPy random state not reproducible"
    
    print("✓ Reproducible initialization working correctly")
    return True

def test_custom_dataset():
    """Test the custom CIFAR-10 dataset."""
    print("Testing custom CIFAR-10 dataset...")
    
    try:
        from datasets import get_dataset
        
        args = Namespace(
            dataset='seq-cifar10-224-custom',
            seed=42,
            custom_class_order=None,
            permute_classes=False,
            custom_task_order=None,
            label_perc=1,
            label_perc_by_class=1,
            validation=None,
            validation_mode='current',
            base_path='./data/',
            transform_type='weak',
            joint=False,
            start_from=None,
            stop_after=None,
            enable_other_metrics=False,
            eval_future=False,
            noise_rate=0.0,
            batch_size=32,
            drop_last=False,
            num_workers=0
        )
        
        dataset = get_dataset(args)
        
        print(f"✓ Dataset loaded: {dataset.NAME}")
        print(f"  - Classes per task: {dataset.N_CLASSES_PER_TASK}")
        print(f"  - Number of tasks: {dataset.N_TASKS}")
        print(f"  - Total classes: {dataset.N_CLASSES}")
        
        class_names = dataset.get_class_names()
        print(f"  - Class names: {class_names}")
        
        task_labels = dataset.get_task_labels()
        print(f"  - Task labels: {task_labels}")
        
        # Verify the expected structure
        expected_classes = ['airplane', 'automobile', 'bird', 'truck']
        expected_tasks = {0: ['airplane', 'automobile'], 1: ['bird', 'truck']}
        
        assert class_names == expected_classes, f"Expected {expected_classes}, got {class_names}"
        assert task_labels == expected_tasks, f"Expected {expected_tasks}, got {task_labels}"
        
        # Test data loaders
        train_loader, test_loader = dataset.get_data_loaders()
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # Test a batch
        data_iter = iter(train_loader)
        data, targets, not_aug_data = next(data_iter)
        print(f"  - Batch shape: {data.shape}")
        print(f"  - Target shape: {targets.shape}")
        print(f"  - Unique targets: {torch.unique(targets).tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vit_backbone():
    """Test ViT backbone loading."""
    print("Testing ViT backbone...")
    
    try:
        from backbone import get_backbone
        
        args = Namespace(
            backbone='vit',
            num_classes=4,
            pretrained=False,
            pretrain_type='in21k-ft-in1k'
        )
        
        backbone = get_backbone(args)
        print(f"✓ Backbone loaded: {type(backbone).__name__}")
        print(f"  - Parameters: {sum(p.numel() for p in backbone.parameters()):,}")
        
        # Test forward pass
        test_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = backbone(test_input)
        
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backbone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_continual_learning_models():
    """Test DER++ and EWC models."""
    print("Testing continual learning models...")
    
    try:
        from models import get_model
        from backbone import get_backbone
        from datasets import get_dataset
        
        args = Namespace(
            dataset='seq-cifar10-224-custom',
            backbone='vit',
            seed=42,
            custom_class_order=None,
            permute_classes=False,
            custom_task_order=None,
            label_perc=1,
            label_perc_by_class=1,
            validation=None,
            validation_mode='current',
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
            optim_wd=0.0,
            optim_mom=0.0,
            optim_nesterov=False,
            device='cpu',
            batch_size=32,
            drop_last=False,
            num_workers=0
        )
        
        dataset = get_dataset(args)
        args.num_classes = dataset.N_CLASSES
        
        backbone_net = get_backbone(args)
        loss = dataset.get_loss()
        transform = dataset.get_transform()
        
        # Test DER++
        print("  Testing DER++...")
        args.model = 'derpp'
        args.buffer_size = 50
        args.alpha = 0.1
        args.beta = 0.5
        
        model_derpp = get_model(args, backbone_net, loss, transform, dataset=dataset)
        print(f"    ✓ DER++ created, buffer size: {model_derpp.buffer.buffer_size}")
        
        # Test EWC
        print("  Testing EWC...")
        args.model = 'ewc_on'
        args.e_lambda = 0.4
        args.gamma = 0.85
        
        backbone_net_ewc = get_backbone(args)  # Fresh backbone for EWC
        model_ewc = get_model(args, backbone_net_ewc, loss, transform, dataset=dataset)
        print(f"    ✓ EWC created")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_visualization():
    """Test attention visualization components."""
    print("Testing attention visualization...")
    
    try:
        from utils.attention_visualization import AttentionExtractor, visualize_attention_map
        from utils.network_flow_visualization import ActivationExtractor, visualize_activation_flow
        from backbone import get_backbone
        from datasets import get_dataset
        from models import get_model
        
        args = Namespace(
            backbone='vit',
            pretrained=False,
            pretrain_type='in21k-ft-in1k',
            dataset='seq-cifar10-224-custom',
            seed=42,
            custom_class_order=None,
            permute_classes=False,
            custom_task_order=None,
            label_perc=1,
            label_perc_by_class=1,
            validation=None,
            validation_mode='current',
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
            optim_wd=0.0,
            optim_mom=0.0,
            optim_nesterov=False,
            device='cpu',
            model='derpp',
            buffer_size=50,
            alpha=0.1,
            beta=0.5,
            batch_size=32,
            drop_last=False,
            num_workers=0
        )
        
        dataset = get_dataset(args)
        args.num_classes = dataset.N_CLASSES
        
        backbone_net = get_backbone(args)
        loss = dataset.get_loss()
        transform = dataset.get_transform()
        model_instance = get_model(args, backbone_net, loss, transform, dataset=dataset)
        
        # Test attention extractor
        attention_extractor = AttentionExtractor(model_instance, device='cpu')
        print("    ✓ AttentionExtractor created")
        
        # Test activation extractor
        activation_extractor = ActivationExtractor(model_instance, device='cpu')
        print("    ✓ ActivationExtractor created")
        
        # Test extraction with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        attention_maps = attention_extractor.extract_attention(dummy_input)
        print(f"    ✓ Attention extraction successful, {len(attention_maps)} layers")
        
        activations = activation_extractor.extract_activations(dummy_input)
        print(f"    ✓ Activation extraction successful, {len(activations)} layers")
        
        # Test visualization (without saving)
        if attention_maps:
            first_layer = list(attention_maps.keys())[0]
            first_attention = attention_maps[first_layer]
            print(f"    ✓ First attention map shape: {first_attention.shape}")
        
        # Clean up
        attention_extractor.remove_hooks()
        activation_extractor.remove_hooks()
        print("    ✓ Hooks cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ATTENTION-FOCUSED EXPERIMENT SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Reproducible Initialization", test_reproducible_initialization),
        ("Custom Dataset", test_custom_dataset),
        ("ViT Backbone", test_vit_backbone),
        ("Continual Learning Models", test_continual_learning_models),
        ("Attention Visualization", test_attention_visualization),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
            import traceback
            traceback.print_exc()
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to run attention experiment.")
        print("\nNext steps:")
        print("1. Run: python run_attention_experiment.py")
        print("2. Or run with specific parameters:")
        print("   python run_attention_experiment.py --epochs 10 --methods derpp ewc_on")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before proceeding.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
