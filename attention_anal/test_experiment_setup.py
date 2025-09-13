"""Test script to verify the shortcut investigation experiment setup."""

import os
import sys
import torch
import numpy as np
from argparse import Namespace
import logging

from utils.attention_visualization import AttentionAnalyzer

logging.basicConfig(level=logging.INFO)

def test_dataset():
    """Test if the custom dataset and task organization loads correctly."""
    print("Testing custom dataset...")

    try:
        from datasets import get_dataset
        from datasets.seq_cifar10_224_custom import SequentialCIFAR10Custom224

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
            num_workers=0 # Added for create_seeded_dataloader
        )
        
        dataset = get_dataset(args)
        if not isinstance(dataset, SequentialCIFAR10Custom224):
            raise TypeError(f"Expected SequentialCIFAR10Custom224, got {type(dataset)}")
            
        print(f"✓ Dataset loaded successfully")
        print(f"  - Name: {dataset.NAME}")
        print(f"  - Setting: {dataset.SETTING}")
        print(f"  - Classes per task: {dataset.N_CLASSES_PER_TASK}")
        print(f"  - Number of tasks: {dataset.N_TASKS}")
        print(f"  - Total classes: {dataset.N_CLASSES}")
        print(f"  - Class order (indices): {dataset.class_order}")
        
        class_names = dataset.get_class_names()
        print(f"  - Class names: {class_names}")
        if len(class_names) != dataset.N_CLASSES:
            print(f"  WARNING: Class names length ({len(class_names)}) "
                  f"does not match dataset N_CLASSES ({dataset.N_CLASSES})")

        task_labels = dataset.get_task_labels() # get_task_labels is always available now
        print(f"  - Task labels: {task_labels}")
        
        train_loader, test_loader = dataset.get_data_loaders()
        print(f"  - Train loader: {len(train_loader)} batches")
        print(f"  - Test loader: {len(test_loader)} batches")
        
        data_iter = iter(train_loader)
        data, targets, not_aug_data = next(data_iter)
        print(f"  - Batch shape: {data.shape}")
        print(f"  - Target shape: {targets.shape}")
        print(f"  - Unique targets in batch: {torch.unique(targets).tolist()}")
        
        expected_class_names = ['airplane', 'automobile', 'bird', 'truck']
        assert class_names == expected_class_names, \
            f"Class names mismatch: Expected {expected_class_names}, Got {class_names}"
        
        expected_task_labels = {0: ['airplane', 'automobile'], 1: ['bird', 'truck']}
        assert task_labels == expected_task_labels, \
            f"Task labels mismatch: Expected {expected_task_labels}, Got {task_labels}"

        # Test task organization
        print("\nValidating task organization...")
        if hasattr(dataset, 'class_order') and dataset.class_order is not None:
            class_order_str = ','.join(map(str, dataset.class_order))
        else:
            class_order_str = "Default"
        print(f"  - Current class order: {class_order_str}")
        print(f"  - Number of tasks: {dataset.N_TASKS}")
        print(f"  - Classes per task: {dataset.N_CLASSES_PER_TASK}")
        
        expected_tasks = {
            0: ['airplane', 'automobile'],  # Task 1: Vehicle types with sky/road shortcuts
            1: ['bird', 'truck']           # Task 2: Mixed types with sky/road shortcuts
        }
        assert task_labels == expected_tasks, \
            f"Task labels mismatch:\nExpected: {expected_tasks}\nGot: {task_labels}"
        print(f"✓ Task organization verified successfully")

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
            backbone='vit',
            num_classes=4,
            pretrained=False,
            pretrain_type='in21k-ft-in1k',
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            kwargs={}  # Required empty kwargs
        )
        backbone = get_backbone(args)
        print(f"✓ Backbone loaded successfully")
        # ... (rest of backbone test)
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
            kwargs={},  # Required empty kwargs
            # ViT specific arguments
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
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
            device='cuda',
            model='derpp', 
            buffer_size=50,
            alpha=0.1,
            beta=0.5,
            e_lambda=0.4, 
            gamma=0.85,
            batch_size=32, 
            drop_last=False,
            num_workers=0 # Added
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
        backbone_net_ewc = get_backbone(args) 
        model_ewc = get_model(args, backbone_net_ewc, loss, transform, dataset=dataset)
        print(f"    ✓ EWC model created successfully")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback; traceback.print_exc()
        return False

# Add this function at the top after the imports
def get_test_device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

# Replace the test_visualization_utils function:
def test_visualization_utils():
    """Test if the visualization utilities work."""
    print("\nTesting visualization utilities...")
    test_device = get_test_device()
    print(f"  Using device: {test_device}")
    
    try:
        from utils.attention_visualization import AttentionAnalyzer
        from utils.network_flow_visualization import ActivationExtractor
        from backbone import get_backbone 
        from datasets import get_dataset 
        from models import get_model

        args = Namespace(
            backbone='vit',
            pretrained=False,
            pretrain_type='in21k-ft-in1k',
            dataset='seq-cifar10-224-custom',
            kwargs={},  # Required empty kwargs
            # ViT specific arguments
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
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
            device=test_device,  # Use the determined device
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

        # Test attention analysis
        attention_analyzer = AttentionAnalyzer(model_instance, device=test_device)
        print(f"    ✓ AttentionAnalyzer created")
        
        # Test activation extraction
        activation_extractor = ActivationExtractor(model_instance, device=test_device)
        print(f"    ✓ ActivationExtractor created")
        
        # Test with dummy input (on CPU, will be moved to device by extractors)
        dummy_input = torch.randn(1, 3, 224, 224) 
        
        # Extract and check attention maps
        attention_maps = attention_analyzer.extract_attention_maps(dummy_input)
        print(f"    ✓ Attention extraction successful")
        print(f"    - Number of attention maps: {len(attention_maps)}")
        
        # Extract and check activations
        activations = activation_extractor.extract_activations(dummy_input)
        print(f"    ✓ Activation extraction successful")
        print(f"    - Number of activation maps: {len(activations)}")
        
        # Clean up
        activation_extractor.remove_hooks()
        del attention_analyzer  # This will trigger __del__ which cleans up hooks
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        import traceback; traceback.print_exc()
        return False

# Also update the test_experiment_runner function to use the correct device:
def test_experiment_runner():
    """Test if the experiment runner can be imported."""
    print("\nTesting experiment runner...")
    test_device = get_test_device()
    
    try:
        from experiments.analyze_attention import ShortcutInvestigationExperiment
        base_args = {
            'n_epochs': 1,
            'batch_size': 32,
            'lr': 0.01,
            'device': '0' if test_device == 'cuda' else 'cpu',  # Use proper format for Mammoth
            'debug_mode': 1,
            'permute_classes': False,
            'custom_class_order': None,
            'custom_task_order': None,
            'validation_mode': 'current',
            'optimizer': 'sgd',
            'optim_wd': 0.0,
            'optim_mom': 0.0,
            'optim_nesterov': False,
            'drop_last': False,
            'num_workers': 0,
            # Model and dataset settings
            'backbone': 'vit',
            'dataset': 'seq-cifar10-224-custom',
            'pretrained': False,
            'pretrain_type': 'in21k-ft-in1k',
            'kwargs': {},  # Required empty kwargs
            
            # ViT specific arguments
            'img_size': 224,
            'patch_size': 16,
            'in_chans': 3,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.,
            'qkv_bias': True,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            
            # Data paths and settings
            'base_path': './data/',
            'transform_type': 'weak'
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
    total_count = len(tests) 
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
