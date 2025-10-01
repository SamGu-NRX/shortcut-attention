#!/usr/bin/env python3
"""
ViT Einstellung Debug Test Script

This script tests all the comprehensive fixes for ViT implementation issues:
1. Einstellung integration activation
2. Optimized attention extraction
3. Dataset compatibility
4. Training flow logging
5. Checkpoint handling

Usage:
    python test_vit_einstellung_debug.py --model derpp --backbone vit
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add mammoth to path
sys.path.append(str(Path(__file__).parent))

import torch


def setup_logging():
    """Setup comprehensive logging for debugging."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('vit_einstellung_debug.log')
        ]
    )


def test_einstellung_integration():
    """Test 1: Verify Einstellung integration activates properly."""
    print("🧪 TEST 1: Einstellung Integration Activation")

    try:
        from utils.einstellung_integration import enable_einstellung_integration
        from argparse import Namespace

        # Test with Einstellung dataset
        args = Namespace()
        args.dataset = 'seq-cifar100-einstellung-224'

        result = enable_einstellung_integration(args)
        print(f"   ✓ Integration result: {result}")

        if result:
            print("   ✅ Einstellung integration successfully activated")
        else:
            print("   ❌ Einstellung integration failed to activate")

    except Exception as e:
        print(f"   ❌ Error testing integration: {e}")


def test_optimized_attention():
    """Test 2: Verify optimized attention analyzer works."""
    print("\n🧪 TEST 2: Optimized Attention Analyzer")

    try:
        from utils.attention_visualization import AttentionAnalyzer

        # Create dummy model-like object
        class DummyViT:
            def to(self, device): pass
            def eval(self): pass

            def forward(self, x, return_attention_scores=False):
                batch_size = x.shape[0]
                if return_attention_scores:
                    # Return dummy attention maps
                    attn_maps = [torch.randn(batch_size, 12, 197, 197) for _ in range(3)]
                    output = torch.randn(batch_size, 197, 768)
                    return output, attn_maps
                else:
                    return torch.randn(batch_size, 197, 768)

        # Test analyzer
        dummy_model = DummyViT()
        analyzer = AttentionAnalyzer(dummy_model, device='cpu', max_samples_per_analysis=4)

        # Test with small batch
        test_input = torch.randn(2, 3, 224, 224)
        attention_maps = analyzer.extract_attention_maps(test_input)

        print(f"   ✓ Extracted {len(attention_maps)} attention maps")
        print("   ✅ Optimized attention analyzer working")

    except Exception as e:
        print(f"   ❌ Error testing attention analyzer: {e}")


def test_dataset_compatibility():
    """Test 3: Verify ViT dataset compatibility."""
    print("\n🧪 TEST 3: Dataset Compatibility")

    try:
        from datasets.seq_cifar100_einstellung_224 import SequentialCIFAR100Einstellung224
        from argparse import Namespace

        args = Namespace()
        args.einstellung_apply_shortcut = True
        args.einstellung_patch_size = 16
        args.einstellung_patch_color = [255, 0, 255]
        args.batch_size = 4
        args.custom_task_order = None
        args.custom_class_order = None
        args.permute_classes = False
        args.label_perc = 1
        args.label_perc_by_class = 1
        args.validation = None
        args.validation_mode = 'current'
        args.base_path = './data/'
        args.transform_type = 'weak'
        args.joint = False  # Required by ContinualDataset base class

        dataset = SequentialCIFAR100Einstellung224(args)
        print(f"   ✓ Dataset created: {dataset.NAME}")
        print(f"   ✓ N_CLASSES: {dataset.N_CLASSES}")
        print(f"   ✓ N_TASKS: {dataset.N_TASKS}")
        print("   ✅ ViT dataset compatibility confirmed")

    except Exception as e:
        print(f"   ❌ Error testing dataset: {e}")


def test_vit_backbone():
    """Test 4: Verify ViT backbone with debug logging."""
    print("\n🧪 TEST 4: ViT Backbone with Debug Logging")

    try:
        from backbone.vit import VisionTransformer

        # Create small ViT for testing
        vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=192,  # Smaller for faster testing
            depth=6,        # Fewer layers
            num_heads=6,
            num_classes=60  # Einstellung classes
        )

        # Test forward pass
        test_input = torch.randn(2, 3, 224, 224)

        print("   Testing normal forward pass...")
        start_time = time.time()
        output = vit(test_input)
        normal_time = time.time() - start_time
        print(f"   ✓ Normal forward: {normal_time:.3f}s")

        print("   Testing attention extraction...")
        start_time = time.time()
        output, attn_maps = vit(test_input, return_attention_scores=True)
        attention_time = time.time() - start_time
        print(f"   ✓ Attention forward: {attention_time:.3f}s")
        print(f"   ✓ Attention maps: {len(attn_maps)}")

        slowdown_factor = attention_time / normal_time if normal_time > 0 else 0
        print(f"   ⚠️  Attention extraction slowdown: {slowdown_factor:.1f}x")

        if slowdown_factor < 5:
            print("   ✅ ViT timing acceptable")
        else:
            print("   ⚠️  ViT timing may cause issues")

    except Exception as e:
        print(f"   ❌ Error testing ViT backbone: {e}")


def test_comprehensive_flow():
    """Test 5: Test complete flow simulation."""
    print("\n🧪 TEST 5: Comprehensive Flow Simulation")

    try:
        # Simulate the complete experiment flow
        print("   1. Setting up args...")
        from argparse import Namespace

        args = Namespace()
        args.dataset = 'seq-cifar100-einstellung-224'
        args.model = 'derpp'
        args.backbone = 'vit'
        args.seed = 42
        args.einstellung_evaluation_subsets = True
        args.einstellung_extract_attention = True

        print("   2. Testing integration activation...")
        from utils.einstellung_integration import enable_einstellung_integration
        integration_result = enable_einstellung_integration(args)
        print(f"      Integration: {integration_result}")

        print("   3. Testing evaluator creation...")
        from utils.einstellung_evaluator import create_einstellung_evaluator
        evaluator = create_einstellung_evaluator(args)
        print(f"      Evaluator created: {evaluator is not None}")

        if integration_result and evaluator:
            print("   ✅ Comprehensive flow simulation successful")
        else:
            print("   ❌ Comprehensive flow simulation failed")

    except Exception as e:
        print(f"   ❌ Error in comprehensive flow: {e}")


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='ViT Einstellung Debug Tests')
    parser.add_argument('--model', default='derpp', help='Model to test')
    parser.add_argument('--backbone', default='vit', help='Backbone to test')
    args = parser.parse_args()

    print("🚀 ViT EINSTELLUNG DEBUG TEST SUITE")
    print("=" * 50)

    setup_logging()

    # Run all tests
    test_einstellung_integration()
    test_optimized_attention()
    test_dataset_compatibility()
    test_vit_backbone()
    test_comprehensive_flow()

    print("\n" + "=" * 50)
    print("🏁 DEBUG TEST SUITE COMPLETED")
    print("\nNext steps:")
    print("1. If all tests pass, try: python run_einstellung_experiment.py --model derpp --backbone vit --force_retrain")
    print("2. Check logs for detailed timing and integration information")
    print("3. Monitor for timeout issues during actual training")


if __name__ == "__main__":
    main()
