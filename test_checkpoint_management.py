#!/usr/bin/env python3
"""
Test script for Einstellung checkpoint management functionality.

This script demonstrates the new checkpoint management features:
1. Automatic checkpoint discovery
2. Skip-training mode
3. Interactive prompts
4. Force retrain options

Usage:
    python test_checkpoint_management.py
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our module
sys.path.append(str(Path(__file__).parent))

from run_einstellung_experiment import (
    find_existing_checkpoints,
    get_checkpoint_info,
    run_einstellung_experiment
)


def test_checkpoint_discovery():
    """Test checkpoint discovery functionality."""
    print("🔍 Testing Checkpoint Discovery")
    print("=" * 50)

    # Test with common strategies
    strategies = ['derpp', 'ewc_on', 'sgd']
    backbones = ['resnet18', 'vit']

    for strategy in strategies:
        for backbone in backbones:
            checkpoints = find_existing_checkpoints(strategy, backbone, seed=42)

            if checkpoints:
                print(f"✅ Found {len(checkpoints)} checkpoint(s) for {strategy}/{backbone}")
                for ckpt in checkpoints[:2]:  # Show first 2
                    info = get_checkpoint_info(ckpt)
                    print(f"   - {os.path.basename(ckpt)} ({info['size_mb']} MB)")
            else:
                print(f"❌ No checkpoints found for {strategy}/{backbone}")

    print()


def test_evaluation_modes():
    """Test different evaluation modes."""
    print("🧪 Testing Evaluation Modes")
    print("=" * 50)

    print("1. Skip-training mode (will fail if no checkpoints exist):")
    result = run_einstellung_experiment(
        strategy='derpp',
        backbone='resnet18',
        seed=42,
        skip_training=True
    )

    if result and result.get('success'):
        print(f"   ✅ Evaluation completed with {result['final_accuracy']:.2f}% accuracy")
        print(f"   📄 Used checkpoint: {result.get('used_checkpoint', False)}")
    else:
        print(f"   ❌ Failed: {result.get('error', 'Unknown error') if result else 'No result'}")

    print("\n2. Auto-checkpoint mode:")
    result = run_einstellung_experiment(
        strategy='derpp',
        backbone='resnet18',
        seed=42,
        auto_checkpoint=True
    )

    if result and result.get('success'):
        print(f"   ✅ Experiment completed with {result['final_accuracy']:.2f}% accuracy")
        print(f"   📄 Used checkpoint: {result.get('used_checkpoint', False)}")
        print(f"   🔄 Evaluation only: {result.get('evaluation_only', False)}")
    else:
        print(f"   ❌ Failed: {result.get('error', 'Unknown error') if result else 'No result'}")

    print()


def main():
    """Main test function."""
    print("🚀 Einstellung Checkpoint Management Test Suite")
    print("=" * 60)
    print()

    # Test checkpoint discovery
    test_checkpoint_discovery()

    # Test evaluation modes
    test_evaluation_modes()

    print("✅ Test suite completed!")
    print("\n💡 Usage Examples:")
    print("   # Use existing checkpoints automatically:")
    print("   python run_einstellung_experiment.py --model derpp --auto_checkpoint")
    print()
    print("   # Skip training, evaluate only:")
    print("   python run_einstellung_experiment.py --model derpp --skip_training")
    print()
    print("   # Force retrain even if checkpoints exist:")
    print("   python run_einstellung_experiment.py --model derpp --force_retrain")
    print()
    print("   # Interactive mode (default - will prompt if checkpoints found):")
    print("   python run_einstellung_experiment.py --model derpp")


if __name__ == '__main__':
    main()
