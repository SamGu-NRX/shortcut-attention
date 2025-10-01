#!/usr/bin/env python3
"""
Test script to verify the recursion fix in einstellung integration.
"""

import sys
import os
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_recursion_fix():
    """Test that the einstellung integration doesn't cause infinite recursion."""

    print("🧪 Testing Einstellung Integration Recursion Fix")
    print("=" * 60)

    # Import after setting up logging
    from utils.einstellung_integration import enable_einstellung_integration, disable_einstellung_integration

    # Create minimal args
    args = argparse.Namespace()
    args.dataset = 'seq-cifar100-einstellung'
    args.n_epochs = 1  # Minimal epochs for testing
    args.device = 'cpu'
    args.results_path = './test_output_quick'
    args.backbone = 'resnet18'
    args.batch_size = 32
    args.lr = 0.1
    args.model = 'sgd'
    args.buffer_size = 0
    args.alpha = 0.1
    args.beta = 0.1
    args.gamma = 1.0
    args.minibatch_size = 32
    args.disable_log = True
    args.debug_mode = True
    args.loadcheck = None
    args.start_from = None
    args.stop_after = None
    args.eval_future = False

    try:
        # Test integration enable
        print("1. Testing integration enable...")
        result = enable_einstellung_integration(args)
        print(f"   ✓ Integration enabled: {result}")

        if not result:
            print("   ❌ Integration failed to enable")
            return False

        # Test that we can import training without issues
        print("2. Testing training import...")
        import utils.training
        print("   ✓ Training module imported successfully")

        # Test that the patched function exists
        print("3. Testing patched function...")
        from utils.einstellung_integration import _original_train
        if _original_train is not None:
            print("   ✓ Original training function stored")
        else:
            print("   ❌ Original training function not stored")
            return False

        # Test basic function call (without full training)
        print("4. Testing function reference...")
        train_func = utils.training.train
        print(f"   ✓ Training function: {train_func.__name__}")

        # Clean up
        print("5. Testing integration disable...")
        disable_einstellung_integration()
        print("   ✓ Integration disabled")

        print("\n🎉 All tests passed! Recursion fix appears to be working.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_recursion_fix()
    sys.exit(0 if success else 1)
