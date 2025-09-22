#!/usr/bin/env python3
"""
Simple test to verify Einstellung data collection fixes.
"""

import subprocess
import sys
from pathlib import Path

def test_simple_einstellung():
    """Run a simple test with direct main.py call."""

    print("🧪 Testing Einstellung fixes with direct main.py call...")

    cmd = [
        sys.executable, "main.py",
        "--dataset", "seq-cifar100-einstellung",
        "--model", "sgd",
        "--backbone", "resnet18",
        "--seed", "99",  # Different seed to avoid conflicts
        "--n_epochs", "2",  # Very short test
        "--batch_size", "32",
        "--lr", "0.01",
        "--num_workers", "4",
        "--non_verbose", "0",
        "--results_path", "./test_simple_results",
        "--savecheck", "last",
        "--base_path", "./data",
        "--einstellung_evaluation_subsets", "1",
        "--einstellung_extract_attention", "0",  # Disable for ResNet
        "--einstellung_apply_shortcut", "1",
        "--einstellung_adaptation_threshold", "0.8"
    ]

    print("Command:", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        print(f"Return code: {result.returncode}")
        print("STDOUT:", result.stdout[-2000:])  # Last 2000 chars
        if result.stderr:
            print("STDERR:", result.stderr[-1000:])  # Last 1000 chars

        # Check CSV
        csv_path = Path("./test_simple_results/eri_sc_metrics.csv")
        if csv_path.exists():
            print(f"\n✅ CSV generated: {csv_path}")
            with open(csv_path) as f:
                content = f.read()
                print("CSV content:")
                print(content)
        else:
            print(f"\n❌ No CSV found at: {csv_path}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_einstellung()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
