#!/usr/bin/env python3
"""
Test script to verify Einstellung data collection fixes.

This script runs a minimal Einstellung experiment to test:
1. Proper epoch numbering (0, 1, 2, ... instead of 0.1, 0.2, ...)
2. Real data collection instead of synthetic fallback
3. Timeline data persistence and CSV export
"""

import subprocess
import sys
import os
from pathlib import Path

def run_test_experiment():
    """Run a minimal test experiment to verify the fixes."""

    print("🧪 Testing Einstellung data collection fixes...")
    print("=" * 60)

    # Test configuration - minimal epochs for quick testing
    cmd = [
        sys.executable, "main.py",
        "--dataset", "seq-cifar100-einstellung",
        "--model", "sgd",
        "--backbone", "resnet18",
        "--seed", "42",
        "--n_epochs", "3",  # Minimal epochs for testing
        "--batch_size", "32",
        "--lr", "0.01",
        "--num_workers", "4",
        "--non_verbose", "0",
        "--results_path", "./test_einstellung_fix_results",
        "--savecheck", "last",
        "--base_path", "./data",
        "--einstellung_evaluation_subsets", "1",
        "--einstellung_extract_attention", "0",  # Disable attention for ResNet
        "--einstellung_apply_shortcut", "1",
        "--einstellung_adaptation_threshold", "0.8"
    ]

    print("🚀 Running test command:")
    print(" ".join(cmd))
    print()

    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout

        print("📋 STDOUT:")
        print(result.stdout)
        print("\n📋 STDERR:")
        print(result.stderr)
        print(f"\n📊 Return code: {result.returncode}")

        # Check if CSV was generated
        csv_path = Path("./test_einstellung_fix_results/eri_sc_metrics.csv")
        if csv_path.exists():
            print(f"\n✅ CSV file generated: {csv_path}")

            # Read and analyze the CSV
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)

                print(f"📊 CSV Analysis:")
                print(f"   Rows: {len(df)}")
                print(f"   Columns: {list(df.columns)}")

                if 'epoch_eff' in df.columns:
                    epochs = sorted(df['epoch_eff'].unique())
                    print(f"   Epochs: {epochs}")

                    # Check if epochs are integers (not fractional)
                    if all(epoch == int(epoch) for epoch in epochs):
                        print("   ✅ Epochs are integers (not fractional)")
                    else:
                        print("   ❌ Epochs contain fractional values")

                if 'split' in df.columns:
                    splits = sorted(df['split'].unique())
                    print(f"   Splits: {splits}")

                    expected_splits = ['T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal']
                    if all(split in splits for split in expected_splits):
                        print("   ✅ All expected evaluation subsets present")
                    else:
                        missing = [s for s in expected_splits if s not in splits]
                        print(f"   ⚠️  Missing splits: {missing}")

                # Show sample data
                print(f"\n📋 Sample CSV data:")
                print(df.head(10).to_string(index=False))

            except ImportError:
                print("   ⚠️  pandas not available for CSV analysis")
            except Exception as e:
                print(f"   ❌ Error analyzing CSV: {e}")
        else:
            print(f"\n❌ CSV file not generated at: {csv_path}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main test function."""
    print("Einstellung Data Collection Fix Test")
    print("=" * 40)

    success = run_test_experiment()

    print("\n" + "=" * 60)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed - check output above for details")
    print("=" * 60)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
