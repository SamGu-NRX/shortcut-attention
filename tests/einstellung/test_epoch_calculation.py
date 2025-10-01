#!/usr/bin/env python3
"""
Test script to verify epoch calculation and data collection in einstellung integration.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_epoch_calculation():
    """Test that epoch calculation and data collection work correctly."""

    print("🧪 Testing Einstellung Epoch Calculation and Data Collection")
    print("=" * 70)

    # Create test output directory
    test_output = Path('./test_output_epoch')
    test_output.mkdir(exist_ok=True)

    try:
        # Run a minimal experiment with debug mode
        print("1. Running minimal einstellung experiment...")

        cmd = [
            'python', 'run_einstellung_experiment.py',
            '--model', 'sgd',
            '--backbone', 'resnet18',
            '--seed', '42',
            '--epochs', '3',  # Very short for testing
            '--debug',  # Enable debug mode for faster execution
        ]

        print(f"   Command: {' '.join(cmd)}")

        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout

        print(f"   Return code: {result.returncode}")

        if result.returncode != 0:
            print("   ❌ Experiment failed!")
            print("   STDOUT:")
            print(result.stdout)
            print("   STDERR:")
            print(result.stderr)
            return False
        else:
            print("   ✓ Experiment completed successfully")

        # Check for CSV output
        print("2. Checking for CSV output...")
        csv_files = list(test_output.glob("**/eri_sc_metrics.csv"))

        if not csv_files:
            print("   ❌ No CSV files found")
            return False

        csv_file = csv_files[0]
        print(f"   ✓ Found CSV file: {csv_file}")

        # Read and analyze CSV content
        print("3. Analyzing CSV content...")
        import pandas as pd

        try:
            df = pd.read_csv(csv_file)
            print(f"   ✓ CSV loaded successfully: {len(df)} rows")

            # Check columns
            expected_cols = ['method', 'seed', 'epoch_eff', 'split', 'acc']
            missing_cols = set(expected_cols) - set(df.columns)
            if missing_cols:
                print(f"   ❌ Missing columns: {missing_cols}")
                return False
            print(f"   ✓ All required columns present: {list(df.columns)}")

            # Check epoch values
            epochs = sorted(df['epoch_eff'].unique())
            print(f"   ✓ Epoch values: {epochs}")

            # Check if epochs are integers (not fractional like 0.1, 0.2)
            fractional_epochs = [e for e in epochs if e != int(e)]
            if fractional_epochs:
                print(f"   ⚠️  Found fractional epochs: {fractional_epochs}")
                print("   This might indicate synthetic data fallback")
            else:
                print("   ✓ All epochs are integers (real data)")

            # Check splits
            splits = sorted(df['split'].unique())
            print(f"   ✓ Splits found: {splits}")

            expected_splits = ['T1_all', 'T2_shortcut_normal', 'T2_shortcut_masked', 'T2_nonshortcut_normal']
            missing_splits = set(expected_splits) - set(splits)
            if missing_splits:
                print(f"   ⚠️  Missing splits: {missing_splits}")
            else:
                print("   ✓ All expected splits present")

            # Show sample data
            print("4. Sample CSV data:")
            print(df.head(10).to_string(index=False))

            return True

        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            return False

    except subprocess.TimeoutExpired:
        print("   ❌ Experiment timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_epoch_calculation()
    sys.exit(0 if success else 1)
