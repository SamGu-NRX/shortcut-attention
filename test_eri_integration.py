#!/usr/bin/env python3
"""
Quick integration test for ERIDataLoader functionality.
"""

import tempfile
import pandas as pd
from pathlib import Path

from eri_vis.data_loader import ERIDataLoader

def test_basic_functionality():
    """Test basic ERIDataLoader functionality."""

    # Create test data
    test_data = pd.DataFrame([
        {'method': 'Scratch_T2', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T1_all', 'acc': 0.85},
        {'method': 'Scratch_T2', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T2_shortcut_normal', 'acc': 0.10},
        {'method': 'Scratch_T2', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T2_shortcut_masked', 'acc': 0.05},
        {'method': 'sgd', 'seed': 42, 'epoch_eff': 0.0, 'split': 'T2_shortcut_normal', 'acc': 0.20},
        {'method': 'sgd', 'seed': 42, 'epoch_eff': 1.0, 'split': 'T2_shortcut_normal', 'acc': 0.40},
    ])

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        csv_path = f.name

    try:
        # Test CSV loading
        loader = ERIDataLoader()
        dataset = loader.load_csv(csv_path)

        print(f"✓ Successfully loaded CSV with {len(dataset.data)} rows")
        print(f"✓ Methods: {dataset.methods}")
        print(f"✓ Splits: {dataset.splits}")
        print(f"✓ Seeds: {dataset.seeds}")
        print(f"✓ Epoch range: {dataset.epoch_range}")

        # Test evaluator export conversion
        export_data = {
            'configuration': {
                'method': 'test_method',
                'seed': 123
            },
            'timeline_data': [
                {
                    'epoch': 0,
                    'task_id': 1,
                    'subset_accuracies': {
                        'T1_all': 0.8,
                        'T2_shortcut_normal': 0.15,
                        'T2_shortcut_masked': 0.08
                    }
                }
            ]
        }

        dataset2 = loader.load_from_evaluator_export(export_data)
        print(f"✓ Successfully converted evaluator export with {len(dataset2.data)} rows")

        print("\n✅ All basic functionality tests passed!")

    finally:
        # Clean up
        Path(csv_path).unlink(missing_ok=True)

if __name__ == "__main__":
    test_basic_functionality()
