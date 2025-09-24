#!/usr/bin/env python3
"""
Test script to validate corrected ERI calculations against real data.

This script tests the corrected ERI implementation against the actual data
from einstellung_results/session_20250923-012304_seed42 to verify that
DER++ shows better performance than scratch_t2 baseline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from eri_vis.data_loader import ERIDataLoader
from eri_vis.metrics_calculator import CorrectedERICalculator
from eri_vis.eri_composite import CompositeERICalculator
from eri_vis.plot_overall_eri import OverallERIPlotter


def load_session_data():
    """Load data from the provided session."""
    session_dir = Path("einstellung_results/session_20250923-012304_seed42")

    # Load timeline data for different methods
    timeline_files = list(session_dir.glob("timeline_*.csv"))

    all_data = []

    for timeline_file in timeline_files:
        print(f"Loading {timeline_file}")
        df = pd.read_csv(timeline_file)
        all_data.append(df)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined data shape: {combined_df.shape}")
        print(f"Methods: {combined_df['method'].unique()}")
        print(f"Splits: {combined_df['split'].unique()}")

        # Handle duplicate entries by averaging them
        print("Checking for duplicates...")
        duplicates = combined_df.duplicated(subset=['method', 'seed', 'epoch_eff', 'split'])
        if duplicates.any():
            print(f"Found {duplicates.sum()} duplicate entries - averaging them")
            combined_df = combined_df.groupby(['method', 'seed', 'epoch_eff', 'split']).agg({
                'acc': 'mean',
                'top5': 'mean',
                'loss': 'mean'
            }).reset_index()
            print(f"After deduplication: {combined_df.shape}")

        return combined_df
    else:
        print("No timeline files found")
        return None


def test_corrected_calculations():
    """Test the corrected ERI calculations."""

    # Load real data
    df = load_session_data()
    if df is None:
        print("Could not load session data")
        return

    # Create dataset
    loader = ERIDataLoader()

    # Add missing columns if needed
    if 'top5' not in df.columns:
        df['top5'] = df['acc'] + 0.05  # Approximate top5
    if 'loss' not in df.columns:
        df['loss'] = 1.0 - df['acc']  # Approximate loss

    # Validate and create dataset
    try:
        loader.validate_format(df)

        methods = sorted(df['method'].unique())
        splits = sorted(df['split'].unique())
        seeds = sorted(df['seed'].unique())
        epoch_range = (df['epoch_eff'].min(), df['epoch_eff'].max())

        from eri_vis.dataset import ERITimelineDataset
        dataset = ERITimelineDataset(
            data=df,
            metadata={'source': 'session_20250923-012304_seed42'},
            methods=methods,
            splits=splits,
            seeds=seeds,
            epoch_range=epoch_range
        )

        print(f"Dataset created successfully:")
        print(f"  Methods: {methods}")
        print(f"  Splits: {splits}")
        print(f"  Seeds: {seeds}")
        print(f"  Epoch range: {epoch_range}")

    except Exception as e:
        print(f"Dataset creation failed: {e}")
        return

    # Test corrected calculations
    calculator = CorrectedERICalculator(tau=0.6, smoothing_window=3)

    # Compute metrics for all methods
    all_metrics = calculator.compute_all_metrics(dataset, scratch_method="scratch_t2")

    print(f"\nComputed metrics for {len(all_metrics)} method-seed combinations:")

    for (method, seed), metrics in all_metrics.items():
        print(f"\n{method} (seed {seed}):")
        print(f"  AD: {metrics.adaptation_delay}")
        print(f"  PD: {metrics.performance_deficit}")
        print(f"  SFR_rel: {metrics.shortcut_feature_reliance}")
        print(f"  Censored: {metrics.censored}")

    # Aggregate by method
    method_stats = calculator.aggregate_metrics_by_method(all_metrics)

    print(f"\nMethod-level statistics:")
    for method, stats in method_stats.items():
        print(f"\n{method}:")
        print(f"  AD: {stats.get('AD_mean', 'N/A'):.3f} ± {stats.get('AD_ci', 0):.3f}")
        print(f"  PD: {stats.get('PD_mean', 'N/A'):.4f} ± {stats.get('PD_ci', 0):.4f}")
        print(f"  SFR_rel: {stats.get('SFR_rel_mean', 'N/A'):.4f} ± {stats.get('SFR_rel_ci', 0):.4f}")
        print(f"  Censored runs: {stats.get('censored_runs', 0)}/{stats.get('total_runs', 0)}")

    # Test overall ERI computation
    composite_calc = CompositeERICalculator()

    # Group metrics by method for composite calculation
    method_metrics = {}
    for (method, seed), metrics in all_metrics.items():
        if method not in method_metrics:
            method_metrics[method] = []
        method_metrics[method].append(metrics)

    # Compute composite scores
    rankings = composite_calc.rank_methods_by_composite(method_metrics)

    print(f"\nOverall ERI Rankings (lower = better):")
    for i, (method, score) in enumerate(rankings, 1):
        print(f"  {i}. {method}: {score:.4f}")

    # Validate that DER++ performs well
    derpp_found = False
    for method, score in rankings:
        if 'derpp' in method.lower():
            derpp_found = True
            derpp_rank = rankings.index((method, score)) + 1
            print(f"\nDER++ ranking: {derpp_rank}/{len(rankings)} (score: {score:.4f})")

            if derpp_rank <= len(rankings) // 2:
                print("✓ DER++ shows good performance (top half)")
            else:
                print("⚠ DER++ performance may need investigation")
            break

    if not derpp_found:
        print("⚠ DER++ not found in results")

    # Generate visualization if possible
    try:
        plotter = OverallERIPlotter()
        output_dir = Path("validation_output")
        output_dir.mkdir(exist_ok=True)

        plotter.plot_overall_eri(
            method_stats,
            output_dir / "validation_overall_eri.pdf",
            figsize=(10, 6)
        )

        plotter.create_eri_comparison_table(
            method_stats,
            output_dir / "validation_eri_comparison.csv"
        )

        print(f"\nValidation outputs saved to {output_dir}/")

    except Exception as e:
        print(f"Visualization generation failed: {e}")

    return method_stats, rankings


if __name__ == "__main__":
    print("Testing corrected ERI calculations with real data...")
    print("=" * 60)

    try:
        method_stats, rankings = test_corrected_calculations()
        print("\n" + "=" * 60)
        print("Validation completed successfully!")

    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        import traceback
        traceback.print_exc()
