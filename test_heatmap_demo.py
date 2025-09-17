#!/usr/bin/env python3
"""
Demo script for ERIHeatmapPlotter functionality.

This script demonstrates the AD(τ) sensitivity heatmap generation
with synthetic data that matches the requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from eri_vis.plot_heatmap import ERIHeatmapPlotter
from eri_vis.processing import AccuracyCurve
from eri_vis.styles import PlotStyleConfig


def create_synthetic_curves():
    """Create synthetic accuracy curves for demonstration."""
    epochs = np.linspace(0, 10, 50)

    # Scratch_T2 baseline - reaches high accuracy quickly
    scratch_acc = 0.2 + 0.7 * (1 - np.exp(-epochs * 0.8))
    scratch_curve = AccuracyCurve(
        epochs=epochs,
        mean_accuracy=scratch_acc,
        conf_interval=np.zeros_like(scratch_acc),
        method="Scratch_T2",
        split="T2_shortcut_normal",
        n_seeds=5
    )

    # SGD - slower adaptation, reaches threshold later
    sgd_acc = 0.15 + 0.75 * (1 - np.exp(-epochs * 0.4))
    sgd_curve = AccuracyCurve(
        epochs=epochs,
        mean_accuracy=sgd_acc,
        conf_interval=np.zeros_like(sgd_acc),
        method="sgd",
        split="T2_shortcut_normal",
        n_seeds=5
    )

    # EWC - even slower adaptation
    ewc_acc = 0.1 + 0.6 * (1 - np.exp(-epochs * 0.3))
    ewc_curve = AccuracyCurve(
        epochs=epochs,
        mean_accuracy=ewc_acc,
        conf_interval=np.zeros_like(ewc_acc),
        method="ewc_on",
        split="T2_shortcut_normal",
        n_seeds=5
    )

    # DER++ - moderate adaptation speed
    derpp_acc = 0.12 + 0.68 * (1 - np.exp(-epochs * 0.5))
    derpp_curve = AccuracyCurve(
        epochs=epochs,
        mean_accuracy=derpp_acc,
        conf_interval=np.zeros_like(derpp_acc),
        method="derpp",
        split="T2_shortcut_normal",
        n_seeds=5
    )

    # GMP - struggles to reach high thresholds (some censored runs)
    gmp_acc = 0.08 + 0.5 * (1 - np.exp(-epochs * 0.25))
    gmp_curve = AccuracyCurve(
        epochs=epochs,
        mean_accuracy=gmp_acc,
        conf_interval=np.zeros_like(gmp_acc),
        method="gmp",
        split="T2_shortcut_normal",
        n_seeds=5
    )

    return {
        "Scratch_T2_T2_shortcut_normal": scratch_curve,
        "sgd_T2_shortcut_normal": sgd_curve,
        "ewc_on_T2_shortcut_normal": ewc_curve,
        "derpp_T2_shortcut_normal": derpp_curve,
        "gmp_T2_shortcut_normal": gmp_curve
    }


def main():
    """Generate demonstration heatmap."""
    print("Creating synthetic accuracy curves...")
    curves = create_synthetic_curves()

    print("Initializing ERIHeatmapPlotter...")
    plotter = ERIHeatmapPlotter()

    print("Computing tau sensitivity analysis...")
    # Use tau range from requirements: [0.50, 0.80], step 0.05
    taus = np.arange(0.50, 0.81, 0.05)
    sensitivity_result = plotter.compute_tau_sensitivity(curves, taus.tolist())

    print(f"Sensitivity matrix shape: {sensitivity_result.ad_matrix.shape}")
    print(f"Methods: {sensitivity_result.methods}")
    print(f"Tau values: {sensitivity_result.taus}")

    # Print some sample AD values
    print("\nSample AD(τ) values:")
    for i, method in enumerate(sensitivity_result.methods):
        for j, tau in enumerate(sensitivity_result.taus):
            if j % 3 == 0:  # Print every 3rd tau value
                ad_val = sensitivity_result.ad_matrix[i, j]
                if not np.isnan(ad_val):
                    print(f"  {method}, τ={tau:.2f}: AD = {ad_val:.1f} epochs")
                else:
                    print(f"  {method}, τ={tau:.2f}: AD = NaN (censored)")

    print("\nGenerating heatmap...")
    fig = plotter.create_tau_sensitivity_heatmap(
        sensitivity_result,
        title="AD(τ) Robustness Analysis - Synthetic Data"
    )

    # Save the heatmap
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "fig_ad_tau_heatmap_demo.pdf"
    plotter.save_heatmap(fig, str(output_path))

    print(f"Heatmap saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Also save as PNG for easy viewing
    png_path = output_path.with_suffix('.png')
    plotter.save_heatmap(fig, str(png_path), format='png', dpi=150)
    print(f"PNG version saved to: {png_path}")

    # Show summary statistics
    valid_entries = ~np.isnan(sensitivity_result.ad_matrix)
    n_valid = np.sum(valid_entries)
    n_total = sensitivity_result.ad_matrix.size
    n_censored = n_total - n_valid

    print(f"\nSummary:")
    print(f"  Total entries: {n_total}")
    print(f"  Valid AD values: {n_valid}")
    print(f"  Censored entries: {n_censored}")
    print(f"  Censoring rate: {n_censored/n_total*100:.1f}%")

    if n_valid > 0:
        valid_ads = sensitivity_result.ad_matrix[valid_entries]
        print(f"  AD range: [{np.min(valid_ads):.1f}, {np.max(valid_ads):.1f}] epochs")
        print(f"  Mean AD: {np.mean(valid_ads):.1f} epochs")

    plt.close(fig)
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
