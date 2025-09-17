#!/usr/bin/env python3
"""
Integration test for ERIDynamicsPlotter with realistic data.

Improved layout & rendering: single suptitle (no duplicate),
figure-level legend in the top margin, larger figure size,
extra vertical spacing, and high-DPI PNG + SVG output.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from eri_vis.plot_dynamics import ERIDynamicsPlotter
from eri_vis.processing import AccuracyCurve, TimeSeries
from eri_vis.styles import PlotStyleConfig


def create_realistic_test_data():
    # (copy your existing create_realistic_test_data implementation)
    epochs = np.linspace(0, 20, 41)
    methods_data = {
        'Scratch_T2': {
            'patched': 0.1 + 0.85 * (1 - np.exp(-epochs / 3)),
            'masked': 0.05 + 0.15 * epochs / 20,
        },
        'sgd': {
            'patched': 0.1 + 0.75 * (1 - np.exp(-epochs / 5)),
            'masked': 0.05 + 0.20 * epochs / 20,
        },
        'ewc_on': {
            'patched': 0.1 + 0.65 * (1 - np.exp(-epochs / 7)),
            'masked': 0.05 + 0.18 * epochs / 20,
        },
        'derpp': {
            'patched': 0.1 + 0.70 * (1 - np.exp(-epochs / 6)),
            'masked': 0.05 + 0.22 * epochs / 20,
        }
    }

    np.random.seed(42)
    patched_curves = {}
    masked_curves = {}

    for method, data in methods_data.items():
        noise_scale = 0.03
        patched_noise = np.random.normal(0, noise_scale, len(epochs))
        masked_noise = np.random.normal(0, noise_scale, len(epochs))
        ci_scale = 0.04 if method == 'Scratch_T2' else 0.06

        patched_curves[f"{method}_T2_shortcut_normal"] = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.clip(data['patched'] + patched_noise, 0, 1),
            conf_interval=ci_scale * np.ones_like(epochs),
            method=method,
            split='T2_shortcut_normal',
            n_seeds=5
        )

        masked_curves[f"{method}_T2_shortcut_masked"] = AccuracyCurve(
            epochs=epochs,
            mean_accuracy=np.clip(data['masked'] + masked_noise, 0, 1),
            conf_interval=ci_scale * np.ones_like(epochs),
            method=method,
            split='T2_shortcut_masked',
            n_seeds=5
        )

    # Performance deficit series
    pd_series = {}
    scratch_patched = patched_curves['Scratch_T2_T2_shortcut_normal'].mean_accuracy
    for method in ['sgd', 'ewc_on', 'derpp']:
        method_patched = patched_curves[f"{method}_T2_shortcut_normal"].mean_accuracy
        pd_series[method] = TimeSeries(
            epochs=epochs,
            values=scratch_patched - method_patched,
            conf_interval=np.array([]),
            method=method,
            metric_name='PD_t'
        )

    # Shortcut forgetting rate
    sfr_series = {}
    scratch_patched_acc = patched_curves['Scratch_T2_T2_shortcut_normal'].mean_accuracy
    scratch_masked_acc = masked_curves['Scratch_T2_T2_shortcut_masked'].mean_accuracy
    delta_scratch = scratch_patched_acc - scratch_masked_acc

    for method in ['sgd', 'ewc_on', 'derpp']:
        method_patched_acc = patched_curves[f"{method}_T2_shortcut_normal"].mean_accuracy
        method_masked_acc = masked_curves[f"{method}_T2_shortcut_masked"].mean_accuracy
        delta_method = method_patched_acc - method_masked_acc
        sfr_series[method] = TimeSeries(
            epochs=epochs,
            values=delta_method - delta_scratch,
            conf_interval=np.array([]),
            method=method,
            metric_name='SFR_rel'
        )

    # Adaptation Delay (AD)
    tau = 0.6
    ad_values = {}
    scratch_acc = patched_curves['Scratch_T2_T2_shortcut_normal'].mean_accuracy
    scratch_crossing_idx = np.where(scratch_acc >= tau)[0]
    scratch_crossing_epoch = epochs[scratch_crossing_idx[0]] if len(scratch_crossing_idx) > 0 else np.nan

    for method in ['sgd', 'ewc_on', 'derpp']:
        method_acc = patched_curves[f"{method}_T2_shortcut_normal"].mean_accuracy
        method_crossing_idx = np.where(method_acc >= tau)[0]
        if len(method_crossing_idx) > 0 and not np.isnan(scratch_crossing_epoch):
            ad_values[method] = epochs[method_crossing_idx[0]] - scratch_crossing_epoch
        else:
            ad_values[method] = np.nan

    return patched_curves, masked_curves, pd_series, sfr_series, ad_values


def tidy_figure_for_image(fig, suptitle_text, target_size=(12, 11)):
    """
    Adjust the figure returned by the plotter:
      - set a single suptitle
      - remove the top axis title (avoid duplication)
      - collect and remove axis legends, produce one figure-level legend
      - increase figure size and spacing for better image rendering
    """

    # Slightly larger base fonts for readability in images:
    mpl.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.titlepad': 6,
    })

    # Set output figure size (inches)
    fig.set_size_inches(*target_size)

    # Put a single suptitle
    fig.suptitle(suptitle_text, fontsize=16, fontweight='bold', y=0.995)

    # --- Gather handles/labels from axes BEFORE removing per-axis legends ---
    unique_labels = []
    unique_handles = []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in unique_labels:
                unique_labels.append(ll)
                unique_handles.append(hh)

    # Remove per-axis legends (they tend to overlap with the lines)
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Remove the top panel axis title to avoid overlapping with the suptitle.
    # The top panel is usually the first axes returned by the plotter.
    if len(fig.axes) > 0:
        top_ax = fig.axes[0]
        # If top axis title is identical or too long, clear it (we already have suptitle).
        top_ax.set_title('')

    # Make room for the suptitle + legend: push subplots down a bit and add vertical space
    fig.subplots_adjust(top=0.87, hspace=0.34, left=0.08, right=0.98)

    # Create a single figure-level legend centered in the top margin
    if unique_handles:
        # number of columns chosen to reduce horizontal crowding
        ncol = min(len(unique_handles), 4)
        fig.legend(
            unique_handles,
            unique_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.94),
            ncol=ncol,
            frameon=True,
            fontsize=10,
            columnspacing=1.2
        )

    # Final layout pass
    fig.canvas.draw()  # ensure text extents are computed
    # Use tight layout limited by a rect to preserve the top margin we explicitly reserved:
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    except Exception:
        # tight_layout may sometimes conflict with constrained_layout; ignore if it fails
        pass

    return fig


def main():
    print("Creating realistic ERI test data...")
    patched_curves, masked_curves, pd_series, sfr_series, ad_values = create_realistic_test_data()

    print("Initializing ERIDynamicsPlotter...")
    # You can still use the publication-ready style; we override fonts via rcParams above.
    style = PlotStyleConfig.create_publication_style()
    plotter = ERIDynamicsPlotter(style=style)

    print("Creating 3-panel dynamics figure (no title passed to avoid duplication)...")
    # don't pass title so we can set a single suptitle later
    fig = plotter.create_dynamics_figure(
        patched_curves=patched_curves,
        masked_curves=masked_curves,
        pd_series=pd_series,
        sfr_series=sfr_series,
        ad_values=ad_values,
        tau=0.6,
    )

    # Tidy layout and rendering for image output
    final_title = "ERI Dynamics: Shortcut-Induced Rigidity Analysis"
    fig = tidy_figure_for_image(fig, final_title, target_size=(12, 11))

    # Save as high-DPI PNG and also SVG (vector)
    out_png = "eri_dynamics_test.png"
    out_svg = "eri_dynamics_test.svg"
    print(f"Saving high-DPI PNG to {out_png} and SVG to {out_svg} ...")
    try:
        fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.06)
        fig.savefig(out_svg, bbox_inches='tight', pad_inches=0.06)
    except Exception as exc:
        # fallback to plotter API saver if it has special hooks
        print("Warning: fig.savefig failed, trying plotter.save_figure():", exc)
        try:
            plotter.save_figure(fig, out_png)
            plotter.save_figure(fig, out_svg)
        except Exception as exc2:
            print("Fallback saver failed:", exc2)
            raise

    # report sizes
    for path in (out_png, out_svg):
        file_path = Path(path)
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  {path}: {size_mb:.2f} MB")

    # Also produce the quick plot (tidy and save)
    print("Creating quick dynamics plot for comparison...")
    quick_fig = plotter.create_quick_dynamics_plot(patched_curves, tau=0.6)
    # smaller size for quick plot
    quick_fig.set_size_inches(10, 4)
    try:
        quick_fig.tight_layout()
    except Exception:
        pass

    quick_out = "eri_quick_test.png"
    try:
        quick_fig.savefig(quick_out, dpi=200, bbox_inches='tight', pad_inches=0.04)
    except Exception:
        try:
            plotter.save_figure(quick_fig, quick_out)
        except Exception as exc:
            print("Failed saving quick plot:", exc)

    print("Done — improved spacing and high-DPI + vector outputs saved.")
    plt.close('all')


if __name__ == "__main__":
    main()
