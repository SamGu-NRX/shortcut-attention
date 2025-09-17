# ERI Visualization System — Design (v1.1)

## Overview

- Modular architecture separating data, processing, visualization, integration.
- Strict data schemas, deterministic processing, robust error handling.
- Method-agnostic; minimal assumptions about training internals.

## High-Level Architecture

```
+---------------------------------------------------------------+
|                         eri_vis package                       |
+-------------------+-------------------+-----------------------+
| Data Layer        | Processing Layer  | Visualization Layer   |
| - ERIDataLoader   | - ERITimelineProc | - ERIDynamicsPlotter  |
| - ERITimelineDS   | - ERIStatsCalc    | - ERIHeatmapPlotter   |
+-------------------+-------------------+-----------------------+
| Integration Layer | CLI Layer         | Output Layer          |
| - MammothIntegration                | - PDF/PNG/SVG writer   |
| - ERIExperimentHooks               | - Meta sidecar JSON    |
+---------------------------------------------------------------+
```

## Directory Layout

```
tools/
  plot_eri.py                  # CLI entry; uses eri_vis/*
eri_vis/
  __init__.py
  data_loader.py               # ERIDataLoader
  dataset.py                   # ERITimelineDataset
  processing.py                # ERITimelineProcessor
  statistics.py                # ERIStatisticsCalculator
  plot_dynamics.py             # ERIDynamicsPlotter
  plot_heatmap.py              # ERIHeatmapPlotter
  styles.py                    # PlotStyleConfig
  integration/
    mammoth_integration.py     # MammothERIIntegration
    hooks.py                   # ERIExperimentHooks
  errors.py                    # ERIErrorHandler and exceptions
  utils.py                     # alignment, smoothing, CI utils
experiments/
  configs/
    cifar100_einstellung224.yaml
    imagenet100_einstellung.yaml (optional)
    text_sst2_imdb.yaml        (optional)
  runners/
    run_einstellung.py         # calls Mammoth, evaluator hooks
docs/
  README_eri_vis.md            # user guide & reviewer notes
```

## Core Data Models (Python-like)

```python
# dataset.py
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

@dataclass
class ERITimelineDataset:
    data: pd.DataFrame
    metadata: Dict[str, Any]
    methods: List[str]
    splits: List[str]
    seeds: List[int]
    epoch_range: Tuple[float, float]

    def get_method_data(self, method: str) -> pd.DataFrame: ...
    def get_split_data(self, split: str) -> pd.DataFrame: ...
    def get_seed_data(self, seed: int) -> pd.DataFrame: ...
    def align_epochs(self, epochs: np.ndarray) -> "ERITimelineDataset": ...
```

## Data Loading and Validation

```python
# data_loader.py
class ERIDataLoader:
    REQUIRED_COLS = ["method", "seed", "epoch_eff", "split", "acc"]
    VALID_SPLITS = {
        "T1_all",
        "T2_shortcut_normal",
        "T2_shortcut_masked",
        "T2_nonshortcut_normal",
    }

    def load_csv(self, filepath: str) -> ERITimelineDataset: ...
    def load_from_evaluator_export(self, export: dict) -> ERITimelineDataset: ...
    def validate_format(self, df: pd.DataFrame) -> None: ...
    def convert_legacy_format(self, legacy: dict) -> ERITimelineDataset: ...
```

## Processing Algorithms

- **Smoothing**: centered moving average window w (default 3), edge-padded
- **CI across seeds**: per-epoch t-interval; store mean, CI, N
- **Alignment**: linear interpolation onto common epoch grid per comparison
- **AD**:
  - E_M(τ): first index where smoothed mean accuracy ≥ τ (per method)
  - AD = E_CL(τ) − E_S(τ); NaN if either censored
  - Record censored flags in metadata
- **PD_t(e)**: For each CL method, align to Scratch epochs and compute difference
- **SFR_rel(e)**: Compute Δ_CL(e) and Δ_S(e) on aligned epochs, subtract

## Processing Interfaces

```python
# processing.py
from dataclasses import dataclass
import numpy as np

@dataclass
class AccuracyCurve:
    epochs: np.ndarray
    mean_accuracy: np.ndarray
    conf_interval: np.ndarray  # half-width 95% CI
    method: str
    split: str

@dataclass
class TimeSeries:
    epochs: np.ndarray
    values: np.ndarray
    conf_interval: np.ndarray  # optional (may be empty)

class ERITimelineProcessor:
    def __init__(self, smoothing_window: int = 3, tau: float = 0.6): ...

    def compute_accuracy_curves(
        self, ds: ERITimelineDataset
    ) -> dict[str, AccuracyCurve]: ...

    def compute_adaptation_delays(
        self, curves: dict[str, AccuracyCurve]
    ) -> dict[str, float]: ...

    def compute_performance_deficits(
        self, curves: dict[str, AccuracyCurve], scratch_key: str = "Scratch_T2"
    ) -> dict[str, TimeSeries]: ...

    def compute_sfr_relative(
        self, curves: dict[str, AccuracyCurve], scratch_key: str = "Scratch_T2"
    ) -> dict[str, TimeSeries]: ...
```

## Statistics (Robustness)

```python
# statistics.py
@dataclass
class TauSensitivityResult:
    methods: list[str]
    taus: np.ndarray
    ad_matrix: np.ndarray  # shape (len(methods), len(taus)), NaN allowed

class ERIStatisticsCalculator:
    def compute_tau_sensitivity(
        self, curves: dict[str, AccuracyCurve], taus: list[float], scratch="Scratch_T2"
    ) -> TauSensitivityResult: ...

    def compute_method_comparisons(...): ...
```

## Visualization

```python
# styles.py
from dataclasses import dataclass, field

@dataclass
class PlotStyleConfig:
    figure_size: tuple[int, int] = (12, 10)
    dpi: int = 300
    color_palette: dict[str, str] = field(
        default_factory=lambda: {
            "Scratch_T2": "#333333",
            "sgd": "#1f77b4",
            "ewc_on": "#2ca02c",
            "derpp": "#d62728",
            "gmp": "#8c564b",
            "Interleaved": "#9467bd",
        }
    )
    font_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "title": 14,
            "axis_label": 12,
            "legend": 10,
            "annotation": 9,
        }
    )
    confidence_alpha: float = 0.15
```

```python
# plot_dynamics.py
class ERIDynamicsPlotter:
    def __init__(self, style: PlotStyleConfig | None = None): ...

    def create_dynamics_figure(
        self,
        patched_curves: dict[str, AccuracyCurve],
        masked_curves: dict[str, AccuracyCurve],
        pd_series: dict[str, TimeSeries],
        sfr_series: dict[str, TimeSeries],
        ad_values: dict[str, float],
        tau: float,
    ) -> "matplotlib.figure.Figure": ...
```

```python
# plot_heatmap.py
class ERIHeatmapPlotter:
    def create_tau_sensitivity_heatmap(
        self, sens: TauSensitivityResult
    ) -> "matplotlib.figure.Figure": ...
```

## Integration with Mammoth

- **Hook points**:
  - EinstellungEvaluator.after_phase2_epoch(e): collect SC patched/masked accuracies (macro over SC superclasses)
  - export_results(path): writes deterministic CSV in required schema
- **File naming**:
  - logs/{exp_id}/eri_sc_metrics.csv
  - logs/{exp_id}/figs/fig_eri_dynamics.pdf
  - logs/{exp_id}/figs/fig_ad_tau_heatmap.pdf

```python
# integration/mammoth_integration.py
class MammothERIIntegration:
    def __init__(self, evaluator): ...

    def setup_auto_export(self, output_dir: str, export_frequency: int = 1): ...
    def export_timeline_for_visualization(self, filepath: str): ...
    def generate_visualizations_from_evaluator(self, output_dir: str): ...
    def register_visualization_hooks(self): ...
```

## Error Handling

- Centralized ERIErrorHandler for:
  - DataValidationError (missing columns/types)
  - ProcessingError (alignment, censored, empty curves)
  - VisualizationError (backend, file I/O)
- Log structured messages with context (method, seed, epoch)

## Experiment Design Enhancements (improving persuasiveness)

- **Shortcut salience sweeps**:
  - Patch size: 2, 4, 6, 8 px (on 224)
  - Location: fixed TL; randomized among 4 corners (per-image deterministic)
  - Injection ratio: 0.5 and 1.0 of SC samples
- **Baselines**:
  - Add DER++ and GPM configs (match Phase 2 budgets; normalize effective epochs)
- **Additional probes (optional add-ons)**:
  - ECE under masking (SC patched vs masked)
  - CKA drift between Phase 1 and Phase 2 layers
- **Expected improved narrative**:
  - AD remains negative across τ; SFR_rel > 0; PD_t negative early; masked accuracy collapse on SC; NSC robust → visually compelling figures.

## Configs

- experiments/configs/cifar100_einstellung224.yaml:
  - seeds: [1, 2, 3, 4, 5]
  - methods: [Scratch_T2, sgd, ewc_on, derpp, gmp]
  - shortcut: {size: [4, 8], location: [fixed, corners], ratio: [0.5, 1.0]}
  - eval_splits: [T1_all, T2_shortcut_normal, T2_shortcut_masked, T2_nonshortcut_normal]
  - visualize: {tau: 0.6, smooth: 3, tau_grid: [0.5..0.8]}

## Testing Strategy

- **Unit tests**:
  - CSV parsing/validation with good/bad inputs
  - Metric computations (AD, PD_t, SFR_rel) with synthetic known curves
  - Heatmap generation with mixed NaN/non-NaN AD values
- **Integration tests**:
  - Full evaluator export → CSV → figs pipeline
  - ViT vs CNN evaluation frequency alignment
- **Performance**:
  - 10 seeds × 1000 epochs × 5 methods within time/memory limits
- **Visual regression**:
  - Golden images (hash/SSIM) for standard synthetic dataset

## Core Data Flow

```
ERI Experiment → EinstellungEvaluator → Timeline Data → ERITimelineDataset
                                                            ↓
CSV Export ←─────────────────────────────────────── ERIDataLoader
    ↓
ERITimelineProcessor → AccuracyCurves + Statistics
    ↓
ERIDynamicsPlotter + ERIHeatmapPlotter → Publication Figures
```

## Data Format Specifications

### CSV Input Format

```csv
method,seed,epoch_eff,split,acc
Scratch_T2,42,0,T2_shortcut_normal,0.1
Scratch_T2,42,1,T2_shortcut_normal,0.3
Scratch_T2,42,0,T2_shortcut_masked,0.05
sgd,42,0,T2_shortcut_normal,0.2
...
```

### ERI Export Format (from EinstellungEvaluator)

```json
{
  "configuration": {...},
  "timeline_data": [
    {
      "epoch": 0,
      "task_id": 1,
      "subset_accuracies": {
        "T1_all": 0.85,
        "T2_shortcut_normal": 0.1,
        "T2_shortcut_masked": 0.05,
        "T2_nonshortcut_normal": 0.15
      },
      "subset_losses": {...},
      "timestamp": 1640995200.0
    }
  ],
  "final_metrics": {...}
}
```

## Dependencies and Constraints

### External Dependencies

- **matplotlib**: Core plotting functionality (>=3.5.0)
- **seaborn**: Statistical visualization enhancements (>=0.11.0)
- **pandas**: Data manipulation and analysis (>=1.3.0)
- **numpy**: Numerical computations (>=1.21.0)
- **scipy**: Statistical functions (>=1.7.0)

### Internal Dependencies

- **utils.einstellung_evaluator**: Integration with existing ERI evaluation
- **utils.einstellung_metrics**: Metric calculation compatibility
- **datasets.seq_cifar100_einstellung_224**: Dataset integration

### Performance Constraints

- **Memory Usage**: Must handle datasets with 1000+ epochs and 10+ seeds
- **Processing Time**: Visualization generation should complete within 30 seconds
- **File Size**: Generated PDFs should be under 5MB for publication submission

### Compatibility Constraints

- **Python Version**: Support Python 3.8+
- **Matplotlib Backends**: Support both GUI and headless environments
- **File Formats**: Generate publication-ready PDF and optional PNG/SVG
- **Platform Support**: Cross-platform compatibility (Linux, macOS, Windows)
