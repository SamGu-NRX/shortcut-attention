# ERI Visualization System — Design (v1.1)

## 🚨 CRITICAL INTEGRATION REQUIREMENT

**MANDATORY MAMMOTH INTEGRATION**: This design MUST build upon the existing Mammoth Einstellung infrastructure:

- **Existing Dataset**: `datasets/seq_cifar100_einstellung_224.py` - ViT-compatible CIFAR-100 with magenta patch injection
- **Existing Evaluator**: `utils/einstellung_evaluator.py` - Plugin-based evaluation with timeline tracking
- **Existing Metrics**: `utils/einstellung_metrics.py` - ERI calculation framework (if exists)
- **Existing Attention**: `utils/attention_visualization.py` - ViT attention analysis capabilities
- **Existing Runners**: Integration with checkpoint management and experiment orchestration

**Architecture Principle**: Extend, don't replace. Build visualization layer on top of proven Mammoth components.

## Overview

- Modular architecture separating data, processing, visualization, integration.
- Strict data schemas, deterministic processing, robust error handling.
- Method-agnostic; minimal assumptions about training internals.
- **Mammoth-native**: Leverages existing datasets, evaluators, and training pipelines.

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
models/
  gpm_adapter.py               # GPM implementation
  gen_replay.py                # Generative replay implementations
  hybrid_methods.py            # Combined GPM + replay methods
  new_methods_registry.py      # Method registration system
  gpm_model.py                 # GPM ContinualModel wrapper
  class_gaussian_replay_model.py  # Gaussian replay ContinualModel wrapper
  vae_replay_model.py          # VAE replay ContinualModel wrapper
  gpm_gaussian_hybrid_model.py    # Hybrid ContinualModel wrapper
  config/
    gpm.yaml                   # GPM configuration
    class_gaussian_replay.yaml # Gaussian replay configuration
    vae_replay.yaml            # VAE replay configuration
    gpm_gaussian_hybrid.yaml   # Hybrid method configuration
experiments/
  configs/
    cifar100_einstellung224.yaml
    cifar100_new_methods.yaml  # Configuration for new methods
    imagenet100_einstellung.yaml (optional)
    text_sst2_imdb.yaml        (optional)
  runners/
    run_einstellung.py         # calls Mammoth, evaluator hooks
docs/
  README_eri_vis.md            # user guide & reviewer notes
  README_new_methods.md        # documentation for GPM and replay methods
tests/
  models/
    test_gpm_adapter.py        # GPM unit tests
    test_gen_replay.py         # Generative replay unit tests
    test_hybrid_methods.py     # Hybrid method unit tests
    test_new_methods_registry.py  # Registry unit tests
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
            "gpm": "#8c564b",
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

## Integration with Mammoth ✅ COMPLETED

**IMPLEMENTATION STATUS:** Successfully integrated with existing Mammoth infrastructure by extending `run_einstellung_experiment.py`.

- **Hook points** ✅ IMPLEMENTED:
  - EinstellungEvaluator.after_phase2_epoch(e): collect SC patched/masked accuracies (macro over SC superclasses)
  - export_results(path): writes deterministic CSV in required schema
  - Training function patches applied successfully
  - meta_begin_task and meta_end_task hooks registered
- **File naming** ✅ WORKING:
  - logs/{exp_id}/eri_sc_metrics.csv
  - logs/{exp_id}/figs/fig_eri_dynamics.pdf
  - logs/{exp_id}/figs/fig_ad_tau_heatmap.pdf

```python
# integration/mammoth_integration.py - IMPLEMENTED
class MammothERIIntegration:
    def __init__(self, evaluator): ...

    def setup_auto_export(self, output_dir: str, export_frequency: int = 1): ...
    def export_timeline_for_visualization(self, filepath: str): ...
    def generate_visualizations_from_evaluator(self, output_dir: str): ...
    def register_visualization_hooks(self): ...
```

**VERIFIED FUNCTIONALITY:**

- ✅ EinstellungEvaluator integration active
- ✅ All required splits evaluated (T1_all, T2_shortcut_normal, T2_shortcut_masked, T2_nonshortcut_normal)
- ✅ AttentionAnalyzer initialized for ViT models
- ✅ Training pipeline with ERI integration running successfully

## Existing Method Integration Design

### GPM (Gradient Projection Memory) Integration Architecture

```python
# models/gpm_adapter.py
import torch
from typing import Dict, List, Optional

class GPM:
    """Gradient Projection Memory for continual learning.

    Implements SVD-based subspace extraction and orthogonal gradient projection
    as described in the GPM paper (ICLR 2021).
    """

    def __init__(self,
                 model: torch.nn.Module,
                 layer_names: List[str],
                 energy_threshold: float = 0.90,
                 device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.energy_threshold = energy_threshold

        # Validate and store layer modules
        modules = dict(model.named_modules())
        self.layer_modules = {}
        for name in layer_names:
            if name not in modules:
                raise KeyError(f"Layer {name} not found in model")
            self.layer_modules[name] = modules[name]

        self.bases: Dict[str, torch.Tensor] = {}  # name -> (d, r) basis matrix
        self._hooks = []
        self._activations = {}

    def collect_activations(self, loader: torch.utils.data.DataLoader,
                          max_batches: Optional[int] = 200) -> Dict[str, torch.Tensor]:
        """Collect activations from specified layers."""
        # Implementation details in actual code
        pass

    def update_memory(self, loader: torch.utils.data.DataLoader,
                     max_batches: int = 200) -> None:
        """Update GPM bases after task completion."""
        # Implementation details in actual code
        pass

    def project_gradients(self) -> None:
        """Apply orthogonal gradient projection: g ← g - U(U^T g)"""
        # Implementation details in actual code
        pass

    def _svd_basis(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute SVD basis with energy threshold."""
        # Implementation details in actual code
        pass
```

### Generative Replay Architecture

```python
# models/gen_replay.py
import torch
from typing import Dict, List, Tuple, Optional

class ClassGaussianMemory:
    """Simple class-conditional Gaussian replay in feature space."""

    def __init__(self, feat_dim: int, device: Optional[torch.device] = None,
                 min_std: float = 1e-4):
        self.feat_dim = feat_dim
        self.device = device
        self.min_std = min_std
        self.means: Dict[int, torch.Tensor] = {}  # class -> mean vector
        self.stds: Dict[int, torch.Tensor] = {}   # class -> std vector

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Fit per-class Gaussian distributions."""
        # Implementation details in actual code
        pass

    def sample(self, n: int, classes: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample synthetic features and labels."""
        # Implementation details in actual code
        pass

class ConditionalVAE(torch.nn.Module):
    """Optional: Conditional VAE for more sophisticated replay."""

    def __init__(self, feat_dim: int, latent_dim: int = 64, num_classes: int = 100):
        super().__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feat_dim + num_classes, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2 * latent_dim)  # mu and logvar
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + num_classes, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, feat_dim)
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode features to latent space."""
        # Implementation details in actual code
        pass

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to feature space."""
        # Implementation details in actual code
        pass

    def sample(self, n: int, classes: List[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample synthetic features."""
        # Implementation details in actual code
        pass
```

### Hybrid Method Architecture

```python
# models/hybrid_methods.py
from typing import Optional, List, Dict, Any
import torch

class GPMGaussianHybrid:
    """Combines GPM gradient projection with Gaussian generative replay."""

    def __init__(self,
                 model: torch.nn.Module,
                 backbone: torch.nn.Module,
                 classifier: torch.nn.Module,
                 gpm_config: Dict[str, Any],
                 replay_config: Dict[str, Any]):

        self.model = model
        self.backbone = backbone
        self.classifier = classifier

        # Initialize GPM
        self.gpm = GPM(
            model=model,
            layer_names=gpm_config['layer_names'],
            energy_threshold=gpm_config.get('energy_threshold', 0.95)
        )

        # Initialize Gaussian replay
        feat_dim = backbone.output_dim if hasattr(backbone, 'output_dim') else 512
        self.replay_memory = ClassGaussianMemory(
            feat_dim=feat_dim,
            min_std=replay_config.get('min_std', 1e-4)
        )

        self.replay_ratio = replay_config.get('replay_ratio', 1.0)

    def training_step(self, real_x: torch.Tensor, real_y: torch.Tensor,
                     optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> torch.Tensor:
        """Combined training step with replay and GPM projection."""

        # Get real features
        real_features = self.backbone(real_x)

        # Sample replay data if memory available
        if self.replay_memory.means:
            replay_size = int(len(real_x) * self.replay_ratio)
            replay_features, replay_labels = self.replay_memory.sample(replay_size)
            replay_features = replay_features.to(real_x.device)
            replay_labels = replay_labels.to(real_y.device)

            # Combine real and replay data
            combined_features = torch.cat([real_features, replay_features], dim=0)
            combined_labels = torch.cat([real_y, replay_labels], dim=0)
        else:
            combined_features = real_features
            combined_labels = real_y

        # Forward pass through classifier
        outputs = self.classifier(combined_features)
        loss = criterion(outputs, combined_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Apply GPM gradient projection
        self.gpm.project_gradients()

        # Optimizer step
        optimizer.step()

        return loss

    def end_task(self, train_loader: torch.utils.data.DataLoader) -> None:
        """Update both GPM bases and replay memory after task completion."""

        # Update GPM memory
        self.gpm.update_memory(train_loader)

        # Update replay memory
        all_features = []
        all_labels = []

        self.model.eval()
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(next(self.model.parameters()).device)
                features = self.backbone(x)
                all_features.append(features.cpu())
                all_labels.append(y.cpu())

        if all_features:
            combined_features = torch.cat(all_features, dim=0)
            combined_labels = torch.cat(all_labels, dim=0)
            self.replay_memory.fit(combined_features, combined_labels)
```

### Method Registration System

```python
# models/new_methods_registry.py
from typing import Dict, Any, Type
import torch
from utils.conf import ContinualModel

class NewMethodRegistry:
    """Registry for new continual learning methods."""

    METHODS = {
        'gpm': {
            'class': 'GPMModel',
            'config_file': 'models/config/gpm.yaml',
            'description': 'Gradient Projection Memory with SVD-based subspace extraction'
        },
        'class_gaussian_replay': {
            'class': 'ClassGaussianReplayModel',
            'config_file': 'models/config/class_gaussian_replay.yaml',
            'description': 'Class-conditional Gaussian replay in feature space'
        },
        'vae_replay': {
            'class': 'VAEReplayModel',
            'config_file': 'models/config/vae_replay.yaml',
            'description': 'Conditional VAE-based generative replay'
        },
        'gpm_gaussian_hybrid': {
            'class': 'GPMGaussianHybridModel',
            'config_file': 'models/config/gpm_gaussian_hybrid.yaml',
            'description': 'Hybrid method combining GPM and Gaussian replay'
        }
    }

    @classmethod
    def get_available_methods(cls) -> List[str]:
        return list(cls.METHODS.keys())

    @classmethod
    def create_method(cls, method_name: str, backbone: torch.nn.Module,
                     loss: torch.nn.Module, args) -> ContinualModel:
        if method_name not in cls.METHODS:
            raise ValueError(f"Unknown method: {method_name}")

        method_info = cls.METHODS[method_name]

        # Import and instantiate the appropriate model class
        if method_name == 'gpm':
            from models.gpm_model import GPMModel
            return GPMModel(backbone, loss, args)
        elif method_name == 'class_gaussian_replay':
            from models.class_gaussian_replay_model import ClassGaussianReplayModel
            return ClassGaussianReplayModel(backbone, loss, args)
        elif method_name == 'vae_replay':
            from models.vae_replay_model import VAEReplayModel
            return VAEReplayModel(backbone, loss, args)
        elif method_name == 'gpm_gaussian_hybrid':
            from models.gpm_gaussian_hybrid_model import GPMGaussianHybridModel
            return GPMGaussianHybridModel(backbone, loss, args)

# Integration with existing Mammoth model loading
def get_model(backbone: torch.nn.Module, loss: torch.nn.Module, args) -> ContinualModel:
    """Extended model factory supporting new methods."""

    # Check if it's one of our new methods
    if args.model in NewMethodRegistry.get_available_methods():
        return NewMethodRegistry.create_method(args.model, backbone, loss, args)

    # Fall back to existing Mammoth model loading
    # ... existing Mammoth logic ...
```

### Configuration Files

```yaml
# models/config/gpm.yaml
model: gpm
layer_names:
  - "backbone.layer3"  # For ResNet
  - "head"
energy_threshold: 0.95
max_collection_batches: 200
device: "auto"  # auto-detect GPU/CPU

# models/config/class_gaussian_replay.yaml
model: class_gaussian_replay
feat_dim: 512  # Will be auto-detected from backbone
min_std: 1e-4
replay_ratio: 1.0  # 1:1 ratio with real samples
memory_device: "cpu"  # Store memory on CPU to save GPU memory

# models/config/vae_replay.yaml
model: vae_replay
latent_dim: 64
vae_lr: 1e-3
vae_epochs: 3
vae_batch_size: 128
replay_ratio: 0.5

# models/config/gpm_gaussian_hybrid.yaml
model: gpm_gaussian_hybrid
gpm:
  layer_names: ["backbone.layer3", "head"]
  energy_threshold: 0.95
  max_collection_batches: 200
replay:
  min_std: 1e-4
  replay_ratio: 1.0
  memory_device: "cpu"
```

### Method-Agnostic Evaluation

The ERI visualization system automatically supports any method that:

1. Inherits from Mammoth's `ContinualModel`
2. Integrates with the existing `EinstellungEvaluator`
3. Provides standard accuracy metrics on required splits

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
  - methods: [Scratch_T2, sgd, ewc_on, derpp, gpm]
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
ERI Experiment (with new methods) → EinstellungEvaluator → Timeline Data → ERITimelineDataset
         ↓                                                                        ↓
New Methods Integration:                                              CSV Export ←─── ERIDataLoader
- GPM: gradient projection                                                ↓
- Generative Replay: feature synthesis                    ERITimelineProcessor → AccuracyCurves + Statistics
- Hybrid: combined approach                                               ↓
         ↓                                                ERIDynamicsPlotter + ERIHeatmapPlotter → Publication Figures
Method-agnostic evaluation                                               ↓
through existing ERI pipeline                            Enhanced visualizations with new method comparisons
```

## New Method Integration Points

### Training Loop Integration

```python
# Pseudocode for method integration in training loop
for task_id, train_loader in enumerate(tasks):

    # Initialize method-specific components
    if method_name == 'gpm':
        gpm = GPM(model, layer_names, energy_threshold)
    elif method_name == 'class_gaussian_replay':
        replay_memory = ClassGaussianMemory(feat_dim)
    elif method_name == 'gpm_gaussian_hybrid':
        hybrid_method = GPMGaussianHybrid(model, backbone, classifier, gpm_config, replay_config)

    # Training epochs
    for epoch in range(epochs_per_task):
        for batch_x, batch_y in train_loader:

            if method_name == 'gpm_gaussian_hybrid':
                # Use hybrid training step
                loss = hybrid_method.training_step(batch_x, batch_y, optimizer, criterion)
            else:
                # Standard training with method-specific modifications
                if method_name == 'class_gaussian_replay' and replay_memory.means:
                    # Augment batch with replay samples
                    replay_features, replay_labels = replay_memory.sample(len(batch_x))
                    # ... combine real and replay data

                # Forward pass
                outputs = model(batch_x)  # or combined data
                loss = criterion(outputs, batch_y)  # or combined labels

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                if method_name == 'gpm':
                    # Apply gradient projection
                    gpm.project_gradients()

                optimizer.step()

            # Standard ERI evaluation hooks (method-agnostic)
            if epoch % eval_frequency == 0:
                einstellung_evaluator.evaluate_all_subsets(model, epoch)

    # End-of-task updates
    if method_name == 'gpm':
        gpm.update_memory(train_loader)
    elif method_name == 'class_gaussian_replay':
        # Collect features and update memory
        features, labels = collect_features(model.backbone, train_loader)
        replay_memory.fit(features, labels)
    elif method_name == 'gpm_gaussian_hybrid':
        hybrid_method.end_task(train_loader)

    # Standard ERI end-of-task evaluation (method-agnostic)
    einstellung_evaluator.end_task_evaluation(task_id)

# Final visualization generation (method-agnostic)
einstellung_evaluator.export_results('eri_results.csv')
generate_eri_visualizations('eri_results.csv', output_dir)
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

## Computational Considerations for New Methods

### Memory Management

**GPM Memory Requirements:**

- Basis storage: O(sum(d_i \* r_i)) where d_i is layer dimension, r_i is basis rank
- Activation collection: Temporary storage during SVD computation
- Gradient projection: In-place operations to minimize memory overhead

**Generative Replay Memory Requirements:**

- Class-Conditional Gaussian: O(num*classes * feat*dim * 2) for means and stds
- VAE Replay: O(model_parameters + latent_samples) during training
- Feature collection: Temporary storage during memory updates

**Hybrid Method Considerations:**

- Combined memory from both GPM and replay components
- Coordinate memory updates to prevent excessive GPU memory usage
- Option to store replay memory on CPU while keeping GPM bases on GPU

### Computational Complexity

**GPM Computational Cost:**

- SVD computation: O(d^2 \* n) where d is feature dimension, n is number of samples
- Gradient projection: O(d \* r) per layer per backward pass
- Memory update frequency: Once per task (amortized cost)

**Generative Replay Computational Cost:**

- Gaussian fitting: O(n \* d) per class per task
- VAE training: O(epochs _ batches _ model_forward_cost) per task
- Sample generation: O(replay_samples \* generation_cost) per training batch

**Optimization Strategies:**

- Use global average pooling for convolutional layers to reduce dimensionality
- Implement efficient QR decomposition for basis compression
- Support configurable memory update frequencies
- Provide CPU/GPU memory management options

### Scalability Guidelines

**Recommended Configurations:**

- Small models (ResNet18): Full GPM + Gaussian replay feasible
- Large models (ViT-Large): Use selective layer GPM + reduced replay ratios
- Limited GPU memory: Store replay memory on CPU, use basis compression
- High-throughput scenarios: Reduce SVD sample counts, increase update intervals

**Performance Monitoring:**

- Track basis growth over tasks and provide compression warnings
- Monitor memory usage during activation collection
- Log computational overhead for each method component
- Provide profiling utilities for method comparison
