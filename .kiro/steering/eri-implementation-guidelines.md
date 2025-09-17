---
inclusion: fileMatch
fileMatchPattern: "eri_vis/**/*.py"
---

# ERI Implementation Guidelines

## Core ERI Concepts and Definitions

### Effective Epochs

```python
# Always use effective epochs for fair method comparison
def compufective_epoch(optimizer_steps: int, task2_train_size: int) -> float:
    """Compute effective epoch normalizing for replay methods.

    Effective epoch = (Phase 2 optimizer updates consuming T2 samples) / T2 train size
    This normalization ensures fair comparison between methods with different
    replay ratios and buffer sizes.
    """
    return optimizer_steps / task2_train_size
```

### Split Definitions

Maintain strict consistency in split definitions across all implementations:

```python
REQUIRED_SPLITS = {
    "T1_all": "All Task 1 test samples",
    "T2_shortcut_normal": "Task 2 samples with shortcuts present (patched)",
    "T2_shortcut_masked": "Task 2 samples with shortcuts removed (masked)",
    "T2_nonshortcut_normal": "Task 2 samples without shortcuts (clean)"
}
```

### Metric Computation Standards

#### Adaptation Delay (AD)

```python
def compute_adaptation_delay(
    accuracy_curve: np.ndarray,
    epochs: np.ndarray,
    tau: float = 0.6,
    smoothing_window: int = 3
) -> float:
    """Compute Adaptation Delay as first epoch where smoothed accuracy >= tau.

    Returns NaN if threshold is never crossed (censored run).
    Always apply smoothing before threshold detection.
    """
    smoothed = apply_smoothing(accuracy_curve, window=smoothing_window)
    crossing_indices = np.where(smoothed >= tau)[0]
    return epochs[crossing_indices[0]] if len(crossing_indices) > 0 else np.nan
```

#### Performance Deficit (PD_t)

```python
def compute_performance_deficit(
    cl_accuracy: np.ndarray,
    scratch_accuracy: np.ndarray,
    epochs: np.ndarray
) -> TimeSeries:
    """Compute PD_t(e) = A_S(e) - A_CL(e) on aligned epochs.

    Positive values indicate CL method underperforms scratch baseline.
    """
    aligned_epochs = align_epoch_grids(epochs)
    cl_interp = interpolate_to_grid(cl_accuracy, epochs, aligned_epochs)
    scratch_interp = interpolate_to_grid(scratch_accuracy, epochs, aligned_epochs)
    return TimeSeries(aligned_epochs, scratch_interp - cl_interp)
```

#### Shortcut Forgetting Rate (SFR_rel)

```python
def compute_sfr_relative(
    patched_acc: np.ndarray,
    masked_acc: np.ndarray,
    epochs: np.ndarray,
    scratch_patched: np.ndarray,
    scratch_masked: np.ndarray
) -> TimeSeries:
    """Compute SFR_rel(e) = Δ_CL(e) - Δ_S(e).

    Where Δ_M(e) = Acc(M, SC_patched, e) - Acc(M, SC_masked, e)
    Positive values indicate CL method relies more on shortcuts than scratch.
    """
    delta_cl = patched_acc - masked_acc
    delta_scratch = scratch_patched - scratch_masked
    return TimeSeries(epochs, delta_cl - delta_scratch)
```

## Data Processing Pipeline Standards

### CSV Schema Validation

```python
REQUIRED_COLUMNS = ["method", "seed", "epoch_eff", "split", "acc"]
COLUMN_TYPES = {
    "method": str,
    "seed": int,
    "epoch_eff": float,
    "split": str,
    "acc": float
}
VALID_RANGES = {
    "seed": (0, 10000),
    "epoch_eff": (0.0, float('inf')),
    "acc": (0.0, 1.0)
}
```

### Smoothing Implementation

```python
def apply_smoothing(
    values: np.ndarray,
    window: int = 3,
    mode: str = 'edge_padded'
) -> np.ndarray:
    """Apply centered moving average with edge padding.

    Edge padding ensures smoothed curve has same length as input.
    Use consistent smoothing across all accuracy curves.
    """
    if window <= 1:
        return values

    # Pad edges to maintain length
    padded = np.pad(values, (window//2, window//2), mode='edge')
    smoothed = np.convolve(padded, np.ones(window)/window, mode='valid')
    return smoothed
```

### Confidence Interval Computation

```python
def compute_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute confidence intervals using t-distribution.

    Returns mean and half-width of confidence interval.
    Document sample size (N seeds) in metadata.
    """
    from scipy import stats

    mean = np.mean(values, axis=axis)
    sem = stats.sem(values, axis=axis)
    n = values.shape[axis]

    # Use t-distribution for small sample sizes
    t_val = stats.t.ppf((1 + confidence) / 2, df=n-1)
    ci_half_width = t_val * sem

    return mean, ci_half_width
```

## Visualization Implementation Standards

### Color Palette Consistency

```python
DEFAULT_METHOD_COLORS = {
    "Scratch_T2": "#333333",      # Dark gray for baseline
    "sgd": "#1f77b4",             # Blue for naive CL
    "ewc_on": "#2ca02c",          # Green for EWC
    "derpp": "#d62728",           # Red for DER++
    "gmp": "#8c564b",             # Brown for GPM
    "Interleaved": "#9467bd",     # Purple for oracle
}
```

### Figure Layout Standards

```python
def create_three_panel_figure(figsize=(12, 10), dpi=300):
    """Standard 3-panel layout for ERI dynamics visualization."""
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)

    # Panel A: Accuracy trajectories with AD markers
    # Panel B: Performance Deficit (PD_t) time series
    # Panel C: Shortcut Forgetting Rate (SFR_rel) time series

    plt.tight_layout(pad=3.0)
    return fig, axes
```

### Annotation Standards

```python
def add_ad_annotations(ax, ad_values: Dict[str, float], tau: float):
    """Add consistent AD markers and annotations to accuracy plots."""
    for method, ad_value in ad_values.items():
        if not np.isnan(ad_value):
            ax.axvline(ad_value, linestyle='--', alpha=0.7,
                      color=DEFAULT_METHOD_COLORS[method])
            ax.annotate(f'AD={ad_value:.1f}',
                       xy=(ad_value, tau),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9)
```

## Error Handling and Validation

### Structured Error Types

```python
class ERIDataValidationError(Exception):
    """Raised when input data fails validation checks."""
    pass

class ERIProcessingError(Exception):
    """Raised when metric computation encounters issues."""
    pass

class ERIVisualizationError(Exception):
    """Raised when figure generation fails."""
    pass
```

### Censored Data Handling

```python
def handle_censored_runs(ad_values: Dict[str, float]) -> Dict[str, float]:
    """Handle runs that never cross threshold (censored data).

    Mark as NaN and log warning with method context.
    Include censoring information in figure legends.
    """
    censored_methods = [m for m, ad in ad_values.items() if np.isnan(ad)]

    if censored_methods:
        logger.warning(f"Censored runs (no threshold crossing): {censored_methods}")

    return ad_values
```

## Integration Patterns

### Mammoth Hook Integration

```python
class ERIVisualizationHooks:
    def __init__(self, output_dir: str, export_frequency: int = 1):
        self.output_dir = Path(output_dir)
        self.export_frequency = export_frequency
        self.timeline_data = []

    def on_epoch_end(self, epoch: int, evaluator: EinstellungEvaluator):
        """Collect subset accuracies for timeline visualization."""
        if epoch % self.export_frequency == 0:
            subset_accs = evaluator.get_subset_accuracies()
            self.timeline_data.append({
                'epoch': epoch,
                'subset_accuracies': subset_accs,
                'timestamp': time.time()
            })

    def on_experiment_end(self, evaluator: EinstellungEvaluator):
        """Export timeline data and generate visualizations."""
        csv_path = self.output_dir / "eri_sc_metrics.csv"
        self.export_timeline_csv(csv_path)
        self.generate_visualizations(csv_path)
```

### Deterministic Output Generation

```python
def export_deterministic_csv(
    dataset: ERITimelineDataset,
    output_path: Path
) -> None:
    """Export CSV with deterministic ordering for reproducibility."""
    # Sort by method, seed, epoch_eff for consistent output
    df = dataset.data.sort_values(['method', 'seed', 'epoch_eff'])

    # Generate metadata sidecar
    metadata = {
        'export_timestamp': datetime.now().isoformat(),
        'methods': sorted(dataset.methods),
        'seeds': sorted(dataset.seeds),
        'splits': sorted(dataset.splits),
        'epoch_range': dataset.epoch_range,
        'git_sha': get_git_sha() if available else None
    }

    df.to_csv(output_path, index=False)
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
```

## Testing and Validation Requirements

### Synthetic Data Testing

```python
def create_synthetic_eri_data() -> ERITimelineDataset:
    """Generate synthetic data with known AD/PD_t/SFR_rel values for testing."""
    # Create curves with predictable threshold crossings
    # Include censored runs for robustness testing
    # Vary noise levels to test confidence interval computation
    pass

def test_metric_computation_accuracy():
    """Verify metric computations against known synthetic results."""
    synthetic_data = create_synthetic_eri_data()
    computed_ad = compute_adaptation_delay(synthetic_data.get_accuracy_curve())
    expected_ad = synthetic_data.metadata['expected_ad']
    assert abs(computed_ad - expected_ad) < 0.1
```

### Visual Regression Testing

```python
def test_figure_generation_consistency():
    """Ensure figure generation produces consistent outputs."""
    test_data = load_test_dataset()
    fig = generate_dynamics_figure(test_data)

    # Compare against golden reference image
    reference_hash = load_reference_image_hash()
    current_hash = compute_figure_hash(fig)
    assert current_hash == reference_hash
```
