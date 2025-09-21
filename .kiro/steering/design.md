---
inclusion: fileMatch
fileMatchPattern: "experiments/**/*.py"
---

# Continual Learning Experiment Design Standards

## Experimental Rigor Requirements

### Reproducibility Standards

- Use fixed randoss all experimental runs
- Document all hyperparameters and configuration settings
- Maintain version control for datasets, models, and evaluation protocols
- Provide deterministic data loading and preprocessing pipelines
- Include environment specifications (Python version, package versions)

### Statistical Power and Design

- Use minimum 5 seeds for statistical reliability (prefer 10+ for publication)
- Implement proper train/validation/test splits with no data leakage
- Normalize evaluation metrics across different replay methods using effective epochs
- Include appropriate baseline comparisons (naive, upper bound, oracle)
- Document sample sizes and power analysis where applicable

### Evaluation Protocol Consistency

- Standardize evaluation frequency and checkpointing procedures
- Use consistent accuracy computation across all methods and splits
- Implement proper cross-validation where applicable
- Document evaluation subset definitions clearly
- Ensure fair comparison across different architectural choices

## ERI-Specific Experimental Standards

### Shortcut Learning Evaluation

- Define shortcut and non-shortcut splits consistently
- Use standardized shortcut injection protocols (patch size, location, ratio)
- Implement proper masking procedures for shortcut evaluation
- Document shortcut salience parameters and their effects
- Include robustness analysis across different shortcut configurations

### Phase-Based Training Protocol

```python
# Standard ERI experimental setup
class ERIExperimentConfig:
    phase1_epochs: int = 50      # Task 1 training duration
    phase2_epochs: int = 100     # Task 2 training duration
    eval_frequency: int = 1      # Evaluation every N epochs
    shortcut_ratio: float = 0.5  # Fraction of images with shortcuts
    patch_size: int = 4          # Shortcut patch size in pixels
    seeds: List[int] = [1,2,3,4,5]  # Random seeds for reproducibility
```

### Metric Computation Standards

- Compute Adaptation Delay (AD) using first threshold crossing
- Calculate Performance Deficit (PD_t) relative to scratch baseline
- Measure Shortcut Forgetting Rate (SFR_rel) with proper normalization
- Handle censored runs (no threshold crossing) with NaN values
- Include confidence intervals for all aggregate statistics

## Integration with Mammoth Framework

### Hook Implementation

```python
# Standard hook pattern for ERI experiments
class ERIVisualizationHooks:
    def on_epoch_end(self, epoch: int, evaluator: EinstellungEvaluator):
        """Collect accuracy data for visualization pipeline."""
        # Extract subset accuracies for all required splits
        # Log to structured format for downstream processing

    def on_experiment_end(self, evaluator: EinstellungEvaluator):
        """Generate final visualizations and export data."""
        # Export timeline data to CSV format
        # Generate publication-ready figures
```

### Data Export Standards

- Use consistent CSV schema: method, seed, epoch_eff, split, acc
- Include metadata sidecar with experimental configuration
- Implement deterministic ordering for reproducible outputs
- Validate data integrity before export
- Support both incremental and batch export modes

## Method Comparison Guidelines

### Baseline Requirements

- Include naive SGD baseline for lower bound comparison
- Implement scratch training on Task 2 for upper bound reference
- Add interleaved training as oracle comparison
- Include relevant continual learning methods (EWC, DER++, GPM)
- Document method-specific hyperparameter tuning procedures

### Fair Comparison Protocols

- Normalize computational budgets using effective epochs
- Use consistent model architectures across methods
- Implement identical evaluation protocols for all methods
- Document any method-specific preprocessing or augmentation
- Include statistical significance testing for method comparisons

## Dataset and Task Design

### CIFAR-100 Einstellung Protocol

```yaml
# Standard configuration for CIFAR-100 ERI experiments
dataset:
  name: "SequentialCIFAR100Einstellung224"
  task1_classes: [0, 1, 2, 3, 4] # First 5 superclasses
  task2_classes: [5, 6, 7, 8, 9] # Second 5 superclasses
  shortcut_config:
    patch_sizes: [2, 4, 6, 8]
    locations: ["top_left", "random_corner"]
    injection_ratios: [0.5, 1.0]
    colors: ["magenta", "high_contrast"]
```

### Evaluation Splits Definition

- **T1_all**: All Task 1 test samples
- **T2_shortcut_normal**: Task 2 samples with shortcuts (patched)
- **T2_shortcut_masked**: Task 2 samples with shortcuts removed (masked)
- **T2_nonshortcut_normal**: Task 2 samples without shortcuts

### Generalization Testing

- Include robustness analysis across different shortcut configurations
- Test sensitivity to hyperparameter choices (tau thresholds, smoothing)
- Validate findings across different architectural choices (CNN vs ViT)
- Document limitations and scope of experimental conclusions

## Documentation and Reporting

### Experimental Logs

- Maintain detailed logs of all experimental runs
- Include system information and computational resources used
- Document any experimental failures or anomalies
- Provide clear audit trail for result reproduction

### Result Presentation

- Use consistent visualization formats across all experiments
- Include uncertainty quantification in all reported metrics
- Provide both aggregate statistics and individual run details
- Document any data preprocessing or filtering decisions

### Reviewer Response Preparation

- Address generalizability limitations explicitly
- Document planned extensions (ImageNet-100, text domains)
- Include robustness analysis and sensitivity studies
- Provide method-agnostic claims with appropriate scope
