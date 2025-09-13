# Einstellung Effect in Mammoth: Implementation Guide

## Overview

This document describes the complete implementation of Einstellung Effect testing procedures integrated into the Mammoth continual learning framework. The implementation provides a rigorous scientific framework for testing cognitive rigidity in continual learning models through artificial shortcuts.

```bash
python run_einstellung_experiment.py --model derpp --backbone resnet18 --auto_checkpoint
```

## Scientific Background

The **Einstellung Effect** is a cognitive rigidity phenomenon where prior experience creates a mental "set" that prevents finding simpler or better solutions. In continual learning, this manifests as:

- **Over-reliance on shortcuts**: Models learn artificial patterns in Task 2
- **Negative transfer**: Shortcut reliance interferes with Task 1 performance
- **Cognitive rigidity**: Inability to adapt when shortcuts are removed

## Experimental Design

### Task Structure

**Task 1 (T1)**: 8 superclasses, 40 fine-grained classes

- Normal learning without shortcuts
- Represents foundational knowledge

**Task 2 (T2)**: 4 superclasses, 20 fine-grained classes

- 2 superclasses with magenta shortcuts (shortcut classes)
- 2 superclasses without shortcuts (non-shortcut classes)
- Tests shortcut learning and generalization

### Evaluation Subsets

1. **T1_all**: All Task 1 data (tests negative transfer)
2. **T2_shortcut_normal**: Task 2 shortcut classes with shortcuts
3. **T2_shortcut_masked**: Task 2 shortcut classes with shortcuts masked
4. **T2_nonshortcut_normal**: Task 2 non-shortcut classes

### Core Metrics

**Einstellung Rigidity Index (ERI)** combines three metrics:

1. **Adaptation Delay (AD)**: Epochs to reach 80% accuracy on T1_all
2. **Performance Deficit (PD)**: `(acc_shortcut - acc_masked) / acc_shortcut`
3. **Shortcut Feature Reliance (SFR)**: `acc_shortcut / (acc_shortcut + acc_nonshortcut)`

**Composite ERI Score**: `0.4 × AD_norm + 0.4 × PD + 0.2 × SFR`

## Implementation Architecture

### Core Components

```
einstellung_integration/
├── datasets/
│   ├── seq_cifar100_einstellung.py     # Dataset with patch injection
│   └── configs/seq-cifar100/einstellung.yaml  # Configuration
├── utils/
│   ├── einstellung_metrics.py          # ERI calculation framework
│   ├── einstellung_attention.py        # Attention analysis (ViT)
│   ├── einstellung_evaluator.py        # Plugin-based evaluator
│   └── einstellung_integration.py      # Mammoth integration hooks
├── experiments/
│   └── einstellung_runner.py           # Comprehensive experiment orchestration
└── run_einstellung_experiment.py       # Simple experiment runner
```

### 1. Dataset Implementation (`seq_cifar100_einstellung.py`)

**Key Features**:

- Inherits from Mammoth's `SequentialCIFAR100` and `SeqCIFAR100224` for ViTs
- Dynamic patch injection based on configuration
- Multi-subset evaluation support
- Proper class filtering and remapping

**CIFAR-100 Superclass Mapping**:

```python
# Task 1: 8 superclasses (40 fine classes)
T1_SUPERCLASSES = [0, 1, 2, 3, 4, 5, 6, 7]

# Task 2: 4 superclasses (20 fine classes)
T2_SUPERCLASSES = [8, 9, 10, 11]

# Shortcut classes (first 2 superclasses of T2)
SHORTCUT_SUPERCLASSES = [8, 9]  # 10 fine classes
```

**Patch Injection**:

```python
class MagentaPatchInjector:
    def apply_patch(self, img, index):
        """Apply magenta shortcut patch at deterministic location."""
        # Deterministic placement based on image index
        # 4x4 magenta patch (RGB: 255, 0, 255)
```

### 2. Metrics Framework (`einstellung_metrics.py`)

**EinstellungMetricsCalculator**: Tracks timeline data and computes ERI metrics

```python
calculator = EinstellungMetricsCalculator(adaptation_threshold=0.8)

# Add data during training
calculator.add_timeline_data(
    epoch=epoch,
    task_id=task_id,
    subset_accuracies={'T1_all': 0.85, 'T2_shortcut_normal': 0.92, ...},
    subset_losses={'T1_all': 0.45, ...}
)

# Calculate final metrics
metrics = calculator.calculate_comprehensive_metrics()
print(f"ERI Score: {metrics.eri_score}")
```

### 3. Attention Analysis (`einstellung_attention.py`)

**Extended Metrics for ViT Models**:

- **Attention Spread (AS)**: Concentration on shortcut patches
- **Shortcut Attention Gain (SAG)**: Attention increase due to shortcuts
- **Cross-Task Attention Similarity (CTAS)**: Pattern similarity across tasks
- **Attention Diversity Index (ADI)**: Diversity across samples

```python
analyzer = EinstellungAttentionAnalyzer(model, device)

# Analyze attention patterns
metrics = analyzer.analyze_einstellung_attention_batch(
    inputs=batch_inputs,
    subset_name='T2_shortcut_normal',
    epoch=epoch
)
```

### 4. Evaluation Plugin (`einstellung_evaluator.py`)

**Plugin-based Integration**: Hooks into Mammoth's training pipeline

```python
class EinstellungEvaluator:
    def meta_begin_task(self, model, dataset):
        """Called at task start"""

    def after_training_epoch(self, model, dataset, epoch):
        """Called after each epoch - core evaluation logic"""

    def meta_end_task(self, model, dataset):
        """Called at task end - comprehensive analysis"""
```

### 5. Integration System (`einstellung_integration.py`)

**Seamless Integration**: Monkey-patches Mammoth without modifying core files

```python
# Enable integration
enable_einstellung_integration(args)

# Automatically hooks into:
# - model.meta_begin_task()
# - model.meta_end_task()
# - After-epoch evaluation
# - Results export
```

## Usage Guide

### Quick Start

```bash
# Basic experiment
python run_einstellung_experiment.py --model derpp --backbone resnet18

# ViT with attention analysis
python run_einstellung_experiment.py --model ewc_on --backbone vit_base_patch16_224

# Comparative analysis
python run_einstellung_experiment.py --comparative
```

### Configuration via YAML

```yaml
# datasets/configs/seq-cifar100/einstellung.yaml
dataset: seq-cifar100-einstellung
model: derpp
backbone: resnet18

# Einstellung parameters
einstellung_apply_shortcut: true
einstellung_patch_size: 4
einstellung_patch_color: [255, 0, 255]
einstellung_adaptation_threshold: 0.8
einstellung_evaluation_subsets: true
einstellung_extract_attention: true
```

### Programmatic Usage

```python
from utils.einstellung_integration import enable_einstellung_integration
from utils.einstellung_evaluator import create_einstellung_evaluator

# Create experiment args
args = create_experiment_args()

# Enable integration
enable_einstellung_integration(args)

# Run experiment
result = mammoth_main(args)

# Get results
evaluator = get_einstellung_evaluator()
metrics = evaluator.get_final_metrics()
```

## Advanced Features

### Multi-Scenario Experiments

```python
# experiments/einstellung_runner.py
runner = EinstellungExperimentRunner(config)

# Run comprehensive experiments
runner.run_comprehensive_experiment()

# Scenarios: sequential, scratch_t2, interleaved
# Strategies: sgd, ewc_on, derpp
# Seeds: [42, 123, 456, 789, 1011]
```

### Statistical Analysis

```python
from utils.einstellung_metrics import calculate_cross_experiment_eri_statistics

# Aggregate results across seeds
stats = calculate_cross_experiment_eri_statistics(metrics_list)

# Get mean, std, min, max for each metric
print(f"ERI Score: {stats['eri_score']['mean']:.4f} ± {stats['eri_score']['std']:.4f}")
```

### Attention Visualization

```python
# For ViT models
analyzer.visualize_einstellung_attention_comparison(
    normal_inputs=t1_inputs,
    shortcut_inputs=t2_shortcut_inputs,
    masked_inputs=t2_masked_inputs,
    save_path='./attention_comparison.png'
)
```

## Results Analysis

### Output Structure

```
einstellung_results/
├── derpp_resnet18_seed42/
│   ├── detailed_results.json           # Complete timeline + metrics
│   ├── einstellung_final_results.json  # Final ERI metrics
│   └── attention_analysis.json         # Attention metrics (ViT only)
└── comprehensive_results.json          # Cross-experiment statistics
```

### Key Result Files

**detailed_results.json**:

```json
{
  "configuration": {...},
  "timeline_data": [...],
  "final_metrics": {
    "adaptation_delay": 15.0,
    "performance_deficit": 0.23,
    "shortcut_feature_reliance": 0.67,
    "eri_score": 0.45
  },
  "attention_analysis": {...}
}
```

### Interpretation Guidelines

**ERI Score Interpretation**:

- **0.0-0.3**: Low rigidity (good adaptation)
- **0.3-0.6**: Moderate rigidity
- **0.6-1.0**: High rigidity (poor adaptation)

**Expected Results**:

- **SGD**: High rigidity (ERI > 0.6) due to catastrophic forgetting
- **EWC**: Moderate rigidity (ERI 0.4-0.6) due to parameter constraints
- **DER++**: Lower rigidity (ERI 0.3-0.5) due to replay mechanisms

## Scientific Validation

### Reproducing Original Results

The implementation reproduces key findings from the original Avalanche version:

1. **Shortcut Learning**: T2 shortcut classes achieve higher accuracy than non-shortcut classes
2. **Performance Deficit**: Accuracy drops when shortcuts are masked
3. **Negative Transfer**: T1 performance degrades when shortcuts are present
4. **Strategy Differences**: Replay methods show lower rigidity than regularization methods

### Experimental Controls

- **Deterministic shortcuts**: Fixed patch placement based on image index
- **Balanced evaluation**: Equal representation across classes and subsets
- **Multiple seeds**: Statistical significance testing
- **Attention analysis**: ViT-specific cognitive rigidity measures

## Integration with Mammoth

### Native Compatibility

- **Existing models**: Works with SGD, EWC, DER++, and other strategies
- **Standard logging**: Integrates with CSV logging and TensorBoard
- **Configuration system**: Uses Mammoth's YAML configuration
- **Evaluation pipeline**: Leverages existing evaluation functions

### Extensibility

- **New strategies**: Automatically compatible with any Mammoth strategy
- **Custom metrics**: Easy to add new rigidity measures
- **Different datasets**: Framework adaptable to other datasets
- **Attention models**: Extensible to new transformer architectures

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure CIFAR-100 is downloaded to `./data/CIFAR100/`
2. **Integration not enabled**: Check that dataset name contains 'einstellung'
3. **Memory issues**: Reduce batch size or disable attention extraction
4. **Missing dependencies**: Install required packages (scipy, sklearn)

### Debug Mode

```bash
python run_einstellung_experiment.py --verbose --model sgd
```

### Manual Integration

```python
# If automatic integration fails
evaluator = EinstellungEvaluator(args)
evaluator.meta_begin_task(model, dataset)
# ... manual hook calls
```

## Contributing

### Adding New Metrics

1. Extend `EinstellungMetrics` dataclass
2. Add calculation method to `EinstellungMetricsCalculator`
3. Update export functions

### Supporting New Models

1. Implement attention extraction for new architectures
2. Add model-specific configuration options
3. Update backbone choices in experiment runners

### Performance Optimization

- Batch attention extraction
- Cached evaluation subsets
- Parallel multi-seed experiments

## References

1. Original Einstellung Effect paper: [Citation needed]
2. Mammoth framework: [GitHub link]
3. CIFAR-100 superclass structure: [Krizhevsky, 2009]

---

This implementation provides a complete, scientifically rigorous framework for testing cognitive rigidity in continual learning through the Einstellung Effect, fully integrated with Mammoth's native capabilities.
