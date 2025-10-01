# Einstellung Experiments Module

This directory contains the core implementation for running Einstellung Effect experiments in the continual learning framework.

## Directory Structure

```
experiments/einstellung/
├── __init__.py                # Module exports
├── config.py                  # Configuration dataclasses
├── runner.py                  # Experiment execution engine
├── args_builder.py            # Mammoth argument construction
├── batch.py                   # Batch experiment orchestration
├── storage.py                 # Checkpoint and artifact management
├── analysis.py                # Data analysis and aggregation
├── visualization.py           # Plot generation
└── reporting.py               # Report generation
```

## Usage

### Running a Single Experiment

```python
from experiments.einstellung import ExperimentConfig, EinstellungRunner

config = ExperimentConfig(
    strategy="sgd",
    backbone="resnet18",
    seed=42,
    n_epochs=10
)

runner = EinstellungRunner()
result = runner.run(config)
```

### Running Comparative Analysis

```python
from experiments.einstellung import ComparativeExperimentPlan, run_comparative_suite

plan = ComparativeExperimentPlan(
    strategies=["sgd", "er", "derpp"],
    backbones=["resnet18"],
    seeds=[42, 43, 44],
    n_epochs=10
)

results = run_comparative_suite(plan)
```

### Command-Line Interface

Use the main CLI script:

```bash
python run_einstellung_experiment.py --strategies sgd er derpp --seeds 42 43 44
```

## Key Components

### Configuration (`config.py`)
- `ExperimentConfig`: Single experiment configuration
- `ComparativeExperimentPlan`: Multi-experiment plan
- `ExecutionMode`: Enum for execution modes (FULL, EVAL_ONLY, SKIP_EXISTING)

### Runner (`runner.py`)
- `EinstellungRunner`: Main execution engine
- Handles checkpoint management
- Orchestrates Mammoth integration
- Collects artifacts and metrics

### Analysis (`analysis.py`)
- Timeline data aggregation
- Metric extraction
- Statistical analysis
- Cross-method comparison

### Visualization (`visualization.py`)
- ERI dynamics plots
- Comparative heatmaps
- Statistical visualizations
- Publication-ready figures

## Testing

Tests for this module are located in `/tests/einstellung/`:
- `test_einstellung_cache_validation.py`
- `test_einstellung_optimization.py`
- `test_einstellung_performance.py`
- `test_einstellung_performance_pipeline.py`
- `test_evaluation_subset_caching.py`
- `test_epoch_calculation.py`

Run tests with:
```bash
pytest tests/einstellung/
```

## Integration

This module integrates with:
- Mammoth's continual learning framework (`main.py`)
- Dataset management (`datasets/`)
- Model implementations (`models/`)
- ERI evaluation system (`utils/einstellung_integration.py`)
