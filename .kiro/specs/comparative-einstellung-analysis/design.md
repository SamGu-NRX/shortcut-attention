# Comparative Einstellung Analysis System — Design

## Overview

This design document outlines how to implement comprehensive comparative Einstellung analysis by making minimal, targeted extensions to the existing Mammoth infrastructure. The approach leverages existing systems (ERI visualization, experiment runner, model registry) and adds only the essential missing components.

## Architecture

The system extends existing Mammoth components in four key areas:

1. **Baseline Methods**: Add Scratch_T2 and Interleaved models to existing model registry
2. **Data Aggregation**: Extend experiment runner to collect and merge CSV results
3. **Visualization Enhancement**: Leverage existing ERI system for multi-method plots
4. **Orchestration**: Extend existing comparative experiment runner

```mermaid
graph TB
    subgraph "Existing Infrastructure (Reuse)"
        A[models/__init__.py<br/>Model Registry]
        B[run_einstellung_experiment.py<br/>Experiment Runner]
        C[eri_vis/<br/>Visualization System]
        D[utils/einstellung_evaluator.py<br/>Evaluation System]
    end

    subgraph "New Components (Minimal)"
        E[models/scratch_t2.py<br/>Baseline Method]
        F[models/interleaved.py<br/>Baseline Method]
        G[aggregate_comparative_results()<br/>Data Aggregation]
    end

    A --> E
    A --> F
    B --> G
    G --> C
    D --> C

    style E fill:#e1f5fe
    style F fill:#e1f5fe
    style G fill:#e1f5fe
```

## Components and Interfaces

### 1. Baseline Method Implementation

#### 1.1 Scratch_T2 Model

**File**: `models/scratch_t2.py`

**Purpose**: Implements Task 2 only training baseline for measuring optimal performance on shortcut task.

**Design Pattern**: Extends ContinualModel following existing Mammoth patterns (similar to `models/joint.py`)

```python
class ScratchT2(ContinualModel):
    NAME = 'scratch_t2'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(ScratchT2, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.task_data = []

    def begin_task(self, dataset):
        # Skip Task 1, only collect Task 2 data
        if dataset.i == 1:  # Task 2 (0-indexed)
            self.task_data = dataset.train_loader

    def end_task(self, dataset):
        # Train only when Task 2 ends
        if dataset.i == 1 and self.task_data:
            # Use existing training loop pattern from joint.py
            # Train on Task 2 data only

    def observe(self, *args, **kwargs):
        # Skip training during task progression
        return 0
```

**Integration Points**:

- Automatic registration via existing `models/__init__.py` discovery
- Uses existing backbone, loss, transform, dataset infrastructure
- Compatible with existing EinstellungEvaluator evaluation protocol

#### 1.2 Interleaved Model

**File**: `models/interleaved.py`

**Purpose**: Implements mixed Task 1 + Task 2 training baseline for measuring performance without task boundaries.

**Design Pattern**: Similar to Scratch_T2 but trains on combined data from both tasks

```python
class Interleaved(ContinualModel):
    NAME = 'interleaved'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Interleaved, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.all_data = []

    def end_task(self, dataset):
        self.all_data.append(dataset.train_loader)
        # Train on combined data after each task
        if len(self.all_data) > 0:
            # Combine all collected data and train
            # Use existing joint training pattern
```

### 2. Data Aggregation System

#### 2.1 Comparative Results Aggregation

**Function**: `aggregate_comparative_results(results_list, output_dir)`

**Purpose**: Merges individual experiment CSV files into single comparative dataset for visualization.

**Location**: Add to existing `run_einstellung_experiment.py`

```python
def aggregate_comparative_results(results_list: List[Dict], output_dir: str) -> str:
    """
    Aggregate CSV results from multiple experiments into single comparative dataset.

    Args:
        results_list: List of experiment result dictionaries
        output_dir: Directory to save aggregated results

    Returns:
        Path to aggregated CSV file
    """
    # Collect all CSV files from individual experiments
    csv_files = []
    for result in results_list:
        if result and result.get('success', False):
            csv_path = find_csv_file(result['output_dir'])
            if csv_path:
                csv_files.append(csv_path)

    # Use existing ERIDataLoader to merge datasets
    from eri_vis.data_loader import ERIDataLoader
    loader = ERIDataLoader()

    merged_dataset = None
    for csv_file in csv_files:
        dataset = loader.load_csv(csv_file)
        if merged_dataset is None:
            merged_dataset = dataset
        else:
            merged_dataset = merged_dataset.merge(dataset)

    # Export merged dataset
    output_path = os.path.join(output_dir, "comparative_eri_metrics.csv")
    merged_dataset.export_csv(output_path)

    return output_path
```

**Integration Points**:

- Uses existing `eri_vis.data_loader.ERIDataLoader` for CSV handling
- Leverages existing dataset merge functionality
- Follows existing CSV format and validation

### 3. Experiment Orchestration Enhancement

#### 3.1 Extended Comparative Experiment Runner

**Modification**: Extend existing `run_comparative_experiment()` in `run_einstellung_experiment.py`

**Changes Required**:

```python
def run_comparative_experiment(skip_training=False, force_retrain=False, auto_checkpoint=True,
                              debug=False):
    """Run comparative experiments across different strategies."""

    # EXISTING CODE: Current configs list
    configs = [
        ('sgd', 'resnet18'),
        ('derpp', 'resnet18'),
        ('ewc_on', 'resnet18'),
        ('gpm', 'resnet18'),
        ('dgr', 'resnet18'),
    ]

    # NEW: Add baseline methods
    baseline_configs = [
        ('scratch_t2', 'resnet18'),
        ('interleaved', 'resnet18'),
    ]

    # Run baselines first for dependency management
    all_configs = baseline_configs + configs

    results = []
    for strategy, backbone in all_configs:
        # EXISTING CODE: Use existing experiment runner
        result = run_einstellung_experiment(...)
        if result:
            results.append(result)

    # NEW: Aggregate results for comparative visualization
    if len(results) > 1:
        comparative_output_dir = "./comparative_results"
        aggregated_csv = aggregate_comparative_results(results, comparative_output_dir)

        # Use existing visualization system with aggregated data
        generate_eri_visualizations(comparative_output_dir, {
            'config': {'csv_path': aggregated_csv}
        })

    # EXISTING CODE: Summary comparison table
    # Enhanced to show comparative metrics when available
```

### 4. Visualization Enhancement

#### 4.1 Multi-Method ERI Visualization

**Approach**: Leverage existing `eri_vis/` system with aggregated data

**No Code Changes Required**: The existing ERI visualization system already supports multi-method datasets. When passed a CSV with multiple methods, it will:

- `ERITimelineProcessor.compute_performance_deficits()` - Automatically calculates PD_t using Scratch_T2 baseline
- `ERITimelineProcessor.compute_sfr_relative()` - Automatically calculates SFR_rel using Scratch_T2 baseline
- `ERIDynamicsPlotter.create_dynamics_figure()` - Shows all methods on same plot
- `ERIHeatmapPlotter` - Generates comparative robustness heatmaps

**Enhancement**: Ensure existing processors handle baseline detection:

```python
# In eri_vis/processing.py - existing code already handles this pattern
def compute_performance_deficits(self, curves: Dict[str, AccuracyCurve],
                               scratch_key: str = "Scratch_T2") -> Dict[str, TimeSeries]:
    """Existing function - no changes needed"""
    # Already computes PD_t = A_Scratch_T2(e) - A_Method(e)
```

## Data Models

### 4.1 CSV Data Format (Existing)

The system uses existing ERI CSV format - no changes required:

```csv
method,seed,epoch_eff,split,acc
scratch_t2,42,0.0,T2_shortcut_normal,0.10
scratch_t2,42,0.0,T2_shortcut_masked,0.05
sgd,42,0.0,T2_shortcut_normal,0.20
sgd,42,0.0,T2_shortcut_masked,0.15
derpp,42,0.0,T2_shortcut_normal,0.25
...
```

### 4.2 Experiment Result Structure (Existing)

Uses existing result dictionary format from `run_einstellung_experiment()`:

```python
{
    'strategy': 'scratch_t2',
    'backbone': 'resnet18',
    'seed': 42,
    'final_accuracy': 85.2,
    'output_dir': './einstellung_results/scratch_t2_resnet18_seed42',
    'success': True,
    'used_checkpoint': False,
    'metrics': EinstellungMetrics(...)  # If available
}
```

## Error Handling

### 5.1 Missing Baseline Detection

**Strategy**: Graceful degradation with clear warnings

```python
def validate_comparative_data(dataset):
    """Validate that required baselines are present for comparative analysis."""
    methods = dataset.get_unique_methods()

    missing_baselines = []
    if 'scratch_t2' not in methods:
        missing_baselines.append('scratch_t2')
    if 'interleaved' not in methods:
        missing_baselines.append('interleaved')

    if missing_baselines:
        warnings.warn(f"Missing baseline methods: {missing_baselines}. "
                     f"PD_t and SFR_rel calculations may be incomplete.")

    return len(missing_baselines) == 0
```

### 5.2 Experiment Failure Handling

**Strategy**: Use existing checkpoint and retry infrastructure

- Leverage existing `find_existing_checkpoints()` for resume capability
- Use existing error reporting and logging systems
- Maintain existing experiment state tracking

## Testing Strategy

### 6.1 Unit Tests

**New Test Files**:

- `tests/models/test_scratch_t2.py` - Test baseline method implementation
- `tests/models/test_interleaved.py` - Test interleaved method implementation
- `tests/test_comparative_aggregation.py` - Test CSV aggregation functionality

**Test Patterns**: Follow existing Mammoth test patterns using `conftest.py` fixtures

### 6.2 Integration Tests

**Approach**: Extend existing integration tests

- Add baseline methods to existing model compatibility tests
- Test comparative experiment runner with baseline methods
- Validate ERI visualization with multi-method datasets

### 6.3 End-to-End Tests

**Test Scenario**: Complete comparative experiment pipeline

```python
def test_comparative_experiment_pipeline():
    """Test full comparative experiment with baselines."""
    # Run comparative experiment with minimal config
    results = run_comparative_experiment(
        skip_training=True,  # Use existing checkpoints
        debug=True  # Minimal iterations
    )

    # Verify baseline methods included
    assert any(r['strategy'] == 'scratch_t2' for r in results)
    assert any(r['strategy'] == 'interleaved' for r in results)

    # Verify comparative visualizations generated
    assert os.path.exists("./comparative_results/eri_dynamics.pdf")
    assert os.path.exists("./comparative_results/eri_heatmap.pdf")
```

## Performance Considerations

### 7.1 Computational Efficiency

**Baseline Training**:

- Scratch_T2: Only trains on Task 2 data (50% of full dataset)
- Interleaved: Trains on combined data but uses existing efficient data loading

**Data Aggregation**:

- CSV merging is O(n) where n is total number of data points
- Uses existing efficient pandas-based operations in ERIDataLoader

**Visualization**:

- Existing ERI system already optimized for multi-method datasets
- No additional computational overhead

### 7.2 Memory Management

**Strategy**: Reuse existing memory management patterns

- Baseline methods follow existing ContinualModel memory patterns
- CSV aggregation uses streaming approach for large datasets
- Visualization system uses existing memory-efficient plotting

## Security Considerations

### 8.1 File System Access

**Approach**: Use existing secure file handling patterns

- CSV aggregation uses existing validated file paths
- Output directories follow existing security constraints
- No new file system access patterns introduced

### 8.2 Data Validation

**Strategy**: Leverage existing validation systems

- Use existing ERIDataLoader CSV validation
- Follow existing experiment result validation patterns
- Maintain existing error handling and logging security

## Deployment and Configuration

### 9.1 Installation

**Requirements**: No additional dependencies

- All new components use existing Mammoth dependencies
- Baseline methods use existing model infrastructure
- Visualization uses existing ERI system dependencies

### 9.2 Configuration

**Approach**: Use existing configuration patterns

- Baseline methods use existing args namespace configuration
- Experiment runner uses existing command-line arguments
- Visualization uses existing ERI configuration system

**New Configuration Options**: None required - all functionality accessible through existing `--comparative` flag

## Migration and Compatibility

### 10.1 Backward Compatibility

**Guarantee**: Full backward compatibility maintained

- Existing experiments continue to work unchanged
- New baseline methods are optional additions
- Existing visualization system enhanced, not replaced

### 10.2 Migration Path

**Strategy**: Zero-migration required

- New functionality activated only when baseline methods are used
- Existing users see no changes in behavior
- Comparative features available immediately when baselines are run

## Monitoring and Observability

### 11.1 Logging

**Approach**: Extend existing logging systems

- Use existing TerminalLogger for experiment progress
- Leverage existing ERI visualization logging
- Add minimal logging for aggregation steps

### 11.2 Metrics

**Strategy**: Use existing metrics infrastructure

- Baseline methods produce same metrics as existing methods
- Comparative metrics calculated by existing ERI processors
- No new metrics collection systems required
