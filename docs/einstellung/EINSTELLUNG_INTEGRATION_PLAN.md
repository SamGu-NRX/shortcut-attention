# Einstellung Effect Integration Plan for Mammoth

## Executive Summary

This document outlines the comprehensive integration of Einstellung Effect testing methodology into the Mammoth continual learning framework. The goal is to quantify "cognitive rigidity" - the tendency of models to stick to previously learned patterns even when more optimal solutions are available.

## Table of Contents

1. [Scientific Foundation](#scientific-foundation)
2. [Technical Architecture](#technical-architecture)
3. [Implementation Strategy](#implementation-strategy)
4. [Critical Implementation Decisions](#critical-implementation-decisions)
5. [Detailed Task Breakdown](#detailed-task-breakdown)
6. [Configuration Design](#configuration-design)
7. [Metrics & Analysis Framework](#metrics--analysis-framework)
8. [Quality Assurance](#quality-assurance)
9. [Timeline & Dependencies](#timeline--dependencies)

---

## Scientific Foundation

### Core Hypothesis

Models trained on semantically demanding Task 1 (T1) will exhibit rigidity when encountering Task 2 (T2) that offers easier shortcuts, leading to:

- Slower adaptation to optimal T2 features
- Over-reliance on spurious shortcuts
- Suboptimal final performance despite shortcut availability

### Experimental Design Principles

#### Task Structure

- **T1 (Semantic Learning)**: 8 CIFAR-100 superclasses (40 fine classes) with heavy augmentation
- **T2 (Shortcut Available)**: 4 CIFAR-100 superclasses (20 fine classes) with magenta patch shortcuts
- **Augmentation Philosophy**: Strong augmentations force semantic learning by making background/texture shortcuts unreliable

#### Shortcut Implementation

- **Type**: Spatial magenta patches (4x4 pixels by default)
- **Injection Rate**: 50% of images in designated shortcut classes
- **Placement**: Random location per image (deterministic per sample index)
- **Rationale**: Creates a reliable but semantically meaningless feature

#### Evaluation Philosophy

Three evaluation modes for T2 shortcut classes:

1. **Normal**: Shortcut patches visible (tests combined semantic + shortcut learning)
2. **Masked**: Shortcut patches blocked (tests pure semantic learning)
3. **Difference**: Normal - Masked = shortcut reliance measure

---

## Technical Architecture

### Integration Approach: Pure Mammoth

- **NO Avalanche dependencies**: Complete rewrite using Mammoth primitives
- **Native pipeline usage**: Leverage Mammoth's training, logging, and evaluation systems
- **Plugin-based extensions**: Add functionality through Mammoth's plugin architecture
- **Configuration-driven**: YAML-based configuration following Mammoth conventions

### Architecture Components

#### 1. Dataset Layer

```
SeqCIFAR100Einstellung (inherits SequentialCIFAR100)
├── EinstellungDatasetConfig (YAML configuration)
├── MagentaPatchInjector (on-the-fly patch injection)
├── MaskedEvaluationSets (evaluation-time patch masking)
└── AugmentationPipeline (semantic learning augmentations)
```

#### 2. Evaluation Layer

```
EinstellungEvaluator (BasePlugin)
├── MultiSubsetAccuracy (T1, T2_normal, T2_masked, T2_nonshortcut)
├── EpochwiseTracking (timeline for AD calculation)
├── LossTracking (complementary to accuracy)
└── MetricAggregation (per-strategy summaries)
```

#### 3. Analysis Layer

```
EinstellungAnalyzer (post-training)
├── ERICalculator (AD, PD, SFR metrics)
├── AttentionAnalyzer (ViT-specific metrics)
├── StatisticalAnalysis (multi-seed aggregation)
└── VisualizationGenerator (plots and reports)
```

#### 4. Experiment Orchestration

```
EinstellungExperiment (experiment runner)
├── ScenarioManager (Scratch-T2, Interleaved, Sequential variants)
├── StrategyFactory (DER++, EWC, SGD configurations)
├── SeedManager (deterministic multi-seed execution)
└── ResultsCollector (checkpoint and metric collection)
```

---

## Implementation Strategy

### Phase 1: Core Integration (Foundation)

1. Dataset implementation with patch injection
2. Basic evaluation plugin with multi-subset tracking
3. Configuration system setup
4. Integration with existing Mammoth pipeline

### Phase 2: Metrics & Analysis (Scientific Core)

1. ERI metric calculation framework
2. Attention analysis integration for ViT
3. Statistical analysis and visualization
4. Multi-seed experiment orchestration

### Phase 3: Advanced Features (Enhancement)

1. Configurable shortcut types
2. Advanced attention metrics
3. Comprehensive reporting system
4. Performance optimization

### Phase 4: Validation & Documentation (Quality)

1. Reproduce original Avalanche results
2. Cross-validation with known baselines
3. Documentation and examples
4. Integration tests

---

## Critical Implementation Decisions

After comprehensive analysis of Mammoth's codebase, here are the critical decisions for the three key aspects:

### 4. Plugin Architecture & Evaluation Integration

**Analysis of Mammoth's Evaluation System:**
Based on my analysis of `utils/evaluate.py`, `models/utils/continual_model.py`, and `utils/loggers.py`, Mammoth uses:

- Hook-based plugin system in `ContinualModel`
- Standardized evaluation function in `utils/evaluate.py`
- CSV-based logging through `Logger` class
- Test loaders management in `ContinualDataset`

**DECISION: Custom Evaluation Plugin**

```python
# Implementation approach:
class EinstellungEvaluator:
    """
    Plugin that hooks into Mammoth's training pipeline to collect
    Einstellung-specific metrics during training.
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.epoch_metrics = []  # Store timeline for AD calculation

    def after_training_epoch(self, model, dataset):
        """Hook called after each training epoch"""
        if dataset.NAME == self.dataset_name:
            # Generate evaluation subsets on-the-fly
            subsets = self._create_evaluation_subsets(dataset)

            # Evaluate each subset
            metrics = {}
            for subset_name, subset_loader in subsets.items():
                acc = self._evaluate_subset(model, subset_loader)
                metrics[f'{subset_name}_acc'] = acc

                # Log to Mammoth's standard system
                if hasattr(model, 'writer') and model.writer is not None:
                    model.writer.add_scalar(f'einstellung/{subset_name}', acc, model.epoch_iteration)

            # Store for timeline analysis
            self.epoch_metrics.append({
                'epoch': len(self.epoch_metrics),
                'task': dataset.current_task,
                'metrics': metrics
            })

    def _create_evaluation_subsets(self, dataset):
        """Create T1_all, T2_shortcut_normal, T2_shortcut_masked, etc."""
        # Leverage existing test_loaders and apply patch masking
        pass

    def _evaluate_subset(self, model, subset_loader):
        """Evaluate model on specific subset using Mammoth's evaluation logic"""
        # Use existing evaluate() function with custom subset
        pass
```

**Integration Points:**

1. **Hook Integration**: Use `meta_observe` and `meta_end_task` hooks
2. **Logging Integration**: Extend `Logger` class with custom metrics
3. **Data Subset Creation**: Dynamic subset generation from existing test loaders
4. **CSV Export**: Leverage existing CSV logging for timeline data

### 5. Dataset Registration & Configuration System

**Analysis of Mammoth's Dataset System:**
From `datasets/__init__.py` and `datasets/utils/continual_dataset.py`:

- Automatic registration through file discovery in `datasets/` folder
- YAML configuration system in `datasets/configs/`
- Base class `ContinualDataset` with required methods
- Flexible `get_data_loaders()` implementation

**DECISION: Inherit + Extend Approach**

```python
# datasets/seq_cifar100_einstellung.py
class SequentialCIFAR100Einstellung(SequentialCIFAR100):
    """
    Einstellung Effect version of CIFAR-100 with:
    - Custom superclass ordering (8+4 split)
    - Patch injection for T2 shortcut classes
    - Enhanced augmentation pipeline for T1
    """

    NAME = 'seq-cifar100-einstellung'
    N_TASKS = 2
    N_CLASSES_PER_TASK = [40, 20]  # Variable classes per task

    def __init__(self, args):
        super().__init__(args)

        # Load Einstellung-specific configuration
        self.config = self._load_einstellung_config(args)

        # Setup superclass mappings
        self.t1_superclasses = [0, 1, 2, 3, 4, 5, 6, 7]
        self.t2_superclasses = [8, 9, 10, 11]
        self.shortcut_superclass = 8  # First T2 superclass gets shortcuts

        # Initialize patch injector
        self.patch_injector = MagentaPatchInjector(
            patch_size=self.config.get('patch_size', 4),
            color=self.config.get('patch_color', [255, 0, 255]),
            injection_rate=self.config.get('injection_rate', 0.5)
        )

    def get_data_loaders(self):
        """Override to apply patch injection and custom transforms"""
        train_loader, test_loader = super().get_data_loaders()

        # Apply patch injection wrapper for T2
        if self.current_task == 1:  # T2
            train_loader = self._wrap_with_patch_injection(train_loader, train=True)
            test_loader = self._wrap_with_patch_injection(test_loader, train=False)

        return train_loader, test_loader

    def get_evaluation_subsets(self):
        """Create evaluation subsets for Einstellung metrics"""
        return {
            't1_all': self._create_t1_subset(),
            't2_shortcut_normal': self._create_t2_shortcut_subset(masked=False),
            't2_shortcut_masked': self._create_t2_shortcut_subset(masked=True),
            't2_nonshortcut_normal': self._create_t2_nonshortcut_subset()
        }
```

**Configuration Strategy:**

```yaml
# datasets/configs/seq-cifar100/einstellung.yaml
N_TASKS: 2
N_CLASSES_PER_TASK: [40, 20]
N_CLASSES: 60

# Superclass configuration
SUPERCLASS_CONFIG:
  t1_superclasses: [0, 1, 2, 3, 4, 5, 6, 7]
  t2_superclasses: [8, 9, 10, 11]
  shortcut_superclass: 8

# Patch injection parameters
PATCH_CONFIG:
  enabled: true
  size: 4
  color: [255, 0, 255]
  injection_rate: 0.5
  seed_offset: 42

# Enhanced augmentation for T1
T1_AUGMENTATION:
  enabled: true
  rotation_degrees: 15
  translation: [0.1, 0.1]
  scale_range: [0.8, 1.2]
  color_jitter: 0.2
```

### 6. Attention Analysis Integration

**Analysis of Existing Attention System:**
From `backbone/vit.py` and `utils/attention_visualization.py`:

- ViT implementation supports `return_attention_scores=True`
- Existing `AttentionAnalyzer` class with basic functionality
- Attention extraction from all transformer blocks
- Integration with model forward pass

**DECISION: Extend Existing Attention System**

```python
# utils/einstellung_attention.py
class EinstellungAttentionAnalyzer(AttentionAnalyzer):
    """
    Extended attention analyzer for Einstellung Effect experiments.
    Builds on existing Mammoth attention utilities.
    """

    def __init__(self, model, device="cuda"):
        super().__init__(model, device)
        self.patch_coords = None  # Will be set during analysis

    def analyze_einstellung_patterns(self, t1_loader, t2_normal_loader, t2_masked_loader):
        """
        Comprehensive attention analysis for Einstellung Effect.

        Returns:
            Dict with attention metrics: AS, SAG, CTAS, ADI
        """
        metrics = {}

        # Extract attention maps for each subset
        t1_attention = self._extract_subset_attention(t1_loader)
        t2_normal_attention = self._extract_subset_attention(t2_normal_loader)
        t2_masked_attention = self._extract_subset_attention(t2_masked_loader)

        # Calculate Einstellung-specific metrics
        metrics['attention_spread_t1'] = self._calculate_attention_spread(t1_attention)
        metrics['attention_spread_t2'] = self._calculate_attention_spread(t2_normal_attention)

        metrics['shortcut_attention_gain'] = self._calculate_shortcut_attention_gain(
            t2_normal_attention, t2_masked_attention
        )

        metrics['cross_task_similarity'] = self._calculate_cross_task_similarity(
            t1_attention, t2_normal_attention
        )

        metrics['attention_diversity_t1'] = self._calculate_attention_diversity(t1_attention)
        metrics['attention_diversity_t2'] = self._calculate_attention_diversity(t2_normal_attention)

        return metrics

    def _calculate_shortcut_attention_gain(self, normal_attention, masked_attention):
        """
        Calculate SAG metric: difference in attention to patch regions
        between normal and masked images.
        """
        # Detect patch regions in normal images
        patch_regions = self._detect_patch_regions(normal_attention)

        # Calculate mean attention to patch regions
        normal_patch_attention = self._extract_patch_attention(normal_attention, patch_regions)
        masked_patch_attention = self._extract_patch_attention(masked_attention, patch_regions)

        return (normal_patch_attention - masked_patch_attention).mean().item()

    def _detect_patch_regions(self, attention_maps):
        """
        Automatically detect magenta patch regions from attention patterns.
        Uses attention peak detection to identify artificial shortcuts.
        """
        # Implementation: Find consistent high-attention regions across samples
        # that correspond to 4x4 patch locations
        pass

    def visualize_einstellung_attention(self, save_dir):
        """
        Create Einstellung-specific attention visualizations:
        - Normal vs masked attention difference maps
        - Cross-task attention similarity heatmaps
        - Attention evolution timeline plots
        """
        # Extend existing visualization with Einstellung-specific plots
        pass
```

**Integration with Existing System:**

1. **Leverage Existing Infrastructure**: Build on `AttentionAnalyzer` class
2. **ViT Forward Pass**: Use existing `return_attention_scores=True` functionality
3. **Automatic Detection**: Implement patch region detection from attention patterns
4. **Visualization Extension**: Enhance existing plotting utilities
5. **Metric Integration**: Add attention metrics to evaluation pipeline

**Key Advantages of This Approach:**

- **Minimal Code Duplication**: Reuses existing attention extraction
- **Automatic Compatibility**: Works with any ViT backbone in Mammoth
- **Extensible Design**: Easy to add new attention metrics
- **Integration Ready**: Hooks into existing evaluation and logging systems

---

## Detailed Task Breakdown

### Dataset Implementation

#### Task D1: Core Dataset Class

**Priority**: Critical
**Dependencies**: None
**Estimated Time**: 4 hours

**Subtasks**:

- [ ] Create `datasets/seq_cifar100_einstellung.py`
  - [ ] Inherit from `SequentialCIFAR100`
  - [ ] Override `NAME = 'seq-cifar100-einstellung'`
  - [ ] Implement custom class ordering (8+4 superclasses)
  - [ ] Integrate with Mammoth's dataset registry
- [ ] Create `datasets/configs/seq-cifar100/einstellung.yaml`
  - [ ] Set `N_TASKS: 2`
  - [ ] Define explicit `CLASS_ORDER` for 8+4 superclasses
  - [ ] Configure patch injection parameters
  - [ ] Set augmentation pipeline
- [ ] Test dataset loading and basic functionality

#### Task D2: Patch Injection System

**Priority**: Critical
**Dependencies**: D1
**Estimated Time**: 6 hours

**Subtasks**:

- [ ] Create `utils/patch_injection.py`
  - [ ] Implement `MagentaPatchInjector` class
  - [ ] Support configurable patch size, color, injection rate
  - [ ] Deterministic placement based on sample index
  - [ ] Handle both training and evaluation modes
- [ ] Create `PatchInjectionWrapper` dataset wrapper
  - [ ] Inherit from `torch.utils.data.Dataset`
  - [ ] Apply patches on-the-fly during `__getitem__`
  - [ ] Support masking mode for evaluation
- [ ] Integration with dataset class
  - [ ] Modify dataset to use wrapper when appropriate
  - [ ] Handle task-specific injection (T2 only)
- [ ] Unit tests for patch injection logic

#### Task D3: Evaluation Dataset Variants

**Priority**: High
**Dependencies**: D2
**Estimated Time**: 4 hours

**Subtasks**:

- [ ] Implement masked evaluation sets
  - [ ] T2 shortcut classes (normal)
  - [ ] T2 shortcut classes (masked)
  - [ ] T2 non-shortcut classes
  - [ ] T1 classes (all)
- [ ] Create evaluation data loaders on-demand
- [ ] Ensure consistent class mapping across variants
- [ ] Test evaluation set generation

### Evaluation System

#### Task E1: Core Evaluation Plugin

**Priority**: Critical
**Dependencies**: D1, D2
**Estimated Time**: 8 hours

**Subtasks**:

- [ ] Create `utils/einstellung_evaluator.py`
  - [ ] Inherit from Mammoth's `BasePlugin`
  - [ ] Implement `after_training_epoch` hook
  - [ ] Implement `after_training` hook
  - [ ] Handle multiple evaluation subsets
- [ ] Accuracy calculation per subset
  - [ ] Use Mammoth's existing accuracy metrics
  - [ ] Support both per-class and overall accuracy
- [ ] Loss calculation per subset
  - [ ] Consistent with accuracy calculation
  - [ ] Handle class mapping correctly
- [ ] Integration with Mammoth's logging system
  - [ ] Register custom metric names
  - [ ] Ensure CSV compatibility

#### Task E2: Timeline Tracking

**Priority**: High
**Dependencies**: E1
**Estimated Time**: 3 hours

**Subtasks**:

- [ ] Implement epoch-wise metric storage
  - [ ] Track T2 shortcut accuracy over time
  - [ ] Store task-specific information
  - [ ] Handle sequential vs interleaved scenarios
- [ ] Memory-efficient storage
- [ ] Export to Mammoth's standard logging

#### Task E3: Strategy-Specific Handling

**Priority**: Medium
**Dependencies**: E1
**Estimated Time**: 4 hours

**Subtasks**:

- [ ] Handle different training scenarios
  - [ ] Sequential training (track T1 and T2 phases)
  - [ ] Interleaved training (single phase)
  - [ ] Scratch-T2 (T2 only)
- [ ] Proper class mapping for each scenario
- [ ] Validation of metric consistency

### Metrics & Analysis

#### Task M1: ERI Metric Calculator

**Priority**: Critical
**Dependencies**: E1, E2
**Estimated Time**: 6 hours

**Subtasks**:

- [ ] Create `utils/einstellung_metrics.py`
  - [ ] Port ERI calculation from Avalanche version
  - [ ] Implement Adaptation Delay calculation
  - [ ] Implement Performance Deficit calculation
  - [ ] Implement Shortcut Feature Reliance calculation
- [ ] Statistical aggregation across seeds
- [ ] NaN handling and edge cases
- [ ] Export to CSV format

#### Task M2: Attention Analysis Integration

**Priority**: High
**Dependencies**: M1, existing attention utils
**Estimated Time**: 8 hours

**Analysis of Approach**:
After analyzing the existing `utils/attention_visualization.py`, the best approach is to extend it with Einstellung-specific metrics:

**Attention Metrics to Implement**:

1. **Attention Spread (AS)**: Entropy of CLS token attention distribution

   - Higher entropy = more distributed attention
   - Lower entropy = more focused attention

2. **Shortcut Attention Gain (SAG)**: Difference in attention to patch region between normal and masked images

   - Measures direct shortcut reliance
   - Computed as mean attention to patch coordinates

3. **Cross-Task Attention Similarity (CTAS)**: Cosine similarity of attention patterns between T1 and T2 classes

   - Measures attention pattern transfer
   - Higher similarity = more rigid attention patterns

4. **Attention Diversity Index (ADI)**: Variation in attention patterns within each task
   - Measures flexibility of attention mechanism
   - Calculated as mean pairwise distance between attention maps

**Subtasks**:

- [ ] Extend `AttentionAnalyzer` class
  - [ ] Add Einstellung-specific methods
  - [ ] Implement patch region detection
  - [ ] Calculate attention statistics
- [ ] Create `EinstellungAttentionAnalyzer`
  - [ ] Integrate with existing attention utilities
  - [ ] Add new metric calculations
  - [ ] Support multi-seed analysis
- [ ] Visualization extensions
  - [ ] Patch region highlighting
  - [ ] Attention difference maps (normal vs masked)
  - [ ] Cross-task attention comparison plots
- [ ] Integration with main analysis pipeline

#### Task M3: Statistical Analysis Framework

**Priority**: Medium
**Dependencies**: M1, M2
**Estimated Time**: 5 hours

**Subtasks**:

- [ ] Multi-seed aggregation
  - [ ] Mean, std, min, max across seeds
  - [ ] Statistical significance testing
  - [ ] Confidence intervals
- [ ] Cross-strategy comparison
  - [ ] Pairwise statistical tests
  - [ ] Effect size calculations
- [ ] Correlation analysis
  - [ ] ERI metrics vs attention metrics
  - [ ] Strategy performance relationships

### Experiment Orchestration

#### Task O1: Scenario Management

**Priority**: High
**Dependencies**: D3, E1
**Estimated Time**: 6 hours

**Subtasks**:

- [ ] Create `experiments/einstellung_runner.py`
  - [ ] Implement scenario definitions
  - [ ] Handle different strategy configurations
  - [ ] Manage multi-seed execution
- [ ] Scenario implementations:
  - [ ] Scratch-T2: `--start_from 1`
  - [ ] Interleaved: `--joint 1` with custom logic
  - [ ] Naive Sequential: basic SGD
  - [ ] CL Strategies: DER++, EWC with tuned parameters
- [ ] Resource management and job scheduling
- [ ] Progress tracking and error handling

#### Task O2: Configuration Management

**Priority**: Medium
**Dependencies**: O1
**Estimated Time**: 4 hours

**Subtasks**:

- [ ] Create strategy-specific configs
  - [ ] `models/config/derpp_einstellung.yaml`
  - [ ] `models/config/ewc_on_einstellung.yaml`
  - [ ] Parameter tuning for Einstellung setup
- [ ] Master experiment configuration
  - [ ] `experiments/configs/einstellung_sweep.yaml`
  - [ ] Seed management
  - [ ] Resource allocation
- [ ] Validation and error checking

#### Task O3: Results Collection & Management

**Priority**: Medium
**Dependencies**: O1, M1
**Estimated Time**: 3 hours

**Subtasks**:

- [ ] Checkpoint management
  - [ ] Organized storage structure
  - [ ] Metadata tracking
  - [ ] Recovery mechanisms
- [ ] Log aggregation
  - [ ] Combine logs from multiple runs
  - [ ] Generate summary reports
  - [ ] Export standardized formats
- [ ] Integration with analysis pipeline

### Visualization & Reporting

#### Task V1: Core Visualization System

**Priority**: Medium
**Dependencies**: M1, M2
**Estimated Time**: 6 hours

**Analysis of Approach**:
The best approach is to extend the existing plotting utilities while creating Einstellung-specific visualizations:

**Runtime Constraints Analysis**:

- **GPU Memory**: ViT-B/16 requires ~6GB for batch_size=32, manageable on modern GPUs
- **Time**: Full experiment (4 strategies × 3 seeds × 2 backbones) ≈ 48 hours on single GPU
- **Storage**: ~500MB per checkpoint, ~24GB total for full experiment

**Report Format Analysis**:
Best approach is multi-format output:

- **CSV**: Machine-readable for further analysis
- **PNG**: High-quality plots for papers
- **Jupyter**: Interactive analysis and debugging
- **JSON**: Structured data for web interfaces

**Subtasks**:

- [ ] Extend existing plotting utilities
  - [ ] Add ERI-specific plot types
  - [ ] Learning curve visualization
  - [ ] Multi-strategy comparison plots
  - [ ] Statistical significance visualization
- [ ] Attention visualization enhancements
  - [ ] Patch region highlighting
  - [ ] Difference maps (normal vs masked)
  - [ ] Cross-task attention evolution
- [ ] Report generation framework
  - [ ] Automated report creation
  - [ ] Multi-format output support
  - [ ] Template-based generation

#### Task V2: Interactive Analysis Tools

**Priority**: Low
**Dependencies**: V1
**Estimated Time**: 4 hours

**Subtasks**:

- [ ] Jupyter notebook templates
  - [ ] Interactive exploration of results
  - [ ] Parameter sensitivity analysis
  - [ ] Custom visualization creation
- [ ] Command-line analysis tools
  - [ ] Quick summary generation
  - [ ] Filtering and querying results
  - [ ] Export utilities

### Integration & Testing

#### Task I1: Mammoth Integration

**Priority**: Critical
**Dependencies**: All core tasks
**Estimated Time**: 8 hours

**Subtasks**:

- [ ] Registry integration
  - [ ] Register new dataset with Mammoth
  - [ ] Register custom evaluation plugins
  - [ ] Register analysis utilities
- [ ] CLI integration
  - [ ] Standard Mammoth argument parsing
  - [ ] Custom arguments for Einstellung features
  - [ ] Help documentation
- [ ] Logging integration
  - [ ] Use Mammoth's standard logging
  - [ ] Custom metric registration
  - [ ] CSV export compatibility
- [ ] Configuration validation
  - [ ] YAML schema validation
  - [ ] Parameter range checking
  - [ ] Dependency verification

#### Task I2: Quality Assurance

**Priority**: High
**Dependencies**: I1
**Estimated Time**: 12 hours

**Subtasks**:

- [ ] Unit tests
  - [ ] Dataset loading and patch injection
  - [ ] Metric calculations
  - [ ] Configuration validation
  - [ ] Edge case handling
- [ ] Integration tests
  - [ ] End-to-end experiment runs
  - [ ] Multi-strategy comparison
  - [ ] Reproducibility verification
- [ ] Performance tests
  - [ ] Memory usage profiling
  - [ ] Training time benchmarks
  - [ ] GPU utilization analysis
- [ ] Validation against original results
  - [ ] Reproduce key findings from Avalanche version
  - [ ] Statistical validation of differences
  - [ ] Document any discrepancies

#### Task I3: Documentation & Examples

**Priority**: Medium
**Dependencies**: I2
**Estimated Time**: 6 hours

**Subtasks**:

- [ ] API documentation
  - [ ] Docstring completion
  - [ ] Type hints
  - [ ] Usage examples
- [ ] User guide
  - [ ] Quick start tutorial
  - [ ] Configuration guide
  - [ ] Troubleshooting section
- [ ] Developer documentation
  - [ ] Architecture overview
  - [ ] Extension guidelines
  - [ ] Contributing guide
- [ ] Example experiments
  - [ ] Minimal working example
  - [ ] Full reproduction script
  - [ ] Custom analysis examples

---

## Configuration Design

### Master Configuration Structure

```yaml
# datasets/configs/seq-cifar100/einstellung.yaml
NAME: "seq-cifar100-einstellung"
SETTING: "class-il"
N_TASKS: 2
N_CLASSES_PER_TASK: [40, 20] # T1: 8 superclasses, T2: 4 superclasses
N_CLASSES: 60
SIZE: [32, 32]

# Superclass definitions
T1_SUPERCLASSES: [0, 1, 2, 3, 4, 5, 6, 7] # First 8 superclasses
T2_SUPERCLASSES: [8, 9, 10, 11] # Next 4 superclasses
SHORTCUT_SUPERCLASS: 8 # First T2 superclass gets shortcuts

# Patch injection configuration
PATCH_CONFIG:
  enabled: true
  size: 4 # 4x4 pixel patch
  color: [255, 0, 255] # Magenta RGB
  injection_rate: 0.5 # 50% of shortcut class images
  position: "random" # Random placement
  seed_offset: 42 # For deterministic placement

# Augmentation configuration
TRANSFORM:
  - RandomCrop:
      size: 32
      padding: 4
  - RandomHorizontalFlip:
      p: 0.5
  - RandomRotation:
      degrees: 10
  - RandomAffine:
      degrees: 0
      translate: [0.05, 0.05]
      scale: [0.95, 1.05]
  - ColorJitter:
      brightness: 0.15
      contrast: 0.15
      saturation: 0.15
  - ToTensor
  - Normalize:
      mean: [0.5071, 0.4867, 0.4408]
      std: [0.2675, 0.2565, 0.2761]

# Evaluation configuration
EVALUATION:
  subsets:
    - "t1_all" # All T1 classes
    - "t2_all_normal" # All T2 classes (shortcuts visible)
    - "t2_shortcut_normal" # T2 shortcut classes (shortcuts visible)
    - "t2_shortcut_masked" # T2 shortcut classes (shortcuts masked)
    - "t2_nonshortcut_normal" # T2 non-shortcut classes

  # ERI metric configuration
  eri_config:
    adaptation_threshold: 0.8 # Accuracy threshold for AD calculation
    random_seeds: [42, 123, 456]

# Default training parameters
batch_size: 64
n_epochs: 100
lr: 0.01
backbone: "resnet18" # Default, can be overridden to 'vit'
```

### Strategy-Specific Configurations

```yaml
# models/config/derpp_einstellung.yaml
alpha: 0.1        # MSE loss weight for buffer logits
beta: 0.5         # CE loss weight for buffer samples
buffer_size: 500  # Replay buffer size
lr: 0.01
n_epochs: 100
batch_size: 32    # Smaller batch for memory efficiency

# models/config/ewc_on_einstellung.yaml
e_lambda: 5.0     # EWC regularization strength (tuned for Einstellung)
gamma: 1.0        # EWC online decay factor
lr: 0.01
n_epochs: 100
batch_size: 64
```

### Experiment Configuration

```yaml
# experiments/configs/einstellung_sweep.yaml
experiment_name: "einstellung_effect_analysis"
seeds: [42, 123, 456]
backbones: ["resnet18", "vit"]
strategies: ["sgd", "derpp", "ewc_on"]
scenarios: ["scratch_t2", "interleaved", "sequential"]

# Resource configuration
gpu_memory_limit: 8000 # MB
max_parallel_jobs: 2
checkpoint_frequency: "task" # Save after each task

# Analysis configuration
analysis:
  attention_analysis: true # Enable for ViT only
  statistical_tests: true
  generate_plots: true
  export_formats: ["csv", "json", "png"]
```

---

## Metrics & Analysis Framework

### Core ERI Metrics

#### 1. Adaptation Delay (AD)

```python
def calculate_adaptation_delay(timeline, threshold=0.8, baseline_timeline=None):
    """
    Calculate epochs needed to reach threshold accuracy on T2 shortcut classes.

    Args:
        timeline: List of {epoch, T2_ShortcutClasses_Normal_Acc} dicts
        threshold: Accuracy threshold for adaptation
        baseline_timeline: Scratch-T2 timeline for comparison

    Returns:
        int: Extra epochs needed compared to baseline (0 if baseline not provided)
    """
```

#### 2. Performance Deficit (PD)

```python
def calculate_performance_deficit(final_acc, baseline_final_acc):
    """
    Calculate final accuracy gap on T2 shortcut classes vs scratch baseline.

    Args:
        final_acc: Final accuracy of continual learner
        baseline_final_acc: Final accuracy of scratch-T2 baseline

    Returns:
        float: Accuracy deficit (positive = worse than baseline)
    """
```

#### 3. Shortcut Feature Reliance (SFR)

```python
def calculate_shortcut_reliance(normal_acc, masked_acc, baseline_reliance=0):
    """
    Calculate reliance on shortcut features.

    Args:
        normal_acc: Accuracy with shortcuts visible
        masked_acc: Accuracy with shortcuts masked
        baseline_reliance: Baseline reliance (from scratch-T2)

    Returns:
        dict: {
            'absolute_reliance': normal_acc - masked_acc,
            'relative_reliance': (normal_acc - masked_acc) - baseline_reliance
        }
    """
```

### Attention-Specific Metrics (ViT Only)

#### 1. Attention Spread (AS)

```python
def calculate_attention_spread(attention_maps):
    """
    Calculate entropy of CLS token attention distribution.
    Higher entropy = more distributed attention.
    """
    # Extract CLS to patch attention (row 0, cols 1:)
    cls_attention = attention_maps[:, 0, 1:]  # [batch, num_patches]

    # Calculate entropy per sample
    entropies = []
    for sample_attention in cls_attention:
        # Normalize to probability distribution
        probs = F.softmax(sample_attention, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        entropies.append(entropy.item())

    return np.mean(entropies)
```

#### 2. Shortcut Attention Gain (SAG)

```python
def calculate_shortcut_attention_gain(normal_attention, masked_attention, patch_coords):
    """
    Calculate difference in attention to patch region between normal and masked images.
    """
    # Extract attention to patch region
    normal_patch_attention = extract_patch_attention(normal_attention, patch_coords)
    masked_patch_attention = extract_patch_attention(masked_attention, patch_coords)

    # Calculate gain
    attention_gain = normal_patch_attention.mean() - masked_patch_attention.mean()
    return attention_gain.item()
```

#### 3. Cross-Task Attention Similarity (CTAS)

```python
def calculate_cross_task_attention_similarity(t1_attention_maps, t2_attention_maps):
    """
    Calculate cosine similarity between T1 and T2 attention patterns.
    Higher similarity = more rigid attention transfer.
    """
    # Flatten attention maps
    t1_flat = t1_attention_maps.view(t1_attention_maps.size(0), -1)
    t2_flat = t2_attention_maps.view(t2_attention_maps.size(0), -1)

    # Calculate pairwise cosine similarities
    similarities = F.cosine_similarity(t1_flat.unsqueeze(1), t2_flat.unsqueeze(0), dim=2)

    return similarities.mean().item()
```

#### 4. Attention Diversity Index (ADI)

```python
def calculate_attention_diversity(attention_maps):
    """
    Calculate variation in attention patterns within a task.
    Higher diversity = more flexible attention mechanism.
    """
    # Flatten attention maps
    flattened = attention_maps.view(attention_maps.size(0), -1)

    # Calculate pairwise distances
    distances = torch.cdist(flattened, flattened, p=2)

    # Return mean pairwise distance (excluding diagonal)
    mask = ~torch.eye(distances.size(0), dtype=torch.bool)
    return distances[mask].mean().item()
```

### Statistical Analysis Framework

#### Multi-Seed Aggregation

```python
def aggregate_across_seeds(metric_values_per_seed):
    """
    Aggregate metrics across multiple random seeds.

    Returns:
        dict: {
            'mean': float,
            'std': float,
            'min': float,
            'max': float,
            'n_seeds': int,
            'confidence_interval': (float, float)
        }
    """
```

#### Statistical Significance Testing

```python
def compare_strategies(strategy_a_values, strategy_b_values, test='welch_t'):
    """
    Compare two strategies using statistical tests.

    Returns:
        dict: {
            'test_statistic': float,
            'p_value': float,
            'effect_size': float,
            'significant': bool
        }
    """
```

---

## Quality Assurance

### Testing Strategy

#### Unit Tests

- **Dataset Tests**: Patch injection correctness, class mapping, augmentation pipeline
- **Metric Tests**: ERI calculations, edge cases, statistical functions
- **Plugin Tests**: Evaluation hooks, logging integration, configuration validation

#### Integration Tests

- **End-to-End**: Complete experiment runs with minimal configurations
- **Multi-Strategy**: Cross-strategy metric consistency
- **Reproducibility**: Deterministic results across runs with same seeds

#### Performance Tests

- **Memory Profiling**: Peak GPU/CPU memory usage during training
- **Time Benchmarking**: Training duration per strategy and backbone
- **Scalability**: Performance with different batch sizes and model sizes

### Validation Protocol

#### Reproduction Validation

1. **Statistical Equivalence**: Results should match Avalanche version within statistical significance
2. **Metric Consistency**: ERI calculations should produce identical values for equivalent inputs
3. **Attention Analysis**: ViT attention patterns should be consistent and interpretable

#### Cross-Validation

1. **Known Baselines**: Scratch-T2 should consistently outperform sequential methods on T2
2. **Strategy Ordering**: Expected performance ranking (varies by metric but should be consistent)
3. **Attention Patterns**: ViT should show interpretable attention differences between normal/masked

---

## Timeline & Dependencies

### Phase 1: Foundation (Weeks 1-2)

**Total Estimated Time**: 40 hours

- Dataset implementation (14 hours)
- Core evaluation system (15 hours)
- Basic integration (8 hours)
- Initial testing (3 hours)

### Phase 2: Scientific Core (Weeks 3-4)

**Total Estimated Time**: 35 hours

- ERI metrics (6 hours)
- Attention analysis (8 hours)
- Statistical framework (5 hours)
- Experiment orchestration (10 hours)
- Configuration system (4 hours)
- Validation (2 hours)

### Phase 3: Enhancement (Week 5)

**Total Estimated Time**: 20 hours

- Visualization system (10 hours)
- Interactive tools (4 hours)
- Advanced features (6 hours)

### Phase 4: Quality & Documentation (Week 6)

**Total Estimated Time**: 25 hours

- Quality assurance (12 hours)
- Documentation (6 hours)
- Final validation (4 hours)
- Performance optimization (3 hours)

### Critical Path Dependencies

1. **D1 → D2 → D3**: Dataset chain must complete before evaluation
2. **D3 → E1 → E2**: Evaluation system depends on dataset variants
3. **E1 → M1**: Metrics depend on evaluation data collection
4. **M1 → V1**: Visualization needs computed metrics
5. **All → I1 → I2**: Integration and testing require all components

### Risk Mitigation

- **Memory Issues**: Implement batch processing for attention analysis
- **Time Overruns**: Prioritize core functionality, defer advanced features
- **Integration Problems**: Early integration testing, modular design
- **Reproduction Issues**: Systematic validation against original results

---

## Success Criteria

### Functional Requirements

- [ ] Complete reproduction of Einstellung methodology in pure Mammoth
- [ ] Support for ResNet-18 and ViT backbones
- [ ] DER++ and EWC strategy implementations
- [ ] All ERI metrics (AD, PD, SFR) working correctly
- [ ] Attention analysis for ViT models
- [ ] Multi-seed statistical analysis
- [ ] Configurable patch injection system

### Performance Requirements

- [ ] Training time within 2x of original Avalanche implementation
- [ ] Memory usage under 8GB GPU for standard configurations
- [ ] Reproducible results across multiple runs
- [ ] Statistical significance maintained across implementations

### Quality Requirements

- [ ] 95%+ test coverage for core functionality
- [ ] All integration tests passing
- [ ] Documentation complete and accurate
- [ ] Code follows Mammoth conventions and style

### Scientific Requirements

- [ ] Results statistically equivalent to Avalanche version
- [ ] Attention patterns interpretable and meaningful
- [ ] ERI scores correlate with expected cognitive rigidity
- [ ] Cross-strategy comparisons show expected patterns

---

This plan provides a comprehensive roadmap for integrating the Einstellung Effect methodology into Mammoth while maintaining scientific rigor and technical excellence. The modular design allows for iterative development and testing, while the detailed task breakdown ensures nothing is overlooked in the implementation process.
