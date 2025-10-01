# Project Directory Structure

This document describes the organization of the continual learning framework codebase.

## Root Directory

```
shortcut-attention/
├── main.py                          # Main entry point for running experiments
├── run_einstellung_experiment.py    # CLI for Einstellung Effect experiments
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Pytest configuration
├── README.md                        # Main project documentation
└── [Other docs]                     # Additional documentation files
```

## Core Directories

### `/experiments/` - Experiment Orchestration
```
experiments/
├── einstellung/                     # Einstellung Effect experiments module
│   ├── README.md                   # Module documentation
│   ├── config.py                   # Configuration dataclasses
│   ├── runner.py                   # Experiment execution
│   ├── args_builder.py             # Argument construction
│   ├── batch.py                    # Batch orchestration
│   ├── storage.py                  # Checkpoint management
│   ├── analysis.py                 # Data analysis
│   ├── visualization.py            # Plot generation
│   └── reporting.py                # Report generation
├── configs/                         # Experiment configurations
└── [Other experiment runners]
```

**Purpose**: High-level experiment orchestration, batch processing, and analysis pipelines.

### `/tests/` - Test Suite
```
tests/
├── README.md                        # Test organization documentation
├── conftest.py                      # Pytest fixtures and configuration
│
├── cache/                           # Dataset caching tests
│   ├── test_cache_build.py
│   ├── test_cache_performance.py
│   └── [7 cache-related tests]
│
├── einstellung/                     # Einstellung-specific tests
│   ├── test_einstellung_performance.py
│   ├── test_epoch_calculation.py
│   └── [6 einstellung tests]
│
├── eri_vis/                         # ERI visualization tests
│   ├── test_heatmap.py
│   ├── test_plot_dynamics.py
│   └── [14 visualization tests]
│
├── integration/                     # Integration tests
│   ├── test_dgr_full_integration.py
│   ├── test_checkpoint_management.py
│   └── [12 integration tests]
│
├── models/                          # Model-specific tests
│   ├── test_dgr_integration.py
│   ├── test_gpm_integration.py
│   └── [8 model tests]
│
├── performance/                     # Performance tests
│   ├── test_cuda_improvements.py
│   ├── test_performance_monitoring.py
│   └── [5 performance tests]
│
├── vit/                             # Vision Transformer tests
│   └── test_vit_einstellung_debug.py
│
└── [Framework-wide tests]           # Core functionality tests
    ├── test_datasets.py
    ├── test_statistical_analysis.py
    └── [50+ model smoke tests]
```

**Purpose**: Comprehensive test coverage organized by functionality.

### `/models/` - Model Implementations
```
models/
├── sgd.py                           # Baseline models
├── er.py, derpp.py                 # Replay-based methods
├── dgr.py, gpm.py                  # Advanced methods
├── utils/                           # Model utilities
└── [70+ continual learning models]
```

**Purpose**: Implementations of continual learning algorithms.

### `/datasets/` - Dataset Implementations
```
datasets/
├── seq_cifar100.py                  # Standard datasets
├── seq_cifar100_einstellung.py      # Einstellung variants
├── utils/                           # Dataset utilities
└── [20+ dataset implementations]
```

**Purpose**: Dataset loaders and transformations.

### `/utils/` - Utility Modules
```
utils/
├── einstellung_integration.py       # Einstellung Effect integration
├── training.py                      # Training utilities
├── args.py                          # Argument parsing
└── [Other utilities]
```

**Purpose**: Shared utilities used across the framework.

### `/backbone/` - Neural Network Architectures
```
backbone/
├── resnet.py                        # ResNet architectures
├── vit.py                           # Vision Transformers
└── [Other backbones]
```

**Purpose**: Neural network backbone implementations.

### `/docs/` - Documentation
```
docs/
├── einstellung/                     # Einstellung documentation
│   ├── EINSTELLUNG_README.md
│   ├── EINSTELLUNG_FIXLIST.md
│   └── [5 einstellung docs]
│
├── implementation/                  # Implementation notes
│   ├── CHECKPOINT_MANAGEMENT_README.md
│   ├── CUDA_PERFORMANCE_SUMMARY.md
│   └── [6 implementation docs]
│
└── [Sphinx documentation]           # API documentation
```

**Purpose**: Project documentation and implementation notes.

## Deprecated/Removed

The following directories and files were part of a cleanup effort and are no longer in the repository:

### Removed Files
- `test_einstellung_diagnostics.py` - Diagnostic tool (obsolete)
- `test_einstellung_fix.py` - Specific fix test (integrated)
- `test_einstellung_simple.py` - Simple diagnostic (redundant)
- `setup_einstellung_integration.py` - Setup script (functionality in experiments/)
- `fix_*.py` - One-time fix scripts (no longer needed)
- `demo_*.py` - Demo scripts (no longer needed)
- `validate_*.py` - Validation scripts (functionality in tests/)
- `cuda_performance_*.py` - Diagnostic tools (functionality integrated)

### Removed Directories
- `test_simple_results/` - Temporary test outputs (in .gitignore)

## Navigation Guide

### Working with Einstellung Experiments
1. Main implementation: `/experiments/einstellung/`
2. Tests: `/tests/einstellung/`
3. Documentation: `/docs/einstellung/`
4. Entry point: `run_einstellung_experiment.py`

### Running Tests
1. All tests: `pytest tests/`
2. Specific category: `pytest tests/einstellung/`
3. Test docs: See `/tests/README.md`

### Adding New Features
1. Models: Add to `/models/`
2. Datasets: Add to `/datasets/`
3. Tests: Add to appropriate `/tests/` subdirectory
4. Documentation: Update relevant README files

## Best Practices

1. **Tests**: Always add tests in the appropriate `/tests/` subdirectory
2. **Documentation**: Keep README files up-to-date with changes
3. **Organization**: Follow the established directory structure
4. **Cleanup**: Use `.gitignore` for generated files and results
5. **Modularity**: Keep experiment orchestration in `/experiments/`
