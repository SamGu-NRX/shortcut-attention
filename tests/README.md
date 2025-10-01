# Test Suite Organization

This directory contains the complete test suite for the continual learning framework, organized by functionality.

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py                              # Pytest configuration and fixtures
│
├── cache/                                   # Dataset caching tests
│   ├── test_224_cache_functionality.py
│   ├── test_cache_build.py
│   ├── test_cache_format_validation.py
│   ├── test_cache_loading_validation.py
│   ├── test_cache_management.py
│   ├── test_cache_performance.py
│   └── test_cross_method_cache_consistency.py
│
├── einstellung/                             # Einstellung Effect tests
│   ├── test_einstellung_cache_validation.py
│   ├── test_einstellung_optimization.py
│   ├── test_einstellung_performance.py
│   ├── test_einstellung_performance_pipeline.py
│   ├── test_evaluation_subset_caching.py
│   └── test_epoch_calculation.py
│
├── eri_vis/                                 # ERI visualization tests
│   ├── test_calculation_fixes.py
│   ├── test_cli.py
│   ├── test_data_loader.py
│   ├── test_dataset.py
│   ├── test_heatmap.py
│   ├── test_hooks.py
│   ├── test_hooks_integration.py
│   ├── test_mammoth_integration.py
│   ├── test_multi_method_integration.py
│   ├── test_multi_method_validation.py
│   ├── test_overall_eri.py
│   ├── test_plot_dynamics.py
│   ├── test_processing.py
│   └── test_styles.py
│
├── integration/                             # Integration tests
│   ├── run_integration_tests.py
│   ├── test_backward_compatibility.py
│   ├── test_checkpoint_management.py
│   ├── test_cross_method_deterministic.py
│   ├── test_data_loading_integration.py
│   ├── test_deterministic_fix.py
│   ├── test_dgr_full_integration.py
│   ├── test_error_handling.py
│   ├── test_experiment_cache_integration.py
│   ├── test_integrated_methods.py
│   ├── test_quick_verification.py
│   ├── test_real_data_validation.py
│   └── test_recursion_fix.py
│
├── models/                                  # Model-specific tests
│   ├── conftest.py
│   ├── test_dgr_integration.py
│   ├── test_gpm_integration.py
│   ├── test_hybrid_methods.py
│   ├── test_integrated_methods_registry.py
│   ├── test_model_wrappers.py
│   ├── test_model_wrappers_minimal.py
│   ├── test_model_wrappers_simple.py
│   └── test_scratch_t2.py
│
├── performance/                             # Performance optimization tests
│   ├── test_auto_performance.py
│   ├── test_cuda_improvements.py
│   ├── test_minimal_optimization.py
│   ├── test_performance_monitoring.py
│   └── test_scratch_t2_efficiency.py
│
├── vit/                                     # Vision Transformer tests
│   └── test_vit_einstellung_debug.py
│
└── [Root-level tests]                       # Framework-wide tests
    ├── test_baseline_experiment_integration.py
    ├── test_baseline_method_integration.py
    ├── test_basic_functionality.py
    ├── test_comparative_aggregation.py
    ├── test_comparative_visualization_pipeline.py
    ├── test_comprehensive_integration.py
    ├── test_datasets.py
    ├── test_statistical_analysis.py
    ├── test_statistical_end_to_end.py
    ├── test_statistical_integration.py
    └── [Model-specific smoke tests: test_bias.py, test_bic.py, etc.]
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test categories
```bash
pytest tests/einstellung/          # Einstellung tests only
pytest tests/cache/                # Cache tests only
pytest tests/performance/          # Performance tests only
pytest tests/integration/          # Integration tests only
```

### Run with coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

## Test Categories

### Cache Tests (`tests/cache/`)
Tests for dataset caching functionality, ensuring efficient data loading and consistency across experiments.

### Einstellung Tests (`tests/einstellung/`)
Tests specific to the Einstellung Effect experiments, including performance optimizations and data validation.

### ERI Visualization Tests (`tests/eri_vis/`)
Tests for the ERI (Einstellung Rigidity Index) visualization and analysis pipeline.

### Integration Tests (`tests/integration/`)
End-to-end integration tests covering interactions between different system components.

### Model Tests (`tests/models/`)
Tests for specific continual learning models (DGR, GPM, etc.) and their integration with the framework.

### Performance Tests (`tests/performance/`)
Tests for performance optimizations, CUDA improvements, and efficiency enhancements.

### VIT Tests (`tests/vit/`)
Tests specific to Vision Transformer models in the Einstellung context.

## Guidelines

- All new tests should be placed in the appropriate subdirectory
- Use descriptive test names following the pattern `test_<feature>_<aspect>.py`
- Include docstrings explaining what each test validates
- Use pytest fixtures from `conftest.py` for common setup
- Mark slow tests with `@pytest.mark.slow` decorator
- Keep tests independent and reproducible
