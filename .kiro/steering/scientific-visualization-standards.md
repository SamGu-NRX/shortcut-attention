---
inclusion: fileMatch
fileMatchPattern: "eri_vis/**/*.py"
---

# Scientific Visualization Standards

## Core Principles

When implementing scientific visualization systems, adhere to these fundamental principles:

### Publication Quality Standards

- All figures must be publication-ready with high DPI (≥300) output
- Use consistent color schemes that are colorblind-friendly
- Include comprehensive legends, axis labels, and titles
- Ensure figures are interpretable in both color and grayscale
- Maintain consistent typography and spacing across all visualizations

### Data Integrity and Reproducibility

- Implement deterministic output generation with consistent ordering
- Include metadata sidecars documenting all processing parameters
- Provide clear error handling for missing or corrupted data
- Support data validation with detailed diagnostic messages
- Enable reproducible results through version tracking and checksums

### Statistical Rigor

- Use appropriate confidence intervals (95% CI via t-distribution)
- Document sample sizes and statistical methods clearly
- Handle censored or incomplete data transparently
- Provide uncertainty quantification where applicable
- Include appropriate statistical tests and effect size reporting

### Performance and Scalability

- Design for datasets with 1000+ epochs and 10+ seeds
- Implement memory-efficient processing with chunked operations
- Support parallel processing for batch operations
- Optimize for sub-30-second visualization generation
- Provide progress indicators for long-running operations

## Implementation Guidelines

### Code Structure

```python
# Always include comprehensive type hints and docstrings
def compute_accuracy_curves(
    dataset: ERITimelineDataset,
    smoothing_window: int = 3
) -> Dict[str, AccuracyCurve]:
    """Compute smoothed accuracy curves with confidence intervals.

    Args:
        dataset: Timeline dataset containing method/seed/epoch data
        smoothing_window: Window size for moving average smoothing

    Returns:
        Dictionary mapping method names to accuracy curve objects

    Raises:
        ProcessingError: If insufficient data for reliable statistics
    """
```

### Error Handling

- Provide actionable error messages with context
- Include suggestions for data correction or parameter adjustment
- Log structured information for debugging (method, seed, epoch context)
- Gracefully handle edge cases without silent failures

### Testing Requirements

- Unit tests for all statistical computations with known synthetic data
- Integration tests for end-to-end pipeline validation
- Visual regression tests with golden reference images
- Performance benchmarks for scalability validation

## Visualization-Specific Standards

### Multi-Panel Figures

- Use consistent subplot spacing and alignment
- Ensure readable font sizes across all panels
- Include panel labels (A, B, C) for reference
- Maintain consistent axis ranges where appropriate

### Time Series Plots

- Include confidence bands for uncertainty visualization
- Mark significant events with clear annotations
- Use consistent time axis formatting
- Provide zero reference lines where meaningful

### Heatmaps and Matrices

- Use perceptually uniform colormaps
- Center diverging colormaps at meaningful zero points
- Handle missing data with clear visual indicators
- Include value annotations for precise reading

### Color and Accessibility

- Use colorblind-friendly palettes (e.g., viridis, plasma)
- Ensure sufficient contrast ratios
- Provide alternative visual encodings (patterns, shapes)
- Test visualizations in grayscale conversion

## File Organization

```
eri_vis/
├── styles.py          # Centralized styling configuration
├── plot_dynamics.py   # Multi-panel time series visualization
├── plot_heatmap.py    # Matrix and sensitivity analysis plots
├── errors.py          # Structured error handling
└── utils.py           # Common visualization utilities
```

## Documentation Standards

- Include example usage for all public APIs
- Provide troubleshooting guides for common issues
- Document all configuration parameters and their effects
- Include performance characteristics and limitations
- Reference relevant statistical methods and assumptions
