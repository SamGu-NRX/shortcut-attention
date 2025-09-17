# ERI Visualization CLI Examples

This document demonstrates the usage of the ERI visualization CLI tool with working examples.

## Basic Usage

### Single CSV File

Generate dynamics plot from a single CSV file:

```bash
python tools/plot_eri.py --csv sample_eri_data.csv --outdir results/
```

### With Robustness Heatmap

Generate both dynamics plot and robustness heatmap:

```bash
python tools/plot_eri.py --csv sample_eri_data.csv --outdir results/ --tau-grid 0.5 0.55 0.6 0.65 0.7
```

### Method Filtering

Generate plots for specific methods only:

```bash
python tools/plot_eri.py --csv sample_eri_data.csv --outdir results/ --methods sgd ewc_on
```

## Batch Processing

### Multiple CSV Files

Process multiple CSV files with glob pattern:

```bash
python tools/plot_eri.py --csv "sample_eri_data*.csv" --outdir results/
```

### With Batch Summary

Generate individual plots plus a combined summary:

```bash
python tools/plot_eri.py --csv "sample_eri_data*.csv" --outdir results/ --batch-summary
```

## Customization Options

### Custom Parameters

Adjust tau threshold and smoothing:

```bash
python tools/plot_eri.py --csv sample_eri_data.csv --outdir results/ --tau 0.65 --smooth 5
```

### Different Output Format

Generate PNG instead of PDF:

```bash
python tools/plot_eri.py --csv sample_eri_data.csv --outdir results/ --format png
```

### Publication Style

Use publication-ready styling:

```bash
python tools/plot_eri.py --csv sample_eri_data.csv --outdir results/ --style publication --dpi 300
```

## Logging Options

### Verbose Output

Enable detailed logging:

```bash
python tools/plot_eri.py --csv sample_eri_data.csv --outdir results/ --verbose
```

### Quiet Mode

Suppress progress messages (errors only):

```bash
python tools/plot_eri.py --csv sample_eri_data.csv --outdir results/ --quiet
```

## Expected CSV Format

The CLI expects CSV files with the following columns:

- `method`: Method name (e.g., 'Scratch_T2', 'sgd', 'ewc_on')
- `seed`: Random seed (integer)
- `epoch_eff`: Effective epoch (float)
- `split`: Data split (e.g., 'T2_shortcut_normal', 'T2_shortcut_masked')
- `acc`: Accuracy value (float between 0 and 1)

Example CSV content:

```csv
method,seed,epoch_eff,split,acc
Scratch_T2,42,0.0,T2_shortcut_normal,0.10
Scratch_T2,42,1.0,T2_shortcut_normal,0.25
sgd,42,0.0,T2_shortcut_normal,0.20
sgd,42,1.0,T2_shortcut_normal,0.30
```

## Error Handling

The CLI provides clear error messages for common issues:

### Invalid tau value:

```bash
$ python tools/plot_eri.py --csv data.csv --outdir results/ --tau 1.5
CLI Error: Tau value must be between 0.0 and 1.0, got 1.5
```

### File not found:

```bash
$ python tools/plot_eri.py --csv nonexistent.csv --outdir results/
CLI Error: CSV file not found: nonexistent.csv
```

### Invalid methods:

```bash
$ python tools/plot_eri.py --csv data.csv --outdir results/ --methods invalid_method
CLI Error: None of the requested methods found in data. Available: ['Scratch_T2', 'sgd'], Requested: ['invalid_method']
```

## Output Files

The CLI generates the following output files:

### Dynamics Plot

- **File**: `{csv_name}_dynamics.pdf`
- **Content**: 3-panel figure with accuracy trajectories, performance deficits, and shortcut forgetting rates

### Robustness Heatmap (when --tau-grid specified)

- **File**: `{csv_name}_heatmap.pdf`
- **Content**: Adaptation delay sensitivity analysis across different tau values

### Batch Summary (when --batch-summary specified)

- **File**: `batch_summary_dynamics.pdf`
- **Content**: Combined visualization from all processed CSV files

All generated files are saved in the specified output directory with automatic directory creation if needed.
