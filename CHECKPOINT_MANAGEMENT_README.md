# Einstellung Experiment Checkpoint Management

## Overview

The enhanced Einstellung experiment runner now includes comprehensive checkpoint management to avoid re-running expensive training operations. This system leverages Mammoth's native checkpointing functionality while providing intelligent discovery and reuse capabilities.

## Key Features

### 🔍 Automatic Checkpoint Discovery

- **Pattern-based search**: Finds existing checkpoints matching experiment parameters
- **Intelligent matching**: Accounts for Mammoth's complex naming convention with timestamps and UIDs
- **Multiple checkpoint types**: Supports both `_last.pt` and task-specific checkpoints
- **Sorted by recency**: Always uses the most recent matching checkpoint

### 🚀 Flexible Execution Modes

#### 1. **Skip Training Mode** (`--skip_training`)

```bash
python run_einstellung_experiment.py --model derpp --backbone resnet18 --skip_training
```

- **Evaluation only**: Skips training entirely, uses existing checkpoints
- **Fast results**: Get metrics in minutes instead of hours
- **Error handling**: Fails gracefully if no checkpoints are found

#### 2. **Auto Checkpoint Mode** (`--auto_checkpoint`)

```bash
python run_einstellung_experiment.py --model derpp --backbone resnet18 --auto_checkpoint
```

- **Intelligent decision**: Automatically uses existing checkpoints if found
- **Seamless fallback**: Trains from scratch if no checkpoints exist
- **No user interaction**: Perfect for automated pipelines

#### 3. **Force Retrain Mode** (`--force_retrain`)

```bash
python run_einstellung_experiment.py --model derpp --backbone resnet18 --force_retrain
```

- **Fresh training**: Ignores existing checkpoints completely
- **Debugging friendly**: Useful when testing code changes
- **Clean slate**: Ensures no contamination from previous runs

#### 4. **Interactive Mode** (default)

```bash
python run_einstellung_experiment.py --model derpp --backbone resnet18
```

- **User choice**: Prompts when checkpoints are found
- **Detailed info**: Shows checkpoint details (size, timestamp)
- **Flexible**: Choose to use, retrain, or cancel

### 📊 Enhanced Result Reporting

- **Source tracking**: Shows whether results came from checkpoints or training
- **Checkpoint metadata**: Displays checkpoint usage in comparative analysis
- **Performance metrics**: Time savings and efficiency indicators

## Technical Implementation

### Native Mammoth Integration

- **`--inference_only`**: Uses Mammoth's native evaluation-only mode
- **`--loadcheck`**: Leverages Mammoth's checkpoint loading system
- **`--savecheck`**: Automatic checkpoint saving for new training
- **No custom training**: Fully compatible with Mammoth's training pipeline

### Checkpoint Discovery Algorithm

```python
def find_existing_checkpoints(strategy, backbone, seed, dataset="seq-cifar100-einstellung"):
    # Pattern: {model}_{dataset}_{config}_{buffer_size}_{n_epochs}_{timestamp}_{uid}_{suffix}.pt
    patterns = [
        f"{strategy}_{dataset}_*_{buffer_size}_{n_epochs}_*_last.pt",
        f"{strategy}_{dataset}_*_{buffer_size}_{n_epochs}_*_1.pt",
    ]
    # Glob search + sort by modification time
```

### Error Handling

- **Graceful failures**: Clear error messages for missing checkpoints
- **Validation**: Checks checkpoint integrity before loading
- **Recovery**: Fallback options when things go wrong

## Usage Examples

### Basic Development Workflow

```bash
# First run: train from scratch
python run_einstellung_experiment.py --model derpp --backbone resnet18

# Subsequent runs: automatically reuse checkpoints
python run_einstellung_experiment.py --model derpp --backbone resnet18 --auto_checkpoint

# Quick evaluation for debugging
python run_einstellung_experiment.py --model derpp --backbone resnet18 --skip_training
```

### Comparative Analysis with Checkpoint Reuse

```bash
# Run all strategies, reusing existing checkpoints where possible
python run_einstellung_experiment.py --comparative --auto_checkpoint

# Force fresh training for all strategies
python run_einstellung_experiment.py --comparative --force_retrain
```

### CI/CD Pipeline Integration

```bash
# Automated mode for continuous integration
python run_einstellung_experiment.py --model derpp --auto_checkpoint --verbose
```

## Output Examples

### Checkpoint Discovery

```
🔍 Found existing checkpoints for derpp/resnet18/seed42:
================================================================================
 1. derpp_seq-cifar100-einstellung_default_500_50_20240709-143621_a1b2c3d4_last.pt
    Size: 245.3 MB | Modified: 20240709-143621
 2. ewc_on_seq-cifar100-einstellung_default_0_50_20240708-091234_e5f6g7h8_last.pt
    Size: 189.7 MB | Modified: 20240708-091234

================================================================================
What would you like to do?
  [1] Use existing checkpoint (skip training)
  [2] Retrain from scratch
  [3] Cancel

Your choice [1-3]: 1
✅ Using existing checkpoint: derpp_seq-cifar100-einstellung_default_500_50_20240709-143621_a1b2c3d4_last.pt
```

### Comparative Analysis with Source Tracking

```
===================================================================================================
COMPARATIVE ANALYSIS
===================================================================================================
Strategy        Backbone             ERI Score    Perf. Deficit   Ad. Delay    Final Acc  Source
---------------------------------------------------------------------------------------------------
sgd             resnet18             N/A          N/A             N/A          72.45%     Training
derpp           resnet18             N/A          N/A             N/A          74.13%     Checkpoint
ewc_on          resnet18             N/A          N/A             N/A          71.89%     Checkpoint
derpp           vit                  N/A          N/A             N/A          78.92%     Training
```

## Time & Resource Savings

### Typical Training Times

- **ResNet-18**: ~45 minutes per experiment
- **ViT**: ~90 minutes per experiment
- **Evaluation only**: ~3-5 minutes per experiment

### Storage Requirements

- **Checkpoint size**: ~200-300 MB per model
- **Result files**: ~10-50 MB per experiment
- **Total for full study**: ~2-5 GB

### Development Efficiency

- **Iteration time**: Reduced from hours to minutes
- **Debugging**: Instant evaluation for code changes
- **Resource usage**: 90%+ reduction in GPU time for repeated runs

## Best Practices

### 1. Development Workflow

```bash
# Step 1: Initial training (once)
python run_einstellung_experiment.py --model derpp --force_retrain

# Step 2: Iterative development (many times)
python run_einstellung_experiment.py --model derpp --skip_training

# Step 3: Final validation (occasionally)
python run_einstellung_experiment.py --model derpp --force_retrain
```

### 2. Checkpoint Management

- **Regular cleanup**: Remove old checkpoints to save disk space
- **Naming convention**: Use `--ckpt_name` for important experiments
- **Backup strategy**: Archive important checkpoints before cleanup

### 3. Reproducibility

- **Seed consistency**: Always use the same seed for comparable results
- **Checkpoint versioning**: Track checkpoint creation parameters
- **Documentation**: Note which results came from checkpoints vs. fresh training

## Troubleshooting

### Common Issues

#### No Checkpoints Found

```
❌ No existing checkpoints found for evaluation-only mode
```

**Solution**: Run training first or use `--auto_checkpoint` for fallback

#### Checkpoint Loading Fails

```
ERROR: Process failed with return code 2
```

**Solution**: Check if checkpoint file is corrupted, use `--force_retrain`

#### Pattern Matching Issues

```
Found 0 checkpoint(s) for strategy/backbone
```

**Solution**: Check naming convention or use manual checkpoint path

### Debug Mode

```bash
# Enable verbose logging for troubleshooting
python run_einstellung_experiment.py --model derpp --verbose --skip_training
```

## Integration with Existing Code

The checkpoint management system is designed to be **completely backward compatible**:

- **Existing scripts**: Work without modification
- **Default behavior**: Interactive mode maintains user control
- **Optional features**: All checkpoint features are opt-in
- **Native compatibility**: Uses standard Mammoth arguments

## Future Enhancements

### Planned Features

- **Checkpoint metadata**: Store experiment parameters in checkpoint files
- **Partial training**: Resume interrupted experiments
- **Remote storage**: Support for cloud checkpoint storage
- **Automatic cleanup**: Smart checkpoint retention policies

### Contribution Guidelines

- **Testing**: Run `python test_checkpoint_management.py` before submitting
- **Documentation**: Update this README for new features
- **Compatibility**: Maintain backward compatibility with existing code

---

This checkpoint management system transforms the Einstellung experiment workflow from a time-consuming process to an efficient, developer-friendly experience while maintaining full scientific rigor and reproducibility.
