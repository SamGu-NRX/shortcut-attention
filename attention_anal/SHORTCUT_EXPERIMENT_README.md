# Shortcut Feature Investigation in Continual Learning

This experiment investigates how different continual learning methods handle shortcut features using a custom CIFAR-10 setup with Vision Transformers.

## Experiment Overview

### Objective
Investigate the effect of shortcut features in continual learning by comparing memory-based (DER++) and regularization-based (EWC) methods.

### Task Setup
- **Task 1**: airplane, automobile (potential shortcuts: sky, road)
- **Task 2**: bird, truck (potential shortcuts: sky, road/wheels)

### Architecture
- **Backbone**: Vision Transformer (ViT) for attention map analysis
- **Input Size**: 32x32 (CIFAR-10 resolution)
- **Classes**: 4 total (2 per task)

### Methods Compared
1. **DER++** (Memory-based): Uses experience replay with dark experience replay
2. **EWC** (Regularization-based): Uses Elastic Weight Consolidation

## Files Created

### Core Implementation
- `datasets/seq_cifar10_custom.py` - Custom CIFAR-10 dataset with specific class ordering
- `utils/attention_visualization.py` - Attention map extraction and visualization
- `utils/network_flow_visualization.py` - Network activation analysis
- `experiments/shortcut_investigation.py` - Main experiment runner
- `experiments/analyze_results.py` - Results analysis and comparison
- `run_shortcut_experiment.py` - Simple script to run experiments

### Key Features
- **Reproducible Seeds**: Same random initialization across all experiments
- **Attention Analysis**: Extract and visualize attention maps from ViT
- **Network Flow Analysis**: Analyze activation patterns through the network
- **Comparative Analysis**: Compare methods across multiple metrics

## Quick Start

### 1. Run a Single Experiment
```bash
# Run DER++ with default settings
python run_shortcut_experiment.py --method derpp --seed 42

# Run EWC with default settings
python run_shortcut_experiment.py --method ewc_on --seed 42
```

### 2. Run Full Comparison
```bash
# Run both methods with multiple seeds
python run_shortcut_experiment.py --comparison
```

### 3. Run with Custom Parameters
```bash
# Custom epochs and debug mode
python run_shortcut_experiment.py --method derpp --epochs 20 --debug
```

## Detailed Usage

### Running Individual Experiments
```bash
# DER++ experiment
python main.py --dataset seq-cifar10-custom --model derpp --backbone vit \
    --seed 42 --n_epochs 10 --batch_size 32 --buffer_size 200 \
    --alpha 0.1 --beta 0.5 --nowand 1

# EWC experiment  
python main.py --dataset seq-cifar10-custom --model ewc_on --backbone vit \
    --seed 42 --n_epochs 10 --batch_size 32 --e_lambda 0.4 --gamma 0.85 --nowand 1
```

### Analyzing Results
```bash
# Analyze experiment results
python experiments/analyze_results.py --results_dir results/shortcut_investigation_YYYYMMDD_HHMMSS
```

## Expected Outputs

### During Training
- Model checkpoints and training logs
- Performance metrics (accuracy, forgetting, etc.)
- Task-specific results

### After Analysis
- **Attention Maps**: Visualizations showing where the model focuses
- **Activation Patterns**: Network flow analysis through layers
- **Comparative Plots**: Method comparison across metrics
- **Summary Report**: Comprehensive analysis results

### Directory Structure
```
results/
└── shortcut_investigation_YYYYMMDD_HHMMSS/
    ├── derpp_seed_42/
    │   ├── model.pth
    │   ├── results.json
    │   └── task_1_analysis/
    │       ├── attention_*.png
    │       ├── activation_flow.png
    │       └── tsne_*.png
    ├── ewc_on_seed_42/
    │   └── ...
    ├── analysis/
    │   ├── attention_entropy_comparison.png
    │   ├── experiment_dashboard.png
    │   └── summary_report.md
    └── experiment_results.json
```

## Key Analysis Questions

### 1. Attention Pattern Analysis
- Do different methods focus on different image regions?
- How do attention patterns change between tasks?
- Are shortcut features (sky, road) being used?

### 2. Network Activation Analysis
- How do activation patterns differ between methods?
- Which layers show the most change between tasks?
- Are there signs of catastrophic forgetting in activations?

### 3. Method Comparison
- Which method better handles shortcut features?
- How do memory-based vs regularization-based approaches differ?
- What are the trade-offs in terms of performance and feature usage?

## Customization

### Modify Task Setup
Edit `datasets/seq_cifar10_custom.py` to change:
- Class ordering
- Number of classes per task
- Data augmentation strategies

### Add New Analysis
Extend `utils/attention_visualization.py` or `utils/network_flow_visualization.py` to add:
- New visualization types
- Additional metrics
- Different analysis methods

### Change Hyperparameters
Modify `run_shortcut_experiment.py` or use command-line arguments to adjust:
- Learning rates
- Buffer sizes (for DER++)
- Regularization strengths (for EWC)
- Number of epochs

## Important Notes

### Reproducibility
- All experiments use the same random seeds
- Custom class order ensures consistent task splits
- Deterministic operations where possible

### Hardware Requirements
- GPU recommended for faster training
- Sufficient memory for Vision Transformer
- Storage space for attention map visualizations

### Limitations
- Limited to CIFAR-10 resolution (32x32)
- Only 2 tasks for simplicity
- Specific to Vision Transformer architecture

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Missing dependencies**: Install required packages (matplotlib, seaborn, sklearn)
3. **Dataset download**: Ensure internet connection for CIFAR-10 download

### Debug Mode
Use `--debug` flag for:
- Reduced number of training steps
- More verbose logging
- Faster iteration during development

## Future Extensions

### Possible Improvements
1. **More Tasks**: Extend to full CIFAR-10 (5 tasks)
2. **Different Architectures**: Compare with ResNet, ConvNet
3. **Additional Methods**: Include more continual learning algorithms
4. **Shortcut Mitigation**: Test methods to reduce shortcut reliance
5. **Quantitative Metrics**: Develop metrics to measure shortcut usage

### Research Questions
- How do different ViT sizes affect shortcut learning?
- Can we design tasks that specifically test shortcut robustness?
- What role does pre-training play in shortcut feature usage?

## References

- DER++: Dark Experience for General Continual Learning
- EWC: Overcoming catastrophic forgetting in neural networks
- Vision Transformer: An Image is Worth 16x16 Words
- Mammoth: A PyTorch Framework for Benchmarking Continual Learning
