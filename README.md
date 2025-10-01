# Diagnosing Shortcut-Induced Rigidity in Continual Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**The Einstellung Rigidity Index (ERI): A Framework for Detecting Shortcut Learning in Sequential Tasks**

This repository implements a comprehensive diagnostic framework for detecting and measuring shortcut-induced rigidity in continual learning systems. The **Einstellung Rigidity Index (ERI)** quantifies how models become overly reliant on spurious correlations learned in earlier tasks, impeding adaptation to new challenges.

> **Paper**: *Diagnosing Shortcut-Induced Rigidity in Continual Learning: The Einstellung Rigidity Index (ERI)*  
> **Authors**: Kai Gu and Weishi Shi  
> **Institution**: Department of Computer Science and Engineering, University of North Texas  
> **Full Paper**: See `main.tex` in this repository

---

## 🔍 Overview

While catastrophic forgetting has been the primary focus in continual learning research, this work addresses its counterpart: **shortcut-induced rigidity**. Instead of discarding past knowledge, models may preferentially reuse features from earlier tasks—including spurious shortcuts that are suboptimal for current tasks. This mirrors the cognitive **Einstellung effect**, where prior strategies impede the discovery of better solutions.

### What is ERI?

The **Einstellung Rigidity Index (ERI)** is a three-component diagnostic that distinguishes genuine transfer learning from cue-inflated performance:

1. **Adaptation Delay (AD)**: How quickly a continual learner reaches an accuracy threshold relative to a from-scratch baseline
2. **Performance Deficit (PD)**: The final accuracy gap between continual and scratch models
3. **Relative Shortcut Feature Reliance (SFR_rel)**: Additional reliance on suspected shortcuts compared to baseline, measured via masking interventions

### Key Features

- 🎯 **Diagnostic Framework**: Detects shortcut learning without modifying training procedures
- 📊 **Comprehensive Metrics**: Three interpretable facets (AD, PD, SFR_rel) for rigidity assessment
- 🔬 **Experimental Protocol**: Controlled two-phase CIFAR-100 benchmark with deterministic shortcut injection
- 🛠️ **Mammoth Integration**: Built on the robust [Mammoth continual learning framework](https://github.com/aimagelab/mammoth)
- 📈 **Visualization Tools**: Automated generation of dynamics plots, heatmaps, and statistical reports
- 🧪 **Multiple Baselines**: Comparison across Naive, EWC, Replay, and other CL strategies

---

## 📑 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Understanding ERI](#-understanding-eri)
- [Experimental Design](#-experimental-design)
- [Running Experiments](#-running-experiments)
- [Visualization](#-visualization)
- [Results Interpretation](#-results-interpretation)
- [Architecture](#-architecture)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch >= 2.1.0 (for scaled_dot_product_attention support)
- CUDA (recommended for faster training)

### Setup

```bash
# Clone the repository
git clone https://github.com/SamGu-NRX/shortcut-attention.git
cd shortcut-attention

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional dependencies for specific models
pip install -r requirements-optional.txt
```

**Note**: If you cannot support PyTorch >= 2.1.0, uncomment lines 136-139 under `scaled_dot_product_attention` in `backbone/vit.py`.

---

## 🎯 Quick Start

### Basic Einstellung Experiment

Run a simple experiment to test shortcut-induced rigidity:

```bash
# Run with DER++ method and ResNet18 backbone
python run_einstellung_experiment.py --model derpp --backbone resnet18

# Run with Vision Transformer for attention analysis
python run_einstellung_experiment.py --model derpp --backbone vit_base_patch16_224

# Run comparative analysis across multiple methods
python run_einstellung_experiment.py --comparative
```

### Using Mammoth Directly

You can also use the Einstellung dataset with Mammoth's main interface:

```bash
# Standard Einstellung experiment
python main.py --dataset seq-cifar100-einstellung --model derpp --backbone resnet18

# With best hyperparameters
python main.py --dataset seq-cifar100-einstellung --model ewc_on --model_config best

# Multi-seed experiment
python main.py --dataset seq-cifar100-einstellung --model sgd --seed 42 43 44 45
```

### Generate Visualizations

After running experiments, generate ERI visualizations:

```bash
# Generate dynamics plots from results
python tools/plot_eri.py --csv logs/eri_metrics.csv --outdir results/

# With custom threshold and smoothing
python tools/plot_eri.py --csv logs/*.csv --tau 0.65 --smooth 5

# Robustness analysis with multiple thresholds
python tools/plot_eri.py --csv data.csv --tau-grid 0.5 0.55 0.6 0.65 0.7
```


---

## 📖 Understanding ERI

### The Problem: Shortcut-Induced Rigidity

In continual learning, models can exploit **shortcuts**—incidental correlations (e.g., distinctive colors, textures, background artifacts) that predict labels without causal meaning. When these shortcuts are learned early, the same mechanisms that prevent catastrophic forgetting can entrench reliance on these spurious features, creating **rigidity** that impedes adaptation to new tasks.

### The Three Components of ERI

#### 1. Adaptation Delay (AD)

Measures how many effective epochs a continual learner needs to reach an accuracy threshold τ compared to a from-scratch baseline:

```
AD = E_CL(τ) - E_S(τ)
```

- **Negative AD**: Continual learner reaches threshold faster (may indicate shortcut reuse)
- **Positive AD**: Continual learner is slower (may indicate rigidity or poor transfer)
- **Zero AD**: Similar learning speed

#### 2. Performance Deficit (PD)

Compares final accuracy on shortcut-bearing data between scratch and continual models:

```
PD = A_S_patch* - A_CL_patch*
```

- **Negative PD**: Continual learner achieves higher accuracy (beware of shortcut inflation)
- **Positive PD**: Continual learner underperforms
- **Zero PD**: Similar final performance

#### 3. Relative Shortcut Feature Reliance (SFR_rel)

Measures additional reliance on shortcuts compared to baseline via masking intervention:

```
Δ_M = A_patch* - A_mask*  (performance drop when shortcut masked)
SFR_rel = Δ_CL - Δ_S
```

- **Positive SFR_rel**: Continual learner relies more on shortcuts
- **Negative SFR_rel**: Continual learner is more robust
- **Zero SFR_rel**: Similar shortcut dependence

### Interpreting ERI Patterns

**Red-flag pattern (likely rigidity)**:
- AD ≪ 0 (faster learning via shortcuts)
- PD ≤ 0 (inflated performance)
- SFR_rel > 0 (higher shortcut reliance)

**Benign transfer (healthy adaptation)**:
- AD ≈ 0 or > 0
- PD ≥ 0
- SFR_rel ≤ 0

**Ambiguous cases**: Require additional probes such as representational drift analysis (CKA), calibration under masking, or counterfactual patch placement.

---

## 🔬 Experimental Design

### Two-Phase CIFAR-100 Protocol

**Phase 1 (Semantic Learning)**:
- 8 CIFAR-100 superclasses (40 fine-grained classes)
- Heavy augmentations to force semantic feature learning
- No shortcuts present

**Phase 2 (Shortcut Available)**:
- 4 CIFAR-100 superclasses (20 fine-grained classes)
- Magenta patch shortcuts injected into 50% of "shortcut superclasses" (SC)
- Remaining classes are "non-shortcut superclasses" (NSC)

### Shortcut Implementation

- **Type**: Spatial magenta patches (4×4 pixels by default)
- **Placement**: Random location per image (deterministic per sample index)
- **Injection Rate**: 50% of SC images during training
- **Masking**: Deterministic removal for evaluation purposes

### Evaluation Subsets

1. **T1**: Phase 1 classes (retention check)
2. **T2 Normal**: Phase 2 SC with patches visible
3. **T2 Masked**: Phase 2 SC with patches removed
4. **T2 Non-shortcut**: Phase 2 NSC classes

### Supported Methods

Built on [Mammoth](https://github.com/aimagelab/mammoth), this framework supports 70+ continual learning methods:

- **Regularization-based**: EWC, SI, LwF, etc.
- **Replay-based**: ER, DER, DER++, GDumb, etc.
- **Architecture-based**: PNN, HAL, etc.
- **Prompt-based**: L2P, DualPrompt, CODA-Prompt, etc.
- **Others**: iCaRL, BiC, LUCIR, etc.

See the full list in the [Mammoth documentation](https://aimagelab.github.io/mammoth/).

---

## 🧪 Running Experiments

### Standard Experiment Runner

The `run_einstellung_experiment.py` script provides a high-level interface:

```bash
# Basic experiment
python run_einstellung_experiment.py \
    --model derpp \
    --backbone resnet18 \
    --seed 42

# Multi-seed run for statistical robustness
python run_einstellung_experiment.py \
    --model ewc_on \
    --backbone resnet18 \
    --seed 42 43 44 45

# With custom hyperparameters
python run_einstellung_experiment.py \
    --model derpp \
    --backbone resnet18 \
    --lr 0.001 \
    --buffer_size 500 \
    --alpha 0.5 \
    --beta 0.5

# Comparative analysis across methods
python run_einstellung_experiment.py \
    --comparative \
    --methods sgd ewc_on derpp \
    --seed 42
```

### Direct Mammoth Usage

For more control, use Mammoth's main interface:

```bash
# Basic Einstellung dataset
python main.py \
    --dataset seq-cifar100-einstellung \
    --model sgd \
    --backbone resnet18 \
    --lr 0.03 \
    --n_epochs 50

# With best hyperparameters
python main.py \
    --dataset seq-cifar100-einstellung \
    --model derpp \
    --model_config best

# Attention analysis with ViT
python main.py \
    --dataset seq-cifar100-einstellung \
    --model derpp \
    --backbone vit_base_patch16_224 \
    --enable_attention_analysis
```

### Configuration Files

Create YAML configuration files for reproducible experiments:

```yaml
# config/einstellung_derpp.yaml
dataset: seq-cifar100-einstellung
model: derpp
backbone: resnet18
lr: 0.001
buffer_size: 500
alpha: 0.5
beta: 0.5
n_epochs: 50
batch_size: 32
seed: [42, 43, 44, 45]

# Einstellung-specific options
shortcut_patch_size: 4
shortcut_injection_rate: 0.5
tau_threshold: 0.6
```

Run with: `python main.py --config config/einstellung_derpp.yaml`

---

## 📊 Visualization

### ERI Dynamics Plots

Generate comprehensive ERI visualizations:

```bash
# Basic visualization
python tools/plot_eri.py \
    --csv logs/eri_metrics.csv \
    --outdir results/

# With custom parameters
python tools/plot_eri.py \
    --csv logs/*.csv \
    --outdir figs/ \
    --tau 0.65 \
    --smooth 5 \
    --baseline sgd \
    --separate-panels

# Robustness analysis
python tools/plot_eri.py \
    --csv data.csv \
    --tau-grid 0.5 0.55 0.6 0.65 0.7 0.75 0.8
```

The tool generates:
- **Panel A**: Shorthand accuracy (patched/masked average) with AD markers
- **Panel B**: Performance deficit trajectories (PDₜ)
- **Panel C**: Relative shortcut forgetting (SFR_rel)
- **Supplementary**: Patched and masked accuracy plots

### Batch Processing

Process multiple experiment results:

```bash
# Multiple CSV files with glob pattern
python tools/plot_eri.py \
    --csv "logs/run_*.csv" \
    --outdir results/ \
    --batch-summary

# Cross-method comparison
python demo_statistical_analysis.py \
    --results-dir comparative_results/ \
    --output-dir analysis/
```

### Interactive Analysis

For detailed exploration:

```python
from eri_vis import ERITimelineDataset, ERIDynamicsPlotter, CorrectedERICalculator

# Load data
dataset = ERITimelineDataset()
dataset.load_from_csv("logs/eri_metrics.csv")

# Calculate metrics
calculator = CorrectedERICalculator(tau=0.6, smoothing_window=3)
metrics = calculator.compute_all_metrics(dataset)

# Generate plots
plotter = ERIDynamicsPlotter()
plotter.create_three_panel_figure(dataset, metrics, output_path="eri_dynamics.pdf")
```

---

## 📈 Results Interpretation

### ERI Score Interpretation

The composite ERI score ranges from 0 to 1:

- **0.0-0.3**: Low rigidity (good adaptation)
- **0.3-0.6**: Moderate rigidity
- **0.6-1.0**: High rigidity (poor adaptation)

### Expected Patterns by Method

Based on our experiments:

- **SGD/Naive**: High rigidity (ERI > 0.6) due to catastrophic forgetting and shortcut exploitation
- **EWC**: Moderate rigidity (ERI 0.4-0.6) - regularization can entrench shortcuts
- **DER++/Replay**: Lower rigidity (ERI 0.3-0.5) - replay helps but doesn't eliminate shortcuts
- **From-Scratch Baseline**: Reference point (ERI components = 0 by definition)

### Red Flags in Your Results

Watch for these warning signs:

1. **Negative AD with high masked accuracy drop**: Fast learning but poor robustness
2. **High patched accuracy with low masked accuracy**: Over-reliance on shortcuts
3. **Positive SFR_rel**: More shortcut-dependent than baseline
4. **Low T1 accuracy**: Catastrophic forgetting masking rigidity effects

### Additional Diagnostics

When ERI indicates rigidity, validate with:

- **Representational drift** (CKA similarity to T1 representations)
- **Calibration analysis** under masking
- **Counterfactual patch placement** tests
- **Attention pattern analysis** (for ViT models)

### Example Interpretation

```
Method: EWC
AD: -12.3 epochs    (reaches τ=0.6 faster than scratch)
PD: -0.08           (slightly higher final accuracy)
SFR_rel: +0.15      (15% more dependent on shortcuts)
Overall ERI: 0.52   (moderate rigidity)

Interpretation: EWC shows the red-flag pattern. While it 
achieves good accuracy quickly, this is partly due to 
shortcut exploitation. The model is 15% more reliant on 
spurious patches than the scratch baseline.
```

---

## 🏗️ Architecture

### Project Structure

```
shortcut-attention/
├── datasets/              # Dataset implementations
│   ├── seq_cifar100_einstellung.py
│   └── transforms/        # Data augmentation
├── models/                # 70+ continual learning methods
│   ├── sgd.py            # Naive fine-tuning
│   ├── ewc_on.py         # Elastic Weight Consolidation
│   ├── derpp.py          # Dark Experience Replay++
│   └── ...               # Many more methods
├── backbone/              # Network architectures
│   ├── resnet.py
│   └── vit.py            # Vision Transformer
├── utils/                 # Utilities
│   ├── einstellung_evaluator.py    # Evaluation hooks
│   ├── einstellung_metrics.py      # ERI calculations
│   └── attention_visualization.py  # Attention analysis
├── eri_vis/               # Visualization system
│   ├── dataset.py         # Timeline data handling
│   ├── metrics_calculator.py       # ERI computation
│   ├── dynamics_plotter.py         # Plot generation
│   └── integration/       # Mammoth integration
├── tools/                 # CLI utilities
│   └── plot_eri.py        # Visualization tool
├── main.py                # Mammoth entry point
├── run_einstellung_experiment.py   # High-level runner
└── main.tex               # Research paper
```

### Key Components

#### Dataset Layer
- **SeqCIFAR100Einstellung**: Two-phase dataset with shortcut injection
- **MagentaPatchInjector**: On-the-fly patch insertion
- **MaskedEvaluationSets**: Patch removal for robustness testing

#### Evaluation Layer
- **EinstellungEvaluator**: Tracks metrics across tasks
- **EinstellungMetricsCalculator**: Computes AD, PD, SFR_rel
- **AttentionAnalyzer**: ViT-specific attention pattern analysis

#### Visualization Layer
- **ERITimelineDataset**: Timeline data structure
- **CorrectedERICalculator**: Paper-specification metrics
- **ERIDynamicsPlotter**: Three-panel figure generation
- **ERIHeatmapPlotter**: Robustness heatmaps

### Extension Points

#### Adding New Metrics

```python
# utils/einstellung_metrics.py
@dataclass
class EinstellungMetrics:
    # Add your new metric
    my_new_metric: Optional[float] = None
    
    def compute_eri_score(self, ...):
        # Update composite score calculation
        pass
```

#### Supporting New Architectures

```python
# utils/attention_visualization.py
class AttentionAnalyzer:
    def extract_attention(self, model, images):
        if isinstance(model.net, MyNewArchitecture):
            # Implement extraction for your architecture
            return self._extract_my_architecture_attention(model, images)
```

#### Custom Shortcuts

```python
# datasets/transforms/shortcuts.py
class MyShortcutInjector:
    def __call__(self, img, target):
        # Implement your shortcut
        return modified_img
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{gu2025eri,
  title={Diagnosing Shortcut-Induced Rigidity in Continual Learning: The Einstellung Rigidity Index (ERI)},
  author={Gu, Kai and Shi, Weishi},
  journal={IEEE Conference Proceedings},
  year={2025},
  institution={Department of Computer Science and Engineering, University of North Texas}
}
```

### Related Work

This project builds upon the [Mammoth continual learning framework](https://github.com/aimagelab/mammoth). If you use Mammoth, please also cite:

```bibtex
@article{boschini2022class,
  title={Class-Incremental Continual Learning into the eXtended DER-verse},
  author={Boschini, Matteo and Bonicelli, Lorenzo and Buzzega, Pietro and Porrello, Angelo and Calderara, Simone},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}

@inproceedings{buzzega2020dark,
  author = {Buzzega, Pietro and Boschini, Matteo and Porrello, Angelo and Abati, Davide and Calderara, Simone},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {15920--15930},
  publisher = {Curran Associates, Inc.},
  title = {Dark Experience for General Continual Learning: a Strong, Simple Baseline},
  volume = {33},
  year = {2020}
}
```

### Complete Attribution

This repository incorporates code and ideas from numerous open-source projects. See [`CITATION.cff`](CITATION.cff) for a comprehensive list of all dependencies and their attributions, including:

- **PyTorch** and **torchvision** (Meta AI)
- **OpenAI CLIP** (OpenAI)
- **Vision Transformer** (Google Research)
- **timm** (Ross Wightman)
- **Various CL methods**: iCaRL, BiC, L2P, DualPrompt, CoOp, DAP, ZSCL, and many more

We are deeply grateful to all these projects and their contributors.
---

## 🙏 Acknowledgments

We thank:
- **Weishi Shi** for supervision and guidance
- **Abdullah Al Forhad** for valuable discussions
- **Texas Academy of Mathematics and Science** for institutional support
- The **Mammoth team** (Matteo Boschini, Lorenzo Bonicelli, Pietro Buzzega, Angelo Porrello, Simone Calderara) for creating an excellent continual learning framework
- All the open-source contributors whose work made this research possible

Special recognition to the continual learning research community for advancing our understanding of lifelong learning systems.

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, new features, documentation improvements, or additional CL methods.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

Please use autopep8 with the following parameters:

```bash
autopep8 --aggressive --max-line-length=200 --ignore=E402 --in-place your_file.py
```

### Areas for Contribution

- **New shortcut types**: Different spurious correlation patterns
- **Additional metrics**: Novel rigidity measurements
- **Architecture support**: New backbone networks
- **Dataset extensions**: Different vision domains
- **Visualization improvements**: Enhanced plotting capabilities
- **Documentation**: Tutorials, examples, and guides

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Special Licenses

Some files in this repository are under different licenses:
- `backbone/vit.py` - Apache 2.0 License
- `models/l2p.py` - Apache 2.0 License
- `models/dualprompt.py` - Apache 2.0 License

See [NOTICE.md](NOTICE.md) for complete license information.

### Acknowledgment of Prior Work

This repository is built upon the [Mammoth continual learning framework](https://github.com/aimagelab/mammoth), which is also MIT licensed. We gratefully acknowledge their foundational work.

---

## 📞 Contact

- **Kai Gu** - [kaigu@my.unt.edu](mailto:kaigu@my.unt.edu)
- **Weishi Shi** - [weishi.shi@unt.edu](mailto:weishi.shi@unt.edu)

**Project Link**: [https://github.com/SamGu-NRX/shortcut-attention](https://github.com/SamGu-NRX/shortcut-attention)

---

## 🔗 Related Resources

- **Paper**: See `main.tex` in this repository
- **Mammoth Documentation**: [https://aimagelab.github.io/mammoth/](https://aimagelab.github.io/mammoth/)
- **Detailed Integration Guide**: See [EINSTELLUNG_README.md](EINSTELLUNG_README.md)
- **Implementation Plan**: See [EINSTELLUNG_INTEGRATION_PLAN.md](EINSTELLUNG_INTEGRATION_PLAN.md)
- **Reproducibility Notes**: See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

---

## 🌟 Star History

If you find this project useful for your research, please consider giving it a star ⭐!

---

<p align="center">
  <i>Understanding what neural networks learn—and why—is essential for building robust AI systems.</i>
</p>

<p align="center">
  Made with ❤️ by the UNT CSE Department
</p>
