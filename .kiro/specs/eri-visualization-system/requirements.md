# ERI Visualization System — Requirements (v1.1)

## 🚨 CRITICAL INTEGRATION REQUIREMENT

**MANDATORY**: This system MUST integrate with the existing Mammoth Einstellung experiment infrastructure in this repository:

- **Dataset**: Use `datasets/seq_cifar100_einstellung_224.py` (ViT-compatible with patch injection)
- **Evaluator**: Integrate with `utils/einstellung_evaluator.py` (plugin-based evaluation system)
- **Methods**: Support all existing Mammoth continual learning strategies (SGD, EWC, DER++, etc.)
- **Pipeline**: Hook into Mammoth's training pipeline, logging, and checkpoint management
- **Export Format**: Compatible with `EinstellungEvaluator.export_results()` JSON format
- **Experiment Runner**: Work with existing experiment orchestration and checkpoint management

**DO NOT** create parallel implementations. **DO** extend and integrate with existing proven infrastructure.

## Goal

- Deliver publication-ready ERI dynamic visualizations integrated with the Mammoth continual learning framework and usable standalone.
- Enhance experiments to be more persuasive and address reviewer points (1–3) directly in the codebase, while fully implementing visualizations (4).

## Scope

- Visualization and robustness tooling
- Data export and processing
- Integration with Mammoth (EinstellungEvaluator, metrics, datasets)
- Experiment enhancements: dataset variants, methods, sensitivity sweeps
- CLI, performance, and reliability

## Key Definitions

- **Effective epoch (epoch_eff)**: Phase 2 optimizer updates that consumed T2 samples divided by T2 train size; normalizes replay methods.
- **Splits**:
  - T1_all
  - T2_shortcut_normal (SC patched)
  - T2_shortcut_masked (SC masked)
  - T2_nonshortcut_normal (NSC)

## Functional Requirements

### 1.1 Dynamic ERI Visualizations

**User Story:** As a researcher analyzing ERI results, I want to see time-resolved visualizations of AD, PD, and SFR_rel metrics, so that I can intuitively understand the dynamic patterns of shortcut-induced rigidity.

#### Acceptance Criteria

1. WHEN processing CSV input THEN the system SHALL accept columns [method, seed, epoch_eff, split, acc]

2. WHEN generating dynamics figure THEN the system SHALL output fig_eri_dynamics.pdf with 3-panel layout:

   - Panel A: Patched accuracy trajectories on SC with 95% CI (seed-level), smoothing window w=3; vertical dashed lines for E_S(τ), E_CL(τ), and "AD=x.x" annotations per method vs Scratch_T2
   - Panel B: Running PD_t(e) = A_S(e) − A_CL(e) on SC (patched)
   - Panel C: Running SFR_rel(e) = Δ_CL(e) − Δ_S(e) with Δ_M(e) computed as Acc(M, SC_patched, e) − Acc(M, SC_masked, e)

3. WHEN creating visualizations THEN the system SHALL include legends, axis labels, titles and save high-quality PDFs

4. WHEN generating overall ERI figure THEN the system SHALL output fig_eri_overall.pdf showing composite ERI scores combining AD, PD, and SFR_rel into a single interpretable metric

### 1.2 Robustness Heatmap Visualization

**User Story:** As a researcher validating ERI robustness, I want to see AD(τ) values across different threshold ranges, so that I can demonstrate the stability of the metric.

#### Acceptance Criteria

1. WHEN generating robustness analysis THEN the system SHALL output fig_ad_tau_heatmap.pdf

2. WHEN creating heatmap THEN the system SHALL show AD(τ) vs Scratch_T2 across τ ∈ [0.50, 0.80], step 0.05

3. WHEN displaying values THEN the system SHALL use diverging colormap centered at 0.0; annotated cells with AD values

4. WHEN handling incomplete data THEN the system SHALL handle right-censored runs (no crossing) by leaving cells blank (NaN) and annotating legend with "no-cross" handling

5. 🚨 WHEN calculating AD, PD, and SFR_rel THEN the system SHALL use the EXACT formulations from the provided ERI specification:
   - AD = E_CL(τ) - E_S(τ) where E_M(τ) is first effective epoch where smoothed accuracy ≥ τ
   - PD = A*S_patch^* - A*CL_patch^* using final checkpoint accuracies selected by best Phase-2 validation
   - SFR_rel = Δ_CL - Δ_S where Δ_M = A_M_patch - A_M_mask on final checkpoints
   - Use macro-averaged accuracy over C_patch classes (equal weight per class, not frequency-weighted)
   - Apply smoothing with window w=3 and trailing moving average for AD threshold detection

### 1.2.1 🚨 CRITICAL: ERI Calculation Fixes

**User Story:** As a researcher analyzing ERI results, I want the AD, PD, and SFR_rel calculations to be mathematically correct according to the paper specification, so that the heatmaps and visualizations accurately reflect method performance.

#### Acceptance Criteria

1. WHEN calculating effective epochs THEN the system SHALL track Phase-2 sample consumption divided by |Phase-2 training set| for replay normalization

2. WHEN computing AD THEN the system SHALL:

   - Apply smoothing with trailing moving average window w=3: smoothed_A[e] = mean(A_M[max(1,e-w+1) .. e])
   - Find E_M(τ) = smallest effective epoch e where smoothed_A_M(e) ≥ τ
   - Compute AD = E_CL(τ) - E_S(τ) only for seeds where both methods reach τ
   - Mark as NaN and exclude from averaging when threshold never reached

3. WHEN computing PD THEN the system SHALL:

   - Use final checkpoint selected by best Phase-2 validation accuracy (same rule for all methods)
   - Compute PD = A*S_patch^* - A*CL_patch^* using final selected checkpoint accuracies
   - Use macro-averaged accuracy: compute per-class accuracy then average (not frequency-weighted)

4. WHEN computing SFR_rel THEN the system SHALL:

   - Use final checkpoint selected by best Phase-2 validation accuracy
   - Compute Δ_M = A_M_patch - A_M_mask for final checkpoint
   - Compute SFR_rel = Δ_CL - Δ_S using final checkpoint deltas
   - Ensure paired test sets (same underlying images for patch vs mask)

5. WHEN validating calculations THEN the system SHALL verify that DER++ shows better performance than scratch_t2 baseline in the provided data (einstellung_results/session_20250923-012304_seed42)

### 1.3 Overall ERI Metric Visualization

**User Story:** As a researcher presenting ERI results, I want a single composite visualization that combines AD, PD, and SFR_rel into an overall ERI metric, so that I can provide a clear summary of method rigidity.

#### Acceptance Criteria

1. WHEN computing overall ERI THEN the system SHALL combine the three facets using weighted formula:

   - ERI_overall = w_AD × AD_norm + w_PD × PD + w_SFR × SFR_rel
   - Default weights: w_AD = 0.4, w_PD = 0.4, w_SFR = 0.2
   - AD_norm = min(AD / 50.0, 1.0) to normalize to [0,1] range

2. WHEN generating overall ERI figure THEN the system SHALL output fig_eri_overall.pdf with:

   - Bar chart showing ERI_overall scores for each method
   - Error bars representing 95% confidence intervals across seeds
   - Color coding: green (low rigidity) to red (high rigidity)
   - Horizontal reference line at ERI_overall for Scratch_T2 baseline

3. WHEN displaying overall ERI THEN the system SHALL include:
   - Method names on x-axis sorted by ERI score (ascending)
   - ERI score values on y-axis with appropriate scale
   - Legend explaining ERI interpretation (higher = more rigid)
   - Annotation showing component breakdown for top/bottom methods

### 1.4 Data Processing and Export

**User Story:** As a researcher with existing experiment logs, I want to convert my data into the required CSV format, so that I can generate the visualizations without re-running experiments.

#### Acceptance Criteria

1. WHEN validating CSV input THEN the system SHALL validate types and domains for required columns

2. WHEN supporting methods THEN the system SHALL support all existing Mammoth methods plus adapted integrated methods:

   - Baselines: Scratch_T2, Interleaved
   - CL methods: sgd, ewc_on, derpp, gpm (and other existing ones)
   - Adapted methods: gpm_adapted, dgr_adapted, gpm_dgr_hybrid

3. WHEN processing splits THEN the system SHALL support: T1_all, T2_shortcut_normal, T2_shortcut_masked, T2_nonshortcut_normal

4. WHEN converting data THEN the system SHALL provide converter from EinstellungEvaluator export (JSON/dict) → CSV

5. WHEN handling incomplete data THEN the system SHALL gracefully handle missing data (warn, skip panel if impossible)

### 1.4 Integration with Mammoth ✅ COMPLETED

**User Story:** As a researcher using the existing Mammoth ERI implementation, I want the visualization system to integrate seamlessly, so that I can generate plots from my current experiment pipeline.

#### Acceptance Criteria

1. ✅ WHEN running experiments THEN the system SHALL auto-log and export required metrics via EinstellungEvaluator hooks:

   - After each effective Phase 2 epoch: SC patched + SC masked accuracies
   - Export timeline to CSV via export_results

2. ✅ WHEN calculating metrics THEN the system SHALL use the exact metric definitions from EinstellungMetricsCalculator

3. ✅ WHEN integrating with datasets THEN the system SHALL work with SequentialCIFAR100Einstellung224 and its evaluation subsets

4. ✅ WHEN processing ViT models THEN the system SHALL optionally integrate ViT attention via EinstellungAttentionAnalyzer

**IMPLEMENTATION STATUS:** The end-to-end pipeline has been successfully implemented by extending the existing `run_einstellung_experiment.py` runner. The integration works with existing Mammoth infrastructure including:

- ✅ EinstellungEvaluator hooks are registered and active
- ✅ All required evaluation subsets (T1_all, T2_shortcut_normal, T2_shortcut_masked, T2_nonshortcut_normal) are evaluated
- ✅ AttentionAnalyzer is initialized for ViT models
- ✅ Checkpoint management and experiment orchestration work seamlessly

### 1.5 Documentation Updates for Reviewer Points

**User Story:** As a paper reviewer, I want to see acknowledgment of limitations and future work plans, so that I understand the scope and planned extensions of the research.

#### Acceptance Criteria

1. WHEN addressing generalizability THEN the system SHALL document CIFAR-100 limitation; add ImageNet-100 and text CL (SST-2 → IMDB) extensions in README and comments

2. WHEN addressing robustness THEN the system SHALL document sensitivity protocol; reference AD(τ) heatmap

3. WHEN addressing baselines THEN the system SHALL commit to DER++ and GPM; ensure method-agnostic visualization

### 1.6 CLI and Usability

**User Story:** As a researcher generating visualizations, I want a simple command-line interface, so that I can easily create plots with different configurations.

#### Acceptance Criteria

1. WHEN running visualization script THEN the system SHALL provide CLI with args: --csv, --outdir, --methods, --tau, --smooth, --tau_grid

2. WHEN configuring appearance THEN the system SHALL allow custom color schemes with sensible defaults

3. WHEN managing outputs THEN the system SHALL auto-create output directories

4. WHEN processing data THEN the system SHALL provide clear progress logging and helpful errors

5. WHEN running batch mode THEN the system SHALL process multiple CSVs; produce comparative summaries

### 1.7 Custom Method Integration

**User Story:** As a researcher implementing novel continual learning methods, I want to easily integrate my custom methods with the ERI visualization system, so that I can evaluate their performance against established baselines.

#### Acceptance Criteria

1. WHEN implementing custom methods THEN the system SHALL support seamless integration with Mammoth's ContinualModel framework

2. WHEN adding new methods THEN the system SHALL automatically include them in visualization outputs without code changes

3. WHEN running experiments THEN the system SHALL support custom method configurations through YAML files

4. WHEN evaluating custom methods THEN the system SHALL apply the same ERI metrics (AD, PD_t, SFR_rel) consistently

5. WHEN generating visualizations THEN the system SHALL automatically assign colors and include custom methods in legends and heatmaps

### 1.8 Extended Method Coverage

**User Story:** As a researcher conducting comprehensive ERI analysis, I want to evaluate additional continual learning methods beyond the basic baselines, so that I can provide more thorough empirical coverage.

#### Acceptance Criteria

1. WHEN running method comparisons THEN the system SHALL support advanced replay methods (e.g., enhanced DER++ variants)

2. WHEN evaluating regularization approaches THEN the system SHALL support advanced EWC variants and other regularization techniques

3. WHEN conducting analysis THEN the system SHALL maintain consistent evaluation protocols across all methods

4. WHEN generating reports THEN the system SHALL provide method-specific insights and recommendations

5. WHEN scaling experiments THEN the system SHALL handle increased method counts efficiently in batch processing

### 1.9 GPM (Gradient Projection Memory) Integration

**User Story:** As a researcher investigating gradient-based continual learning approaches, I want to integrate the existing GPM implementation with the ERI visualization system, so that I can evaluate its effectiveness against shortcut learning while maintaining compatibility with existing methods.

#### Acceptance Criteria

1. WHEN integrating GPM THEN the system SHALL adapt the existing GPM implementation from `/GPM` directory to work with Mammoth's ContinualModel framework

2. WHEN adapting GPM THEN the system SHALL preserve the original GPM functionality including SVD-based subspace extraction and gradient projection mechanisms

3. WHEN configuring GPM THEN the system SHALL maintain the original energy threshold parameters (default 0.90-0.99) and sample limits (200-2000 samples)

4. WHEN projecting gradients THEN the system SHALL preserve the original orthogonal gradient projection: g ← g - U(U^T g)

5. WHEN managing memory THEN the system SHALL maintain the original basis growth handling through QR decomposition and pruning

6. WHEN integrating with ERI THEN the system SHALL work seamlessly with existing EinstellungEvaluator and produce standard ERI metrics (AD, PD_t, SFR_rel)

7. WHEN configuring GPM THEN the system SHALL support YAML-based configuration using parameters from the original GPM implementation

### 1.10 Deep Generative Replay (DGR) Integration

**User Story:** As a researcher exploring generative approaches to continual learning, I want to integrate the existing DGR implementation, so that I can evaluate its effectiveness in preventing catastrophic forgetting while maintaining ERI compatibility.

#### Acceptance Criteria

1. WHEN integrating DGR THEN the system SHALL adapt the existing DGR implementation from `/DGR_wrapper` directory to work with Mammoth's ContinualModel framework

2. WHEN adapting DGR THEN the system SHALL preserve the original VAE-based generative replay functionality including encoder-decoder architecture

3. WHEN training VAE THEN the system SHALL maintain the original VAE training loop with reconstruction and KL divergence losses

4. WHEN generating replay data THEN the system SHALL preserve the original replay generation mechanisms and sample quality

5. WHEN updating generative memory THEN the system SHALL maintain the original VAE training and memory update strategies

6. WHEN handling feature space THEN the system SHALL preserve the original feature space handling and consistency mechanisms

7. WHEN integrating with ERI THEN the system SHALL work with existing evaluation pipeline and produce consistent ERI metrics using the adapted DGR approach

### 1.11 Combined GPM + DGR Hybrid Methods

**User Story:** As a researcher investigating hybrid continual learning approaches, I want to combine the adapted GPM and DGR techniques, so that I can leverage both gradient projection and generative replay mechanisms for enhanced performance.

#### Acceptance Criteria

1. WHEN combining methods THEN the system SHALL support simultaneous adapted GPM gradient projection and adapted DGR generative replay in a single training loop

2. WHEN managing memory updates THEN the system SHALL coordinate both DGR VAE training and GPM basis updates after each task

3. WHEN training hybrid methods THEN the system SHALL apply GPM gradient projection after DGR replay-augmented loss computation but before optimizer steps

4. WHEN configuring hybrid methods THEN the system SHALL provide YAML configurations with both adapted GPM and DGR parameters

5. WHEN handling feature space consistency THEN the system SHALL coordinate the original GPM and DGR feature space handling mechanisms

6. WHEN evaluating hybrid methods THEN the system SHALL maintain full ERI evaluation compatibility and produce comparative visualizations

7. WHEN scaling hybrid approaches THEN the system SHALL manage computational overhead efficiently using the original methods' optimization strategies

## Integration Implementation Requirements

### 2.1 GPM Integration Requirements

**User Story:** As a developer integrating GPM, I want clear technical specifications for adapting the existing implementation, so that I can create a robust and efficient integration.

#### Acceptance Criteria

1. WHEN integrating GPM THEN the system SHALL create `models/gpm_mammoth_adapter.py` that adapts the existing GPM implementation:

   - Extract core GPM functionality from `/GPM/main_cifar100.py` and related files
   - Preserve original model registration and layer selection mechanisms
   - Maintain original energy threshold configuration (default 0.90, range 0.80-0.99)
   - Ensure GPU/CPU compatibility from original implementation

2. WHEN adapting activation collection THEN the system SHALL:

   - Preserve original forward hook registration mechanisms
   - Maintain original global average pooling for convolutional layers
   - Keep original configurable maximum batch limits (default 200, range 50-2000)
   - Ensure proper hook cleanup to prevent memory leaks

3. WHEN adapting SVD computation THEN the system SHALL:

   - Preserve original activation centering and SVD computation
   - Maintain original PyTorch SVD usage for numerical stability
   - Keep original basis size determination using cumulative energy threshold
   - Handle edge cases as in original implementation

4. WHEN adapting memory updates THEN the system SHALL:

   - Preserve original basis concatenation and QR decomposition mechanisms
   - Maintain original orthogonality preservation
   - Keep original pruning of near-zero components (threshold 1e-6)
   - Ensure proper device placement as in original implementation

5. WHEN adapting gradient projection THEN the system SHALL:
   - Preserve original projection formula: g ← g - U(U^T g)
   - Maintain original parameter reshaping for convolutional layers
   - Keep original gradient device placement consistency
   - Process parameters as in original implementation

### 2.2 DGR Integration Requirements

**User Story:** As a developer integrating DGR, I want clear specifications for adapting the existing implementation, so that I can provide effective replay mechanisms.

#### Acceptance Criteria

1. WHEN integrating DGR THEN the system SHALL create `models/dgr_mammoth_adapter.py` that adapts the existing DGR implementation:

   - Extract VAE and replay functionality from `/DGR_wrapper/models/vae.py`
   - Preserve original VAE architecture and training mechanisms
   - Maintain original replay generation and sample quality
   - Ensure device-aware tensor operations for GPU/CPU compatibility

2. WHEN adapting VAE training THEN the system SHALL:

   - Preserve original encoder-decoder architecture from existing implementation
   - Maintain original reconstruction and KL divergence loss computation
   - Keep original training loop and optimization strategies
   - Ensure proper convergence and sample quality

3. WHEN adapting replay generation THEN the system SHALL:

   - Preserve original sample generation mechanisms
   - Maintain original class-conditional generation if present
   - Keep original batch composition and replay ratios
   - Ensure consistent sample quality and diversity

4. WHEN adapting memory management THEN the system SHALL:

   - Preserve original VAE parameter storage and updates
   - Maintain original memory efficiency strategies
   - Keep original device placement for VAE components
   - Ensure proper cleanup and resource management

5. WHEN adapting feature space handling THEN the system SHALL:
   - Preserve original feature space consistency mechanisms
   - Maintain original handling of feature drift
   - Keep original recommendations for backbone freezing
   - Document usage patterns from original implementation

### 2.3 Integration and Configuration Requirements

**User Story:** As a researcher configuring experiments, I want seamless integration with existing Mammoth infrastructure, so that I can easily run comparative studies.

#### Acceptance Criteria

1. WHEN creating method configurations THEN the system SHALL provide YAML files:

   - `models/config/gpm_adapted.yaml` with parameters from original GPM implementation
   - `models/config/dgr_adapted.yaml` with parameters from original DGR implementation
   - `models/config/gpm_dgr_hybrid.yaml` combining both approaches
   - Parameter documentation and recommended values from original implementations

2. WHEN integrating with Mammoth THEN the system SHALL:

   - Extend existing model registry to include adapted methods
   - Maintain compatibility with existing `get_model()` function
   - Support all existing Mammoth training hooks and callbacks
   - Preserve existing checkpoint and logging functionality

3. WHEN handling method combinations THEN the system SHALL:

   - Provide clear execution order: DGR replay generation → loss computation → GPM projection → optimizer step
   - Coordinate memory updates for both GPM bases and DGR VAE training
   - Handle potential conflicts between different memory management approaches
   - Include comprehensive error handling and validation

4. WHEN documenting adapted methods THEN the system SHALL:
   - Reference original GPM and DGR papers and implementations
   - Include adaptation notes and integration-specific considerations
   - Document computational complexity and memory requirements
   - Provide troubleshooting guidance for integration issues

## Non-Functional Requirements

### 2.4 Robustness and Error Handling

**User Story:** As a researcher working with diverse experimental setups, I want the visualization system to handle edge cases robustly, so that I can generate reliable results across different configurations.

#### Acceptance Criteria

1. WHEN encountering missing epochs THEN the system SHALL interpolate/align where safe; warn on gaps > 3 epochs

2. WHEN processing incomplete runs THEN the system SHALL mark AD as NaN, annotate figure legend

3. WHEN handling different epoch ranges THEN the system SHALL align timelines for pairwise comparisons

4. WHEN computing statistics THEN the system SHALL compute 95% CI via t-distribution; document N seeds used

5. WHEN processing different models THEN the system SHALL handle ViT vs CNN evaluation frequencies; handle alignment gracefully

6. WHEN encountering corrupted data THEN the system SHALL provide detailed diagnostics with row/column context

### 2.2 Performance and Scalability

**User Story:** As a researcher processing large-scale experiments, I want the visualization system to be efficient and scalable, so that I can handle comprehensive multi-seed, multi-method studies.

#### Acceptance Criteria

1. WHEN processing large datasets THEN the system SHALL use memory efficient structures; no full in-memory duplication; chunk reading allowed

2. WHEN generating multiple plots THEN the system SHALL support parallel processing for batch plotting (optional flag)

3. WHEN creating figures THEN the system SHALL provide configurable DPI and output formats (PDF mandatory; PNG/SVG optional)

4. WHEN smoothing data THEN the system SHALL use O(N) per curve; avoid quadratic ops

5. WHEN caching results THEN the system SHALL provide optional caching of aggregated curves (pickle) with checksum validation

### 2.3 Reproducibility

**User Story:** As a researcher ensuring reproducible results, I want deterministic outputs and comprehensive metadata, so that results can be validated and reproduced.

#### Acceptance Criteria

1. WHEN exporting CSV THEN the system SHALL create deterministic CSV exports (sorted by method, seed, epoch)

2. WHEN generating metadata THEN the system SHALL record meta in a sidecar JSON (methods, seeds, τ, smoothing, time)

3. WHEN versioning THEN the system SHALL version the visualization script; include git SHA if available

## New Method Implementation Requirements

### 2.4 GPM Implementation Requirements

**User Story:** As a developer implementing GPM, I want clear technical specifications, so that I can create a robust and efficient implementation.

#### Acceptance Criteria

1. WHEN implementing GPM adapter THEN the system SHALL create `models/gpm_adapter.py` with the GPM class supporting:

   - Model registration with configurable layer names from `dict(model.named_modules())`
   - Energy threshold configuration (default 0.90, range 0.80-0.99)
   - Device management for GPU/CPU compatibility

2. WHEN collecting activations THEN the system SHALL:

   - Register forward hooks on specified layers during collection phase
   - Apply global average pooling for convolutional layers to reduce dimensionality
   - Support configurable maximum batch limits (default 200, range 50-2000)
   - Clean up hooks after collection to prevent memory leaks

3. WHEN computing SVD bases THEN the system SHALL:

   - Center activations by subtracting mean before SVD
   - Use PyTorch's `torch.linalg.svd` for numerical stability
   - Determine basis size using cumulative energy threshold
   - Handle edge cases where energy threshold is never reached

4. WHEN updating memory THEN the system SHALL:

   - Concatenate new bases with existing bases per layer
   - Apply QR decomposition to maintain orthogonality
   - Prune near-zero components (threshold 1e-6) to prevent basis explosion
   - Store bases on appropriate device for gradient projection

5. WHEN projecting gradients THEN the system SHALL:
   - Apply projection formula: g ← g - U(U^T g) for each registered layer
   - Handle parameter reshaping for convolutional layers
   - Maintain gradient device placement consistency
   - Process only parameters with non-None gradients

### 2.5 Generative Replay Implementation Requirements

**User Story:** As a developer implementing generative replay, I want clear specifications for both simple and advanced variants, so that I can provide effective replay mechanisms.

#### Acceptance Criteria

1. WHEN implementing ClassGaussianMemory THEN the system SHALL create `models/gen_replay.py` with:

   - Per-class mean and standard deviation storage in feature space
   - Configurable minimum standard deviation (default 1e-4) to prevent numerical issues
   - Device-aware tensor operations for GPU/CPU compatibility
   - Efficient sampling with balanced class representation

2. WHEN fitting class distributions THEN the system SHALL:

   - Compute per-class statistics from backbone features (not raw inputs)
   - Handle classes with insufficient samples gracefully
   - Update statistics incrementally or replace based on configuration
   - Store statistics on CPU to minimize GPU memory usage

3. WHEN sampling replay data THEN the system SHALL:

   - Generate balanced samples across specified classes
   - Apply Gaussian sampling with stored means and standard deviations
   - Return both synthetic features and corresponding labels
   - Support configurable batch sizes and class selection

4. WHEN implementing VAE replay (optional) THEN the system SHALL:

   - Provide conditional VAE architecture suitable for feature space
   - Support latent dimensions in range 32-128
   - Include training loop for VAE fitting after each task
   - Handle reconstruction and KL divergence losses appropriately

5. WHEN managing feature space consistency THEN the system SHALL:
   - Provide backbone freezing option to maintain feature space stability
   - Support memory refitting when backbone continues training
   - Include warnings about feature drift when backbone is trainable
   - Document recommended usage patterns for different scenarios

### 2.6 Integration and Configuration Requirements

**User Story:** As a researcher configuring experiments, I want seamless integration with existing Mammoth infrastructure, so that I can easily run comparative studies.

#### Acceptance Criteria

1. WHEN creating method configurations THEN the system SHALL provide YAML files:

   - `models/config/gpm.yaml` with layer selection and energy threshold parameters
   - `models/config/class_gaussian_replay.yaml` with replay ratio and memory parameters
   - `models/config/gpm_gaussian_hybrid.yaml` combining both approaches
   - Parameter documentation and recommended values for each configuration

2. WHEN integrating with Mammoth THEN the system SHALL:

   - Extend existing model registry to include new methods
   - Maintain compatibility with existing `get_model()` function
   - Support all existing Mammoth training hooks and callbacks
   - Preserve existing checkpoint and logging functionality

3. WHEN handling method combinations THEN the system SHALL:

   - Provide clear execution order: replay augmentation → loss computation → GPM projection → optimizer step
   - Coordinate memory updates for both GPM bases and generative models
   - Handle potential conflicts between different memory management approaches
   - Include comprehensive error handling and validation

4. WHEN documenting new methods THEN the system SHALL:
   - Provide implementation notes in method docstrings
   - Include hyperparameter recommendations based on literature
   - Document computational complexity and memory requirements
   - Reference original papers and implementations

## Data Formats

### 3.1 CSV Schema (strict)

**Columns:**

- method: str (e.g., Scratch_T2, sgd, ewc_on, derpp, gpm, Interleaved)
- seed: int
- epoch_eff: float (monotonic per method×seed)
- split: str ∈ {T1_all, T2_shortcut_normal, T2_shortcut_masked, T2_nonshortcut_normal}
- acc: float ∈ [0, 1]

**Example:**

```csv
method,seed,epoch_eff,split,acc
Scratch_T2,42,0.0,T2_shortcut_normal,0.10
Scratch_T2,42,0.0,T2_shortcut_masked,0.05
sgd,42,0.0,T2_shortcut_normal,0.20
```

### 3.2 Evaluator Export (JSON/dict)

Must contain timeline_data with per-epoch subset accuracies for T2_shortcut_normal and T2_shortcut_masked at Phase 2 epochs.

## Experiment Enhancements (Addressing Reviewer Points)

### 4.1 Datasets

**CIFAR-100 (224) Einstellung:** Current implementation with variable shortcut salience:

- Patch size ∈ {2, 4, 6, 8}, color = magenta or high-contrast pair
- Patch location: fixed top-left (baseline) and random among 4 corners
- Injection ratio ∈ {0.5, 1.0} (fraction of SC images patched)

**ImageNet-100 Einstellung (optional module):**

- Resize to 224; same shortcut injection variants

**Text CL pilot (optional module):**

- SST-2 → IMDB with lexical cue masking via synonym neutralization

### 4.2 Methods

Ensure support for:

- sgd (naive), ewc_on, replay variants already present
- Add derpp and gpm configs; pass through to visualization unchanged
- **New methods**: gpm, class_gaussian_replay, vae_replay (optional)
- **Hybrid methods**: gpm_gaussian_hybrid, gpm_vae_hybrid (optional)
- All methods must integrate seamlessly with existing ERI evaluation pipeline

### 4.3 Metrics (additional informative probes)

Keep core ERI metrics; add optional:

- Calibration under masking (ECE) for SC patched vs masked
- Representation drift (CKA) pre/post Phase 2 on SC vs NSC (optional)

These must not block core visualization; integrate as extras.

### 4.4 Seeds and Budgets

- Default seeds: 5–10
- Normalize effective epochs across methods (replay aware)
- Fixed validation checkpoint selection (best Phase 2 val on T2)

## New Method Usage Guidelines

### GPM Hyperparameters and Tips

**Recommended Settings:**

- Energy threshold: 0.90-0.99 (0.95 default)
- Activation samples: 200-2000 per layer (more is better but slower)
- Layer selection: Use global-average-pooled activations for conv layers
- Memory management: Compress bases with QR/SVD if growth becomes excessive

**Integration Notes:**

- Use with existing Mammoth methods by adding GPM projection after loss.backward()
- Compatible with all existing datasets and evaluation protocols
- Computational cost scales with basis size and number of layers

### Generative Replay Hyperparameters and Tips

**Class-Conditional Gaussian (Recommended for simplicity):**

- Replay ratio: 1:1 with real samples (or lower if compute limited)
- Minimum std: 1e-4 to prevent numerical issues
- Memory recomputation: After each task if backbone is trainable

**VAE-based Replay (Advanced option):**

- Latent dimension: 32-128
- Training epochs: 2-5 per task on collected features
- Conditional architecture: Use class embeddings for conditioning

**Feature Space Considerations:**

- Prefer backbone freezing for stable feature space
- If backbone trainable, recompute memory frequently
- Monitor feature drift through validation accuracy

### Hybrid Method Combinations

**GPM + Generative Replay Recipe:**

1. After each task: fit generative memory and update GPM bases
2. During training: sample replay data → compute loss → apply GPM projection → optimizer step
3. Coordinate memory updates to maintain consistency
4. Use same evaluation protocol as individual methods

## CLI Commands (Examples)

**Single run visualization:**

```bash
python tools/plot_eri.py --csv logs/eri_sc_metrics.csv \
  --outdir logs/figs --methods Scratch_T2 sgd ewc_on derpp gpm \
  --tau 0.6 --smooth 3
```

**Batch mode (multiple runs):**

```bash
python tools/plot_eri.py --csv logs/run_*.csv \
  --outdir logs/figs_batch --methods Scratch_T2 sgd ewc_on \
  --tau 0.6 --smooth 3 --tau_grid 0.5 0.55 0.6 0.65 0.7 0.75 0.8
```

## Acceptance Criteria (Summary)

- Generates fig_eri_dynamics.pdf (A/B/C panels) and fig_ad_tau_heatmap.pdf
- Uses CI bands (95%) with smoothing w=3, correct labels/legends
- Correct AD markers and annotations; heatmap centered at 0.0
- CSV schema validated; missing/censored data handled with clear warnings
- Integrates with Mammoth's EinstellungEvaluator to auto-export CSV
- Documentation updated to reflect generalizability, robustness, baselines
- Performance: Each figure renders under 30s with ≤10 seeds and ≤1000 epochs
- Outputs reproducible, publication-ready PDFs (≤5 MB)
