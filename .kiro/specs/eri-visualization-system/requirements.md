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

### 1.2 Robustness Heatmap Visualization

**User Story:** As a researcher validating ERI robustness, I want to see AD(τ) values across different threshold ranges, so that I can demonstrate the stability of the metric.

#### Acceptance Criteria

1. WHEN generating robustness analysis THEN the system SHALL output fig_ad_tau_heatmap.pdf

2. WHEN creating heatmap THEN the system SHALL show AD(τ) vs Scratch_T2 across τ ∈ [0.50, 0.80], step 0.05

3. WHEN displaying values THEN the system SHALL use diverging colormap centered at 0.0; annotated cells with AD values

4. WHEN handling incomplete data THEN the system SHALL handle right-censored runs (no crossing) by leaving cells blank (NaN) and annotating legend with "no-cross" handling

### 1.3 Data Processing and Export

**User Story:** As a researcher with existing experiment logs, I want to convert my data into the required CSV format, so that I can generate the visualizations without re-running experiments.

#### Acceptance Criteria

1. WHEN validating CSV input THEN the system SHALL validate types and domains for required columns

2. WHEN supporting methods THEN the system SHALL support all existing Mammoth methods:

   - Baselines: Scratch_T2, Interleaved
   - CL methods: sgd, ewc_on, derpp, gmp (and other existing ones)

3. WHEN processing splits THEN the system SHALL support: T1_all, T2_shortcut_normal, T2_shortcut_masked, T2_nonshortcut_normal

4. WHEN converting data THEN the system SHALL provide converter from EinstellungEvaluator export (JSON/dict) → CSV

5. WHEN handling incomplete data THEN the system SHALL gracefully handle missing data (warn, skip panel if impossible)

### 1.4 Integration with Mammoth

**User Story:** As a researcher using the existing Mammoth ERI implementation, I want the visualization system to integrate seamlessly, so that I can generate plots from my current experiment pipeline.

#### Acceptance Criteria

1. WHEN running experiments THEN the system SHALL auto-log and export required metrics via EinstellungEvaluator hooks:

   - After each effective Phase 2 epoch: SC patched + SC masked accuracies
   - Export timeline to CSV via export_results

2. WHEN calculating metrics THEN the system SHALL use the exact metric definitions from EinstellungMetricsCalculator

3. WHEN integrating with datasets THEN the system SHALL work with SequentialCIFAR100Einstellung224 and its evaluation subsets

4. WHEN processing ViT models THEN the system SHALL optionally integrate ViT attention via EinstellungAttentionAnalyzer

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

## Non-Functional Requirements

### 2.1 Robustness and Error Handling

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

## Data Formats

### 3.1 CSV Schema (strict)

**Columns:**

- method: str (e.g., Scratch_T2, sgd, ewc_on, derpp, gmp, Interleaved)
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
- Add derpp and gmp configs; pass through to visualization unchanged

### 4.3 Metrics (additional informative probes)

Keep core ERI metrics; add optional:

- Calibration under masking (ECE) for SC patched vs masked
- Representation drift (CKA) pre/post Phase 2 on SC vs NSC (optional)

These must not block core visualization; integrate as extras.

### 4.4 Seeds and Budgets

- Default seeds: 5–10
- Normalize effective epochs across methods (replay aware)
- Fixed validation checkpoint selection (best Phase 2 val on T2)

## CLI Commands (Examples)

**Single run visualization:**

```bash
python tools/plot_eri.py --csv logs/eri_sc_metrics.csv \
  --outdir logs/figs --methods Scratch_T2 sgd ewc_on derpp gmp \
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
