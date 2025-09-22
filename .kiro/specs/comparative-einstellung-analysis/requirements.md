# Comparative Einstellung Analysis System — Requirements

## Introduction

This specification addresses the critical gaps in the current Einstellung experiment system that prevent comprehensive comparative analysis. The current system runs individual experiments but fails to generate comparative visualizations, baseline comparisons, and cross-method metrics that are essential for meaningful research analysis.

## Current Issues Identified

From the terminal output analysis:

- No scratch baseline (Scratch_T2) is being generated or compared against
- Individual dynamics plots are generated per method, but no comparative dynamics across methods
- Performance deficit (PD_t) and SFR_rel calculations fail due to missing baseline data
- Heatmaps and comprehensive visualizations are not generated
- T1 and T2 task boundaries are unclear in outputs

## Implementation Constraints

**CRITICAL**: All implementations MUST be minimal and build on existing Mammoth infrastructure:

### Existing Infrastructure to Reuse:

- ✅ **ERI Visualization System**: Complete `eri_vis/` package with data loading, processing, and plotting
- ✅ **Experiment Runner**: `run_einstellung_experiment.py` with comparative mode (`--comparative` flag)
- ✅ **Model Registry**: `models/__init__.py` with `get_model()` function and automatic model discovery
- ✅ **Joint Training**: `models/joint.py` provides template for baseline method implementation
- ✅ **EinstellungEvaluator**: Complete evaluation system with timeline tracking and CSV export
- ✅ **Visualization Integration**: `generate_eri_visualizations()` function already calls ERI plotting system

### Implementation Principles:

- **NO CODE DUPLICATION**: Extend existing functions, don't recreate them
- **MINIMAL NEW FILES**: Only create essential baseline model classes
- **REUSE EXISTING PATTERNS**: Follow existing model implementation patterns exactly
- **LEVERAGE EXISTING SYSTEMS**: Use existing CSV formats, visualization pipeline, and experiment orchestration

## Requirements

### Requirement 1: Baseline Method Integration

**User Story:** As a researcher conducting Einstellung analysis, I want to include proper baseline methods (Scratch_T2, Interleaved) in my comparative experiments, so that I can measure the relative performance of continual learning methods against optimal and naive baselines.

#### Acceptance Criteria

1. WHEN implementing Scratch_T2 THEN the system SHALL create a baseline method that trains only on Task 2 data using the same backbone and training configuration as continual learning methods
2. WHEN implementing Interleaved THEN the system SHALL create a baseline method that trains on mixed Task 1 and Task 2 data simultaneously
3. WHEN registering baseline methods THEN the system SHALL integrate with existing Mammoth model registry for automatic discovery
4. WHEN evaluating baselines THEN the system SHALL apply the same Einstellung evaluation protocol (T1_all, T2_shortcut_normal, T2_shortcut_masked, T2_nonshortcut_normal)
5. WHEN storing baseline results THEN the system SHALL save results in the same CSV format as continual learning methods for comparative analysis

### Requirement 2: Comparative Visualization Generation

**User Story:** As a researcher analyzing multiple continual learning methods, I want to see comparative dynamics plots showing all methods on the same figure, so that I can directly compare their performance trajectories and identify relative strengths and weaknesses.

#### Acceptance Criteria

1. WHEN generating comparative visualizations THEN the system SHALL create a single dynamics plot showing all methods' accuracy trajectories on the same axes
2. WHEN plotting comparative dynamics THEN the system SHALL use distinct colors and line styles for each method with clear legends
3. WHEN showing performance deficits THEN the system SHALL calculate PD_t = A_Scratch_T2(e) - A_Method(e) for each continual learning method relative to Scratch_T2 baseline
4. WHEN showing shortcut forgetting rates THEN the system SHALL calculate SFR_rel = (A_Method_patched(e) - A_Method_masked(e)) - (A_Scratch_T2_patched(e) - A_Scratch_T2_masked(e))
5. WHEN generating heatmaps THEN the system SHALL create robustness heatmaps showing adaptation delay across different threshold values for all methods

### Requirement 3: Cross-Method Metrics Calculation

**User Story:** As a researcher evaluating continual learning effectiveness, I want to calculate performance deficits and shortcut forgetting rates relative to appropriate baselines, so that I can quantify the impact of catastrophic forgetting and shortcut reliance across different methods.

#### Acceptance Criteria

1. WHEN calculating performance deficits THEN the system SHALL compute PD_t for each method using Scratch_T2 as the reference baseline
2. WHEN calculating shortcut forgetting rates THEN the system SHALL compute SFR_rel for each method relative to Scratch_T2 baseline performance
3. WHEN handling missing baseline data THEN the system SHALL provide clear error messages indicating which baseline methods need to be run
4. WHEN computing adaptation delays THEN the system SHALL calculate AD values for all methods using consistent threshold crossing detection
5. WHEN aggregating metrics THEN the system SHALL provide summary statistics (mean, std, confidence intervals) across multiple seeds for each method

### Requirement 4: Comprehensive Experiment Orchestration

**User Story:** As a researcher running comparative studies, I want an orchestration system that automatically runs all required methods including baselines and generates complete comparative analysis, so that I can obtain comprehensive results without manual coordination of individual experiments.

#### Acceptance Criteria

1. WHEN running comparative analysis THEN the system SHALL automatically determine which baseline methods need to be run based on the selected continual learning methods
2. WHEN orchestrating experiments THEN the system SHALL run baseline methods first, followed by continual learning methods, ensuring proper dependency management
3. WHEN managing experiment state THEN the system SHALL track completion status of each method and resume from partial completions when possible
4. WHEN generating final outputs THEN the system SHALL produce a comprehensive report including individual method results, comparative visualizations, and summary statistics
5. WHEN handling experiment failures THEN the system SHALL provide detailed diagnostics and allow selective re-running of failed components

### Requirement 5: Enhanced Reporting and Output Generation

**User Story:** As a researcher presenting Einstellung analysis results, I want comprehensive reports that include both individual method performance and comparative analysis with clear visualizations, so that I can effectively communicate findings and support research conclusions.

#### Acceptance Criteria

1. WHEN generating reports THEN the system SHALL create a master comparative report showing all methods' performance in standardized format
2. WHEN creating visualizations THEN the system SHALL generate publication-ready figures including comparative dynamics plots, heatmaps, and summary charts
3. WHEN presenting results THEN the system SHALL clearly indicate task boundaries (T1 vs T2) and evaluation phases in all outputs
4. WHEN documenting experiments THEN the system SHALL include experimental metadata, method configurations, and statistical significance tests in reports
5. WHEN organizing outputs THEN the system SHALL create a structured directory hierarchy with clear naming conventions for easy navigation and analysis

### Requirement 6: Statistical Analysis and Significance Testing

**User Story:** As a researcher validating experimental results, I want statistical analysis tools that assess the significance of performance differences between methods, so that I can make robust claims about method effectiveness and identify meaningful performance gaps.

#### Acceptance Criteria

1. WHEN running `--comparative` mode THEN the system SHALL extend the existing configs list in `run_comparative_experiment()` to include ('scratch_t2', 'resnet18') and ('interleaved', 'resnet18')
2. WHEN experiments complete THEN the system SHALL call a new `aggregate_comparative_results()` function to merge all CSV files
3. WHEN generating final visualizations THEN the system SHALL pass the merged dataset to existing `generate_eri_visualizations()` function
4. WHEN handling failures THEN the system SHALL use existing checkpoint and error handling infrastructure
5. WHEN reporting results THEN the system SHALL extend existing summary table to show comparative metrics (PD_t, SFR_rel, AD) when baseline data is available
