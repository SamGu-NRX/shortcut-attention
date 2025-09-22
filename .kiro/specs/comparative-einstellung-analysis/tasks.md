# Comparative Einstellung Analysis System — Implementation Plan

## Implementation Overview

This implementation plan creates comprehensive comparative Einstellung analysis through minimal, targeted extensions to existing Mammoth infrastructure. Each task builds incrementally and leverages existing systems to avoid code duplication.

## Task Organization

Tasks are organized in dependency order, with each task building on previous work and existing infrastructure.

---

## Phase 1: Baseline Method Implementation

- [x] **1. Implement Scratch_T2 Baseline Method**

  - Create `models/scratch_t2.py` extending ContinualModel following existing `models/joint.py` pattern
  - Implement Task 2 only training: skip Task 1, collect Task 2 data, train at end of Task 2
  - Use existing backbone, loss, transform, dataset infrastructure without modification
  - Follow existing model naming convention with NAME = 'scratch_t2' for automatic registry discovery
  - Ensure compatibility with existing EinstellungEvaluator evaluation protocol
  - _Requirements: 1.1, 1.4_

- [x] **2. Implement Interleaved Baseline Method**

  - Create `models/interleaved.py` extending ContinualModel following existing `models/joint.py` pattern
  - Implement mixed task training: collect data from both tasks, train on combined dataset
  - Reuse existing joint training loop patterns and data combination logic
  - Follow existing model naming convention with NAME = 'interleaved' for automatic registry discovery
  - Ensure seamless integration with existing Mammoth training pipeline and evaluation system
  - _Requirements: 1.1, 1.4_

- [x] **3. Validate Baseline Method Integration**
  - Test that baseline methods are automatically discovered by existing `models/__init__.py` registry
  - Verify baseline methods work with existing `get_model()` function and argument parsing
  - Confirm baseline methods integrate with existing EinstellungEvaluator without modifications
  - Test baseline methods produce valid CSV output in existing format for ERI visualization
  - Validate baseline methods work with existing checkpoint management and experiment orchestration
  - _Requirements: 1.1, 1.4, 1.5_

---

## Phase 2: Data Aggregation System

- [x] **4. Implement Comparative Results Aggregation**

  - Add `aggregate_comparative_results()` function to existing `run_einstellung_experiment.py`
  - Use existing `eri_vis.data_loader.ERIDataLoader` for CSV loading and validation
  - Implement CSV merging using existing dataset merge functionality from ERI system
  - Export aggregated results in existing CSV format compatible with ERI visualization pipeline
  - Add error handling for missing CSV files and validation failures using existing patterns
  - _Requirements: 2.2, 2.3_

- [x] **5. Implement CSV File Discovery and Validation**

  - Add `find_csv_file()` helper function to locate ERI CSV files in experiment output directories
  - Use existing file path patterns and naming conventions from current experiment system
  - Implement validation to ensure CSV files contain required columns and valid data
  - Add graceful handling for missing or corrupted CSV files with clear error messages
  - Integrate with existing logging system for progress tracking and error reporting
  - _Requirements: 2.2, 2.5_

- [x] **6. Test Data Aggregation Pipeline**
  - Create unit tests for CSV aggregation using existing test patterns and fixtures
  - Test aggregation with multiple methods including baseline methods
  - Verify aggregated CSV maintains existing format and passes ERI validation
  - Test error handling for missing files, invalid data, and edge cases
  - Validate aggregated data produces correct comparative visualizations
  - _Requirements: 2.2, 2.3, 2.5_

---

## Phase 3: Experiment Orchestration Enhancement

- [x] **7. Extend Comparative Experiment Runner**

  - Modify existing `run_comparative_experiment()` in `run_einstellung_experiment.py`
  - Add baseline method configs to existing configs list: ('scratch_t2', 'resnet18'), ('interleaved', 'resnet18')
  - Reorder configs to run baseline methods first for proper dependency management
  - Use existing experiment runner infrastructure without modification for individual experiments
  - Maintain existing checkpoint management, error handling, and progress reporting
  - _Requirements: 4.1, 4.2, 4.4_

- [x] **8. Integrate Aggregation with Experiment Runner**

  - Call `aggregate_comparative_results()` after all individual experiments complete
  - Pass aggregated CSV to existing `generate_eri_visualizations()` function
  - Create comparative output directory following existing naming conventions
  - Use existing error handling patterns for aggregation and visualization failures
  - Maintain existing experiment result reporting and summary generation
  - _Requirements: 4.2, 4.3, 4.4_

- [x] **9. Enhance Comparative Results Reporting**
  - Extend existing summary table in `run_comparative_experiment()` to show comparative metrics
  - Add PD_t, SFR_rel, and AD columns when baseline data is available
  - Use existing metrics calculation from ERI system for consistent reporting
  - Add clear indicators when baseline methods are missing or incomplete
  - Maintain existing result formatting and console output patterns
  - _Requirements: 4.5, 5.2, 5.3_

---

## Phase 4: Visualization Enhancement Validation

- [x] **10. Validate Multi-Method ERI Visualization**

  - Test existing `eri_vis.processing.ERITimelineProcessor` with multi-method datasets
  - Verify existing `compute_performance_deficits()` correctly uses Scratch_T2 as baseline
  - Confirm existing `compute_sfr_relative()` calculates relative metrics properly
  - Test existing `ERIDynamicsPlotter` generates comparative plots with multiple methods
  - Validate existing `ERIHeatmapPlotter` creates robustness heatmaps across all methods
  - _Requirements: 2.1, 2.3, 3.1, 3.2_

- [x] **11. Test Comparative Visualization Pipeline**

  - Run end-to-end test with aggregated multi-method dataset
  - Verify comparative dynamics plots show all methods on same axes with distinct styling
  - Confirm PD_t and SFR_rel panels display cross-method comparisons correctly
  - Test heatmap generation with multiple methods and threshold sensitivity analysis
  - Validate all visualizations maintain existing quality and formatting standards
  - _Requirements: 2.1, 2.2, 2.4, 2.5_

- [x] **12. Implement Missing Baseline Detection and Warnings**
  - Add validation function to check for required baseline methods in aggregated datasets
  - Provide clear warnings when Scratch_T2 or Interleaved baselines are missing
  - Allow visualization pipeline to continue with available data when baselines are incomplete
  - Add informative error messages indicating which baseline methods need to be run
  - Integrate warnings with existing logging system for consistent user experience
  - _Requirements: 3.3, 3.4_

---

## Phase 5: Statistical Analysis and Enhanced Reporting

- [x] **13. Implement Statistical Significance Testing**

  - Add statistical analysis functions for comparing methods on key metrics (final accuracy, AD, PD_t, SFR_rel)
  - Implement t-tests and ANOVA for pairwise and multi-group comparisons
  - Add multiple comparison corrections (Bonferroni, FDR) to control family-wise error rates
  - Calculate confidence intervals and effect sizes for robust statistical reporting
  - Integrate statistical functions with existing metrics calculation pipeline
  - _Requirements: 6.1, 6.2, 6.3_

- [x] **14. Enhance Comparative Reporting with Statistics**

  - Extend comparative summary table to include statistical significance indicators
  - Add confidence intervals and p-values to method comparison outputs
  - Create detailed statistical analysis report with effect sizes and power analysis
  - Include significance markers (\*, **, \***) in visualizations and tables
  - Generate comprehensive comparative analysis report with statistical validation
  - _Requirements: 5.1, 5.4, 6.4, 6.5_

- [-] **15. Implement Enhanced Output Organization**
  - Create structured directory hierarchy for comparative analysis outputs
  - Organize individual method results, aggregated data, and comparative visualizations
  - Generate master comparative report combining individual and cross-method analysis
  - Include experimental metadata, method configurations, and statistical summaries
  - Follow existing naming conventions and output formatting standards
  - _Requirements: 5.1, 5.2, 5.5_

---

## Phase 6: Integration Testing and Validation

- [ ] **16. Comprehensive Integration Testing**

  - Create end-to-end integration test for complete comparative analysis pipeline
  - Test baseline method training, evaluation, and CSV generation
  - Validate data aggregation, visualization generation, and statistical analysis
  - Test error handling, checkpoint management, and experiment orchestration
  - Verify backward compatibility with existing single-method experiments
  - _Requirements: All requirements validation_

- [ ] **17. Performance and Scalability Testing**

  - Test comparative analysis with multiple seeds and methods for performance
  - Validate memory usage and computational efficiency with large datasets
  - Test CSV aggregation performance with multiple large experiment results
  - Verify visualization generation time remains acceptable with increased method counts
  - Ensure statistical analysis scales appropriately with number of methods and seeds
  - _Requirements: Performance validation_

- [ ] **18. Documentation and User Guide Updates**
  - Update README with comparative analysis usage instructions
  - Document new baseline methods and their intended use cases
  - Create examples showing how to run comparative experiments with baselines
  - Document statistical analysis outputs and interpretation guidelines
  - Add troubleshooting guide for common comparative analysis issues
  - _Requirements: Documentation and usability_

---

## Success Criteria

The implementation is complete when:

**Core Functionality:**

- Baseline methods (Scratch_T2, Interleaved) integrate seamlessly with existing Mammoth infrastructure
- Comparative experiments automatically include baseline methods and generate cross-method analysis
- ERI visualization system produces comparative dynamics plots, heatmaps, and statistical analysis
- All functionality works with existing checkpoint management, error handling, and experiment orchestration

**Quality Assurance:**

- All unit and integration tests pass including new baseline method and aggregation tests
- End-to-end comparative experiment pipeline works reliably with multiple methods and seeds
- Statistical analysis provides robust significance testing and effect size calculations
- Performance remains acceptable with increased method counts and dataset sizes

**User Experience:**

- Existing single-method experiments continue to work unchanged (backward compatibility)
- Comparative analysis is accessible through existing `--comparative` flag without additional configuration
- Clear error messages and warnings guide users when baseline methods are missing
- Comprehensive documentation enables users to understand and interpret comparative results

**Technical Excellence:**

- Implementation follows existing Mammoth patterns and conventions exactly
- No code duplication - all new functionality extends existing systems appropriately
- Error handling and edge cases are managed consistently with existing infrastructure
- All outputs maintain existing quality standards and formatting conventions
