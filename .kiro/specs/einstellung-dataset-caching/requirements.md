# Requirements Document

## Introduction

The Einstellung dataset implementation in this codebase suffers from severe performance bottlenecks due to on-the-fly image processing during training. Currently, every image undergoes expensive PIL→numpy→PIL conversions and patch application for every training iteration, resulting in only 3 it/s with 20% GPU utilization instead of the expected 15-30+ it/s with 80-95% GPU utilization.

This feature will implement a robust dataset caching system that integrates seamlessly with Mammoth's existing continual learning framework. The solution must follow Mammoth's native dataset patterns, inherit from existing proven implementations, and maintain full compatibility with the framework's evaluation and logging systems. No custom or hacky solutions will be implemented - only robust, well-tested approaches that align with Mammoth's architecture.

This feature will implement a comprehensive dataset caching system to preprocess and cache all Einstellung images, eliminating the performance bottleneck and enabling proper GPU utilization during training - best practices in PyTorch.

## Requirements

### Requirement 1: Robust Dataset Preprocessing and Caching

**User Story:** As a researcher running Einstellung experiments, I want the dataset to preprocess all images once using proven, robust methods and cache them reliably, so that training runs at optimal speed without repeated image processing overhead.

#### Acceptance Criteria

1. WHEN the Einstellung dataset is initialized for the first time THEN the system SHALL preprocess all images using the existing proven Einstellung effect methods and cache them to disk using robust serialization
2. WHEN the dataset is accessed during training THEN the system SHALL load preprocessed images from cache using Mammoth's standard data loading patterns
3. WHEN cache parameters match existing cache THEN the system SHALL reuse existing cached data without reprocessing, using secure hash-based validation
4. WHEN cache parameters differ from existing cache THEN the system SHALL rebuild the cache with new parameters using atomic operations to prevent corruption
5. IF cache building fails THEN the system SHALL fall back to the original on-the-fly processing implementation with comprehensive error logging
6. WHEN caching is performed THEN it SHALL use only proven Python standard library methods (pickle, os, hashlib) and avoid custom serialization formats

### Requirement 2: Robust Cache Management and Storage

**User Story:** As a researcher with limited disk space, I want reliable cache management that follows standard practices and handles storage efficiently, so that I can run experiments without running out of disk space or dealing with corrupted cache files.

#### Acceptance Criteria

1. WHEN cache is created THEN the system SHALL store it in Mammoth's standard data directory structure following existing patterns (base_path() + 'CIFAR100/einstellung_cache/')
2. WHEN cache files are written THEN the system SHALL use atomic operations and checksums to prevent corruption during concurrent access
3. WHEN dataset parameters change THEN the system SHALL automatically detect parameter mismatches using secure hash comparison and rebuild cache safely
4. WHEN cache corruption is detected THEN the system SHALL automatically rebuild the cache using the original proven dataset implementation as fallback
5. WHEN cache operations are performed THEN the system SHALL use standard Python file locking mechanisms to handle concurrent access
6. WHEN cache cleanup is needed THEN the system SHALL provide standard utilities that follow filesystem best practices

### Requirement 3: Performance Optimization

**User Story:** As a researcher running time-sensitive experiments, I want the caching system to deliver maximum performance improvements, so that I can complete experiments in reasonable time with full GPU utilization.

#### Acceptance Criteria

1. WHEN using cached dataset THEN training speed SHALL achieve 15-30+ iterations per second (5-10x improvement over current 3 it/s)
2. WHEN using cached dataset THEN GPU utilization SHALL reach 80-95% (improvement from current 20%)
3. WHEN cache is being built THEN the system SHALL show progress indicators and estimated completion time
4. WHEN cache is loaded THEN memory usage SHALL be optimized through lazy loading or memory mapping
5. WHEN multiple workers access cache THEN the system SHALL handle concurrent access safely

### Requirement 4: Mammoth Framework Integration and Compatibility

**User Story:** As a researcher using Mammoth's continual learning framework, I want the caching system to integrate seamlessly with all existing Mammoth components, so that I can benefit from performance improvements without breaking any framework functionality.

#### Acceptance Criteria

1. WHEN the cached dataset is used THEN it SHALL inherit from existing Mammoth dataset base classes (ContinualDataset, MyCIFAR100, etc.)
2. WHEN Mammoth's store_masked_loaders is called THEN the cached dataset SHALL integrate properly with the framework's data loading pipeline
3. WHEN Mammoth's evaluation systems are used THEN the cached dataset SHALL provide identical results to the original implementation
4. WHEN Mammoth's logging and metrics systems access the dataset THEN all existing functionality SHALL work without modification
5. WHEN the dataset is used with any Mammoth model THEN it SHALL maintain full compatibility with the framework's training and evaluation loops
6. WHEN existing experiment configurations are used THEN they SHALL work without any script modifications
7. WHEN Mammoth's class name fixing and ordering systems are used THEN the cached dataset SHALL integrate properly

### Requirement 5: Multi-Resolution and Multi-Configuration Support

**User Story:** As a researcher testing different model architectures, I want the caching system to support multiple image resolutions and configurations, so that I can run experiments with both ResNet (32x32) and ViT (224x224) models efficiently.

#### Acceptance Criteria

1. WHEN using 32x32 resolution THEN the system SHALL cache images optimized for ResNet training
2. WHEN using 224x224 resolution THEN the system SHALL cache images optimized for ViT training
3. WHEN different patch sizes are used THEN the system SHALL maintain separate caches for each configuration
4. WHEN different shortcut configurations are used THEN the system SHALL cache appropriate variants (normal, shortcut, masked)
5. WHEN switching between configurations THEN the system SHALL automatically select the correct cached dataset

### Requirement 6: Comprehensive Validation and Quality Assurance

**User Story:** As a researcher ensuring experiment validity, I want the caching system to rigorously validate that cached images are identical to on-the-fly processed images using proven testing methods, so that I can trust that results are not affected by the caching implementation.

#### Acceptance Criteria

1. WHEN cache is built THEN the system SHALL validate that cached images match on-the-fly processing results using pixel-perfect comparison and cryptographic hashes
2. WHEN cache is loaded THEN the system SHALL perform integrity checks using checksums and file size validation to detect corruption
3. WHEN validation fails THEN the system SHALL rebuild cache automatically using the original proven implementation and log comprehensive error details
4. WHEN debugging is enabled THEN the system SHALL provide standard Python unittest-compatible tools to compare cached vs on-the-fly results
5. WHEN cache statistics are requested THEN the system SHALL report metrics using Mammoth's existing logging infrastructure
6. WHEN the cached dataset is tested THEN it SHALL pass all existing Mammoth dataset tests without modification
