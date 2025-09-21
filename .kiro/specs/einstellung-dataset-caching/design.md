# Design Document

## Overview

This design implements a robust dataset caching system for the Einstellung dataset that integrates seamlessly with Mammoth's continual learning framework. The solution eliminates the performance bottleneck caused by on-the-fly image processing by preprocessing and caching all Einstellung images, improving training speed from 3 it/s to 15-30+ it/s while maintaining full compatibility with Mammoth's architecture.

The design follows Mammoth's proven patterns by inheriting from existing dataset classes, using the framework's standard data loading pipeline, and maintaining identical behavior to the original implementation. No custom or hacky solutions are introduced - only robust, well-tested approaches that align with Mammoth's native processes.

## Architecture

### Core Components

#### 1. CachedEinstellungMixin
A mixin class that provides caching functionality to any Einstellung dataset implementation. This follows Mammoth's pattern of using mixins for extending functionalitye maintaining inheritance from proven base classes.

**Key Features:**
- Integrates with existing Einstellung dataset classes through multiple inheritance
- Uses Mammoth's standard `base_path()` for cache storage location
- Implements atomic cache operations using Python's standard library
- Provides fallback to original implementation on cache failures

#### 2. Cache Storage System
Utilizes Mammoth's existing data directory structure and follows proven storage patterns.

**Storage Location:** `base_path() + 'CIFAR100/einstellung_cache/'`
**Cache Structure:**
```
data/CIFAR100/einstellung_cache/
├── metadata/
│   ├── cache_params_<hash>.json    # Cache parameter validation
│   └── integrity_<hash>.json       # Integrity checksums
├── train/
│   ├── cached_images_<hash>.pkl    # Preprocessed training images
│   └── cached_targets_<hash>.pkl   # Corresponding targets
└── test/
    ├── cached_images_<hash>.pkl    # Preprocessed test images
    └── cached_targets_<hash>.pkl   # Corresponding targets
```

#### 3. Enhanced Dataset Classes
New dataset classes that inherit from proven Mammoth implementations and add caching capabilities.

**Class Hierarchy:**
```
CachedSequentialCIFAR100Einstellung(SequentialCIFAR100Einstellung, CachedEinstellungMixin)
CachedSequentialCIFAR100Einstellung224(SequentialCIFAR100Einstellung224, CachedEinstellungMixin)
```

## Components and Interfaces

### CachedEinstellungMixin Interface

```python
class CachedEinstellungMixin:
    def __init__(self, enable_cache=True, cache_validation=True, **kwargs):
        """Initialize caching system with validation"""

    def _get_cache_key(self) -> str:
        """Generate secure hash-based cache key from parameters"""

    def _build_cache(self) -> bool:
        """Build cache using original dataset implementation"""

    def _load_cache(self) -> bool:
        """Load existing cache with integrity validation"""

    def _validate_cache_integrity(self) -> bool:
        """Validate cache integrity using checksums"""

    def _fallback_to_original(self):
        """Fallback to original implementation on cache failure"""
```

### Cache Management Interface

```python
class CacheManager:
    def create_cache_directory(self, cache_path: str) -> bool:
        """Create cache directory structure"""

    def atomic_write(self, data: Any, filepath: str) -> bool:
        """Atomic write operation with integrity checks"""

    def validate_parameters(self, cache_key: str, current_params: dict) -> bool:
        """Validate cache parameters match current configuration"""

    def cleanup_old_caches(self, keep_recent: int = 3) -> None:
        """Clean up old cache files"""
```

### Integration with Mammoth Framework

The cached datasets integrate with Mammoth's framework through:

1. **ContinualDataset Inheritance**: All cached classes inherit from proven Mammoth dataset implementations
2. **store_masked_loaders Integration**: Cache system works seamlessly with Mammoth's data loading pipeline
3. **MammothDatasetWrapper Compatibility**: Cached datasets work with Mammoth's wrapper system
4. **Evaluation System Integration**: All evaluation subsets and metrics work identically

## Data Models

### Cache Metadata Model

```python
@dataclass
class CacheMetadata:
    cache_version: str
    dataset_name: str
    parameters: Dict[str, Any]  # patch_size, patch_color, etc.
    creation_timestamp: float
    dataset_size: int
    checksum: str
    python_version: str
    torch_version: str
```

### Cache Entry Model

```python
@dataclass
class CacheEntry:
    images: np.ndarray  # Preprocessed images as numpy arrays
    targets: np.ndarray  # Corresponding targets
    task_ids: np.ndarray  # Task assignments
    metadata: CacheMetadata
    integrity_hash: str
```

## Error Handling

### Cache Failure Recovery
The system implements comprehensive error handling with automatic fallback:

1. **Cache Corruption Detection**: Checksum validation on cache load
2. **Parameter Mismatch**: Automatic cache rebuild when parameters change
3. **Disk Space Issues**: Graceful degradation to original implementation
4. **Concurrent Access**: File locking to prevent corruption during multi-process training
5. **Version Incompatibility**: Automatic cache rebuild for version mismatches

### Error Logging
All cache operations use Mammoth's standard logging infrastructure:
- Cache hits/misses
- Build progress and completion
- Fallback activations
- Error conditions with detailed context

## Testing Strategy

### Unit Tests
Following Mammoth's testing patterns in the `tests/` directory:

1. **Cache Functionality Tests**
   - Cache creation and loading
   - Parameter validation
   - Integrity checking
   - Atomic operations

2. **Dataset Compatibility Tests**
   - Identical behavior to original implementation
   - Integration with store_masked_loaders
   - Evaluation subset generation
   - Class name ordering

3. **Performance Tests**
   - Cache build time measurement
   - Loading speed comparison
   - Memory usage validation
   - Concurrent access handling

### Integration Tests
1. **Mammoth Framework Integration**
   - Full training loop compatibility
   - Model evaluation consistency
   - Logging system integration
   - Multi-task scenario testing

2. **Cross-Resolution Testing**
   - 32x32 (ResNet) and 224x224 (ViT) compatibility
   - Parameter-specific cache isolation
   - Configuration switching validation

### Validation Tests
1. **Pixel-Perfect Comparison**
   - Cached vs on-the-fly image comparison
   - Statistical validation of results
   - Einstellung effect preservation
   - Evaluation metric consistency

## Performance Optimizations

### Memory Management
1. **Lazy Loading**: Load cache data only when needed
2. **Memory Mapping**: Use memory-mapped files for large caches
3. **Batch Processing**: Process cache in configurable batches
4. **Garbage Collection**: Explicit cleanup of temporary objects

### Disk I/O Optimization
1. **Atomic Operations**: Prevent corruption during writes
2. **Compression**: Optional compression for cache files
3. **Parallel I/O**: Multi-threaded cache building where safe
4. **Progress Tracking**: User feedback during cache operations

### CUDA Integration
The cached dataset maintains full compatibility with existing CUDA optimizations:
- TF32 and cuDNN benchmark settings
- Optimal DataLoader worker configuration
- Pin memory for GPU transfers
- Proper tensor device placement

## Implementation Phases

### Phase 1: Core Caching Infrastructure
- Implement CachedEinstellungMixin
- Create cache management utilities
- Add parameter validation system
- Implement atomic file operations

### Phase 2: Dataset Integration
- Create cached dataset classes
- Integrate with Mammoth's data loading pipeline
- Implement fallback mechanisms
- Add comprehensive error handling

### Phase 3: Validation and Testing
- Implement pixel-perfect validation
- Create comprehensive test suite
- Performance benchmarking
- Integration testing with existing experiments

### Phase 4: Documentation and Optimization
- Add usage documentation
- Performance tuning
- Memory optimization
- Cache cleanup utilities

## Backward Compatibility

The design ensures complete backward compatibility:

1. **Existing Scripts**: No changes required to existing experiment scripts
2. **Parameter Compatibility**: All existing Einstellung parameters supported
3. **Evaluation Systems**: Identical evaluation results and metrics
4. **Framework Integration**: Seamless integration with all Mammoth components

## Security and Robustness

### Data Integrity
- SHA-256 checksums for all cache files
- Parameter validation using secure hashing
- Atomic write operations to prevent corruption
- Version compatibility checking

### Concurrent Access
- File locking for multi-process safety
- Atomic cache building operations
- Safe fallback during concurrent access
- Process-safe cache validation

### Error Recovery
- Comprehensive exception handling
- Automatic cache rebuilding on corruption
- Graceful degradation to original implementation
- Detailed error logging for debugging
