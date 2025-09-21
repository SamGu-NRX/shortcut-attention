# Design Document

## Overview

This design implements a minimal, robust dataset caching system for the Einstellung dataset that leverages Mammoth's existing, proven infrastructure. Rather than duplicating functionality, we make targeted modifications to eliminate the on-the-fly image processing bottleneck while maintaining full compatibility with Mammoth's continual learning framework.

The solution improves training speed from 3 it/s to 15-30+ it/s by preprocessing and caching Einstellung images once, then serving them directly from cache. All existing Mammoth functionality (task splitting, evaluation, class ordering, etc.) works unchanged.

This design implements a robust dataset caching system for the Einstellung dataset that integrates seamlessly with Mammoth's continual learning framework. The solution eliminates the performance bottleneck caused by on-the-fly image processing by preprocessing and caching all Einstellung images, improving training speed while maintaining full compatibility with Mammoth's architecture.


## Architecture

### Minimal Wrapper Approach

The design leverages Mammoth's existing, proven functionality rather than duplicating it. The core insight is that Mammoth's `ContinualDataset` and `store_masked_loaders` already handle all the complex data management - we only need to modify the underlying dataset's `__getitem__` method to return cached images instead of processing them on-the-fly.

#### 1. Minimal Cache Integration
Instead of complex mixins, we modify the existing `MyCIFAR100Einstellung` class to:
- Check for cached images on first access
- Build cache if missing using existing `_apply_einstellung_effect` method
- Replace `__getitem__` to return cached data
- Maintain identical interface to original implementation

#### 2. Leverage Existing Mammoth Infrastructure
**What we DON'T duplicate:**
- Data loading pipeline (store_masked_loaders works as-is)
- Task splitting logic (ContinualDataset handles this)
- Evaluation systems (all existing evaluation works unchanged)
- Class ordering and permutation (fix_class_names_order works as-is)
- Validation and noise injection (existing systems work unchanged)

**What we DO add:**
- Simple cache check in dataset initialization
- Cache building using existing processing methods
- Cache loading in `__getitem__`

#### 3. Simple Cache Structure
**Storage Location:** `base_path() + 'CIFAR100/einstellung_cache/'`
**Minimal Structure:**
```
data/CIFAR100/einstellung_cache/
├── train_<params_hash>.pkl    # All training data in one file
└── test_<params_hash>.pkl     # All test data in one file
```

## Components and Interfaces

### Modified Dataset Classes

We create minimal modifications to existing classes:

```python
class MyCIFAR100EinstellungCached(MyCIFAR100Einstellung):
    """Minimal cached version - only overrides __init__ and __getitem__"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_cache()  # Simple cache setup

    def __getitem__(self, index):
        # Return cached data if available, otherwise fallback to original
        if hasattr(self, '_cached_data'):
            return self._get_cached_item(index)
        return super().__getitem__(index)
```

### Cache Management (Minimal)

```python
def _setup_cache(self):
    """Minimal cache setup - check if cache exists, build if needed"""
    cache_key = self._get_cache_key()
    cache_path = self._get_cache_path(cache_key)

    if os.path.exists(cache_path):
        self._load_cache(cache_path)
    else:
        self._build_cache(cache_path)

def _build_cache(self, cache_path):
    """Build cache using existing _apply_einstellung_effect method"""
    # Use existing proven processing logic
    # Store results in simple pickle file

def _load_cache(self, cache_path):
    """Load cache from pickle file"""
    # Simple pickle load with error handling
```

## Data Models

### Minimal Cache Structure

```python
@dataclass
class CacheData:
    processed_images: np.ndarray  # All preprocessed images
    original_data: np.ndarray     # Original CIFAR data (unchanged)
    targets: np.ndarray          # Targets (unchanged)
    params_hash: str             # Parameter validation
```

## Error Handling

### Simple Fallback Strategy
- If cache loading fails → fallback to original implementation
- If cache building fails → fallback to original implementation
- If parameter mismatch → rebuild cache automatically
- All errors logged using existing Mammoth logging

## Testing Strategy

### Validation Approach
1. **Pixel-Perfect Validation**: Compare cached vs on-the-fly results
2. **Integration Testing**: Ensure all existing Mammoth functionality works
3. **Performance Testing**: Measure speed improvement

### Existing Test Compatibility
All existing Mammoth tests continue to work unchanged since we maintain identical interfaces.

## Implementation Strategy

### Phase 1: Core Caching (Minimal)
- Add simple cache check to existing dataset classes
- Implement basic cache building using existing processing methods
- Add cache loading in `__getitem__`

### Phase 2: Integration and Validation
- Ensure compatibility with all existing Mammoth systems
- Add pixel-perfect validation
- Performance benchmarking

### Phase 3: Robustness
- Add error handling and fallback
- Cache parameter validation
- Documentation

## Key Design Principles

### 1. Leverage Existing Mammoth Code
- Use existing `_apply_einstellung_effect` for cache building
- Maintain existing class hierarchy and interfaces
- Work with existing `store_masked_loaders` pipeline

### 2. Minimal Changes
- Only modify `__init__` and `__getitem__` methods
- No new base classes or complex inheritance
- No duplication of existing Mammoth functionality

### 3. Robust Fallback
- Always fallback to original implementation on any error
- Maintain identical behavior when cache is disabled
- Comprehensive error logging

### 4. Simple Cache Format
- Single pickle file per dataset split
- Parameter-based cache keys using secure hashing
- No complex metadata or directory structures

## Backward Compatibility

The design ensures complete backward compatibility:
- Existing experiment scripts work unchanged
- All Mammoth framework features work identically
- Cache can be disabled with a single parameter
- Fallback to original implementation is transparent

## Performance Expectations

- **Cache Building**: One-time cost, ~2-5 minutes for full dataset
- **Training Speed**: 5-10x improvement (3 it/s → 15-30+ it/s)
- **Memory Usage**: Minimal increase (cache loaded on-demand)
- **Disk Usage**: ~500MB-1GB for cached dataset

This minimal approach ensures we get maximum performance benefit while maintaining full compatibility with Mammoth's proven, robust infrastructure.
