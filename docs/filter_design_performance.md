# Filter Design Performance Analysis

## Overview

This document summarizes the performance characteristics of the torchscience filter design module compared to scipy. The implementations prioritize PyTorch integration (autograd, torch.compile, vmap) over raw performance.

## Benchmark Results

### Filter Design Functions

| Function | torchscience | scipy | Ratio |
|----------|--------------|-------|-------|
| butterworth_design (order=8) | 394µs | 292µs | 1.35x slower |
| firwin (101 taps) | 79µs | 52µs | 1.53x slower |
| remez (51 taps) | Uses scipy internally | - | ~1x |

### Filter Application Functions

| Function | torchscience | scipy | Ratio |
|----------|--------------|-------|-------|
| fftfilt (100k samples, 101 taps) | 9.9ms | 2.3ms | 4.3x slower |
| lfilter | Pure Python loop | C extension | Slower for long signals |
| sosfilt | Pure Python loop | C extension | Slower for long signals |

## Performance Trade-offs

### Why torchscience is slower

1. **Pure PyTorch implementation**: All operations are PyTorch tensor operations for autograd compatibility
2. **No C extensions**: scipy uses highly optimized C/Fortran code
3. **IIR loops**: lfilter/sosfilt require sequential sample-by-sample processing, which is slower in Python

### Why this is acceptable

1. **Differentiability**: All functions support autograd for gradient-based optimization
2. **Batching**: Multiple filters/signals can be processed in parallel
3. **GPU acceleration**: Operations can run on GPU (not benchmarked above)
4. **torch.compile**: JIT compilation can improve performance
5. **Integration**: Seamless integration with PyTorch training pipelines

## Optimization Opportunities

### Short-term
- Use scipy as backend when gradients aren't needed (already done for remez)
- Implement C++ kernels for critical paths (lfilter, sosfilt)

### Medium-term
- torch.compile optimization for fftfilt
- CUDA kernels for filter application

### Long-term
- Custom autograd functions with efficient backward passes
- Fused operations for common patterns

## Recommendations

1. **For training pipelines**: Use torchscience functions directly for gradient support
2. **For inference without gradients**: Consider scipy for maximum performance
3. **For batch processing**: torchscience batched functions can be competitive
4. **For GPU workloads**: torchscience will be faster due to GPU acceleration

## GPU Performance (Future)

GPU benchmarks will be added when CUDA implementations are available. Expected improvements:
- FFT-based operations (fftfilt): 10-100x faster on GPU
- Batched filter design: Linear scaling with batch size
- Adaptive filters: Significant speedup for parallel processing
