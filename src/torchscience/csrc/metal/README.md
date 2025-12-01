# Metal Shaders

This directory contains Metal shader implementations for MPS backend.

## Structure

```
metal/
  special_functions/   # Special function kernels
  signal_processing/   # Signal processing kernels
  statistics/          # Statistics kernels
  ...
```

## Adding a New Shader

1. Create `.metal` file mirroring the CUDA structure
2. Add file path to `cmake/mps.cmake` TORCHSCIENCE_METAL_SOURCES
3. Implement MPS dispatch in `torchscience.cpp`
