# cmake/cpu.cmake
# CPU backend configuration - always included

# CPU-specific sources (if any beyond torchscience.cpp)
# Currently empty - CPU kernels are header-only and included in torchscience.cpp

# No additional compile definitions needed for CPU
# CPU is the default fallback, always available
