# cmake/backend_detection.cmake
# Platform-aware backend detection with user override support

# User-configurable options (AUTO means auto-detect)
set(TORCHSCIENCE_ENABLE_CUDA "AUTO" CACHE STRING "Enable CUDA backend (ON/OFF/AUTO)")
set(TORCHSCIENCE_ENABLE_MPS "AUTO" CACHE STRING "Enable MPS/Metal backend (ON/OFF/AUTO)")

# CPU is always enabled, no option needed

# CUDA auto-detection
if(TORCHSCIENCE_ENABLE_CUDA STREQUAL "AUTO")
  if(TORCH_CUDA_LIBRARIES)
    set(TORCHSCIENCE_ENABLE_CUDA ON)
    message(STATUS "CUDA auto-detected: PyTorch has CUDA support")
  else()
    set(TORCHSCIENCE_ENABLE_CUDA OFF)
    message(STATUS "CUDA auto-detected: PyTorch built without CUDA")
  endif()
endif()

# MPS auto-detection (Apple Silicon only)
if(TORCHSCIENCE_ENABLE_MPS STREQUAL "AUTO")
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(TORCHSCIENCE_ENABLE_MPS ON)
    message(STATUS "MPS auto-detected: Apple Silicon Mac")
  else()
    set(TORCHSCIENCE_ENABLE_MPS OFF)
    if(APPLE)
      message(STATUS "MPS auto-detected: Intel Mac (MPS not supported)")
    else()
      message(STATUS "MPS auto-detected: Not macOS")
    endif()
  endif()
endif()

# Print backend summary
message(STATUS "")
message(STATUS "torchscience backends:")
message(STATUS "  CPU:  ON (always enabled)")
message(STATUS "  CUDA: ${TORCHSCIENCE_ENABLE_CUDA}")
message(STATUS "  MPS:  ${TORCHSCIENCE_ENABLE_MPS}")
message(STATUS "")
