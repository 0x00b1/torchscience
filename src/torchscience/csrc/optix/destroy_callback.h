// src/torchscience/csrc/optix/destroy_callback.h
// Global callback for OptiX scene cleanup, avoids coupling CPU and OptiX code
#pragma once

#include <cstdint>

namespace torchscience::optix {

// Called by OptiX context during initialization to register its cleanup
// function. The CPU bvh_destroy calls this for negative handles.
inline void (*g_scene_destroy)(int64_t) = nullptr;

}  // namespace torchscience::optix
