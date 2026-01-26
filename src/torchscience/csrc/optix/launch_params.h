// src/torchscience/csrc/optix/launch_params.h
// Shared between host and OptiX device programs
#pragma once

#include <cstdint>

namespace torchscience::optix {

enum class LaunchMode : int {
  INTERSECT = 0,
  OCCLUDE = 1
};

// Number of ray types determines SBT hit group stride
enum { RAY_TYPE_RADIANCE = 0, RAY_TYPE_SHADOW = 1, RAY_TYPE_COUNT = 2 };

struct LaunchParams {
  // Mode selector
  LaunchMode mode;

  // Ray data (flattened [N, 3])
  const float* ray_origins;
  const float* ray_directions;
  int num_rays;

  // Acceleration structure handle (OptixTraversableHandle is unsigned long long)
  unsigned long long traversable;

  // Intersection outputs [N]
  float* t_out;
  int* hit_out;           // stored as int (0/1) for device atomics
  int64_t* geometry_id_out;
  int64_t* primitive_id_out;
  float* u_out;
  float* v_out;

  // Occlusion output [N]
  int* occluded_out;      // stored as int (0/1)
};

// SBT record types (empty - we use launch params for all data)
struct RaygenRecord {
  char header[32]; // OPTIX_SBT_RECORD_HEADER_SIZE
};

struct MissRecord {
  char header[32];
};

struct HitGroupRecord {
  char header[32];
};

}  // namespace torchscience::optix
