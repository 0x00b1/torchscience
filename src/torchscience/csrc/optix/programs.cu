// src/torchscience/csrc/optix/programs.cu
// OptiX device programs for BVH ray tracing
// Compiled to PTX at build time, embedded and loaded at runtime

#include <optix_device.h>

#include "launch_params.h"

using namespace torchscience::optix;

extern "C" __constant__ LaunchParams params;

// Helper: reinterpret float as uint for payload transport
static __forceinline__ __device__ unsigned int float_as_uint(float f) {
  return __float_as_uint(f);
}

static __forceinline__ __device__ float uint_as_float(unsigned int u) {
  return __uint_as_float(u);
}

// Ray generation program - handles both intersection and occlusion
extern "C" __global__ void __raygen__rg() {
  const unsigned int idx = optixGetLaunchIndex().x;
  if (idx >= static_cast<unsigned int>(params.num_rays))
    return;

  const float ox = params.ray_origins[idx * 3 + 0];
  const float oy = params.ray_origins[idx * 3 + 1];
  const float oz = params.ray_origins[idx * 3 + 2];
  const float dx = params.ray_directions[idx * 3 + 0];
  const float dy = params.ray_directions[idx * 3 + 1];
  const float dz = params.ray_directions[idx * 3 + 2];

  if (params.mode == LaunchMode::INTERSECT) {
    // Trace intersection ray (ray type 0)
    unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0;

    optixTrace(
        params.traversable,
        make_float3(ox, oy, oz),
        make_float3(dx, dy, dz),
        0.0f,                          // tmin
        1e20f,                         // tmax
        0.0f,                          // rayTime
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,             // SBT offset
        RAY_TYPE_COUNT,                // SBT stride
        RAY_TYPE_RADIANCE,             // miss index
        p0, p1, p2, p3, p4, p5);

    params.t_out[idx] = uint_as_float(p0);
    params.hit_out[idx] = static_cast<int>(p1);
    params.geometry_id_out[idx] = static_cast<int64_t>(p2);
    params.primitive_id_out[idx] = static_cast<int64_t>(p3);
    params.u_out[idx] = uint_as_float(p4);
    params.v_out[idx] = uint_as_float(p5);
  } else {
    // Trace occlusion ray (ray type 1)
    unsigned int p0 = 0;

    optixTrace(
        params.traversable,
        make_float3(ox, oy, oz),
        make_float3(dx, dy, dz),
        0.0f,
        1e20f,
        0.0f,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        RAY_TYPE_SHADOW,               // SBT offset
        RAY_TYPE_COUNT,                // SBT stride
        RAY_TYPE_SHADOW,               // miss index
        p0);

    params.occluded_out[idx] = static_cast<int>(p0);
  }
}

// Closest hit program - records intersection data for radiance rays
extern "C" __global__ void __closesthit__radiance() {
  const float t = optixGetRayTmax();
  const float2 bary = optixGetTriangleBarycentrics();
  const unsigned int prim_id = optixGetPrimitiveIndex();
  const unsigned int geom_id = optixGetSbtGASIndex();

  optixSetPayload_0(float_as_uint(t));
  optixSetPayload_1(1u);  // hit = true
  optixSetPayload_2(geom_id);
  optixSetPayload_3(prim_id);
  optixSetPayload_4(float_as_uint(bary.x));
  optixSetPayload_5(float_as_uint(bary.y));
}

// Any hit program - terminates ray immediately for shadow/occlusion rays
extern "C" __global__ void __anyhit__shadow() {
  optixSetPayload_0(1u);  // occluded = true
  optixTerminateRay();
}

// Miss program for radiance rays - no intersection found
extern "C" __global__ void __miss__radiance() {
  optixSetPayload_0(float_as_uint(1e20f));  // t = far
  optixSetPayload_1(0u);                     // hit = false
  optixSetPayload_2(0u);                     // geom_id = 0
  optixSetPayload_3(0u);                     // prim_id = 0
  optixSetPayload_4(0u);                     // u = 0
  optixSetPayload_5(0u);                     // v = 0
}

// Miss program for shadow rays - ray is not occluded
extern "C" __global__ void __miss__shadow() {
  optixSetPayload_0(0u);  // occluded = false
}
