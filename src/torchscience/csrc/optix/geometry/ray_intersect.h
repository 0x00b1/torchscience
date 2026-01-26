// src/torchscience/csrc/optix/geometry/ray_intersect.h
// OptiX-accelerated BVH ray intersection
#pragma once

#ifdef TORCHSCIENCE_OPTIX

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>
#include <optix.h>

#include "../context.h"

namespace torchscience::optix::geometry {

inline std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
bvh_ray_intersect(
    int64_t scene_handle,
    const at::Tensor& origins,
    const at::Tensor& directions) {
  TORCH_CHECK(
      origins.dim() >= 1 && origins.size(-1) == 3,
      "bvh_ray_intersect: origins must have shape (..., 3), got ",
      origins.sizes());
  TORCH_CHECK(
      directions.dim() >= 1 && directions.size(-1) == 3,
      "bvh_ray_intersect: directions must have shape (..., 3), got ",
      directions.sizes());
  TORCH_CHECK(
      origins.sizes() == directions.sizes(),
      "bvh_ray_intersect: origins and directions must have matching shapes");

  c10::cuda::CUDAGuard device_guard(origins.device());

  auto& ctx = OptiXContext::instance();

  // Negate handle to get positive registry key
  GASResources gas = ctx.get_gas(-scene_handle);

  // Flatten batch dimensions
  auto batch_shape = origins.sizes().vec();
  batch_shape.pop_back();  // remove trailing 3

  at::Tensor origins_flat = origins.reshape({-1, 3}).contiguous().to(at::kFloat);
  at::Tensor dirs_flat = directions.reshape({-1, 3}).contiguous().to(at::kFloat);
  int64_t num_rays = origins_flat.size(0);

  // Allocate outputs on device
  auto float_opts = origins.options().dtype(at::kFloat);
  at::Tensor t_out = at::full({num_rays}, 1e20f, float_opts);
  at::Tensor hit_out = at::zeros({num_rays}, float_opts.dtype(at::kInt));
  at::Tensor geom_id_out = at::full({num_rays}, -1, float_opts.dtype(at::kLong));
  at::Tensor prim_id_out = at::full({num_rays}, -1, float_opts.dtype(at::kLong));
  at::Tensor u_out = at::zeros({num_rays}, float_opts);
  at::Tensor v_out = at::zeros({num_rays}, float_opts);

  // Set up launch params
  LaunchParams params = {};
  params.mode = LaunchMode::INTERSECT;
  params.ray_origins = origins_flat.data_ptr<float>();
  params.ray_directions = dirs_flat.data_ptr<float>();
  params.num_rays = static_cast<int>(num_rays);
  params.traversable = gas.traversable;
  params.t_out = t_out.data_ptr<float>();
  params.hit_out = hit_out.data_ptr<int>();
  params.geometry_id_out = geom_id_out.data_ptr<int64_t>();
  params.primitive_id_out = prim_id_out.data_ptr<int64_t>();
  params.u_out = u_out.data_ptr<float>();
  params.v_out = v_out.data_ptr<float>();

  // Upload launch params
  CUdeviceptr d_params = 0;
  cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams));
  cudaMemcpy(
      reinterpret_cast<void*>(d_params), &params,
      sizeof(LaunchParams), cudaMemcpyHostToDevice);

  // Launch OptiX pipeline
  OPTIX_CHECK(optixLaunch(
      ctx.pipeline(),
      at::cuda::getCurrentCUDAStream(),
      d_params, sizeof(LaunchParams),
      &ctx.sbt(),
      static_cast<unsigned int>(num_rays), 1, 1));

  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  cudaFree(reinterpret_cast<void*>(d_params));

  // Convert hit from int to bool
  at::Tensor hit_bool = hit_out.to(at::kBool);

  // Reshape outputs to original batch shape
  return std::make_tuple(
      t_out.reshape(batch_shape),
      hit_bool.reshape(batch_shape),
      geom_id_out.reshape(batch_shape),
      prim_id_out.reshape(batch_shape),
      u_out.reshape(batch_shape),
      v_out.reshape(batch_shape));
}

}  // namespace torchscience::optix::geometry

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
  m.impl(
      "bvh_ray_intersect",
      torchscience::optix::geometry::bvh_ray_intersect);
}

#endif  // TORCHSCIENCE_OPTIX
