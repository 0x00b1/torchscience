// src/torchscience/csrc/optix/geometry/ray_occluded.h
// OptiX-accelerated BVH ray occlusion test
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

inline at::Tensor bvh_ray_occluded(
    int64_t scene_handle,
    const at::Tensor& origins,
    const at::Tensor& directions) {
  TORCH_CHECK(
      origins.dim() >= 1 && origins.size(-1) == 3,
      "bvh_ray_occluded: origins must have shape (..., 3), got ",
      origins.sizes());
  TORCH_CHECK(
      directions.dim() >= 1 && directions.size(-1) == 3,
      "bvh_ray_occluded: directions must have shape (..., 3), got ",
      directions.sizes());
  TORCH_CHECK(
      origins.sizes() == directions.sizes(),
      "bvh_ray_occluded: origins and directions must have matching shapes");

  c10::cuda::CUDAGuard device_guard(origins.device());

  auto& ctx = OptiXContext::instance();
  GASResources gas = ctx.get_gas(-scene_handle);

  auto batch_shape = origins.sizes().vec();
  batch_shape.pop_back();

  at::Tensor origins_flat = origins.reshape({-1, 3}).contiguous().to(at::kFloat);
  at::Tensor dirs_flat = directions.reshape({-1, 3}).contiguous().to(at::kFloat);
  int64_t num_rays = origins_flat.size(0);

  at::Tensor occluded_out = at::zeros(
      {num_rays}, origins.options().dtype(at::kInt));

  LaunchParams params = {};
  params.mode = LaunchMode::OCCLUDE;
  params.ray_origins = origins_flat.data_ptr<float>();
  params.ray_directions = dirs_flat.data_ptr<float>();
  params.num_rays = static_cast<int>(num_rays);
  params.traversable = gas.traversable;
  params.occluded_out = occluded_out.data_ptr<int>();

  CUdeviceptr d_params = 0;
  cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams));
  cudaMemcpy(
      reinterpret_cast<void*>(d_params), &params,
      sizeof(LaunchParams), cudaMemcpyHostToDevice);

  OPTIX_CHECK(optixLaunch(
      ctx.pipeline(),
      at::cuda::getCurrentCUDAStream(),
      d_params, sizeof(LaunchParams),
      &ctx.sbt(),
      static_cast<unsigned int>(num_rays), 1, 1));

  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  cudaFree(reinterpret_cast<void*>(d_params));

  return occluded_out.to(at::kBool).reshape(batch_shape);
}

}  // namespace torchscience::optix::geometry

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
  m.impl(
      "bvh_ray_occluded",
      torchscience::optix::geometry::bvh_ray_occluded);
}

#endif  // TORCHSCIENCE_OPTIX
