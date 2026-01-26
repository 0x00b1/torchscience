// src/torchscience/csrc/optix/geometry/closest_point.h
// OptiX-accelerated BVH closest point query via multi-directional ray casting
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
bvh_closest_point(
    int64_t scene_handle,
    const at::Tensor& query_points) {
  TORCH_CHECK(
      query_points.dim() >= 1 && query_points.size(-1) == 3,
      "bvh_closest_point: query_points must have shape (..., 3), got ",
      query_points.sizes());

  c10::cuda::CUDAGuard device_guard(query_points.device());

  auto& ctx = OptiXContext::instance();
  GASResources gas = ctx.get_gas(-scene_handle);

  auto batch_shape = query_points.sizes().vec();
  batch_shape.pop_back();

  at::Tensor queries_flat =
      query_points.reshape({-1, 3}).contiguous().to(at::kFloat);
  int64_t num_queries = queries_flat.size(0);

  // Cast 6 directional rays per query point
  constexpr int NUM_DIRS = 6;
  constexpr float dirs[NUM_DIRS][3] = {
      {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

  int64_t total_rays = num_queries * NUM_DIRS;

  // Build ray origins and directions for all 6 directions
  // origins: repeat each query point 6 times [N*6, 3]
  // directions: tile the 6 directions N times [N*6, 3]
  at::Tensor all_origins =
      queries_flat.unsqueeze(1)
          .expand({num_queries, NUM_DIRS, 3})
          .reshape({total_rays, 3})
          .contiguous();

  // Create direction tensor on same device
  at::Tensor dir_tensor = at::zeros({NUM_DIRS, 3}, query_points.options().dtype(at::kFloat));
  {
    float dir_data[NUM_DIRS * 3];
    for (int d = 0; d < NUM_DIRS; ++d) {
      dir_data[d * 3 + 0] = dirs[d][0];
      dir_data[d * 3 + 1] = dirs[d][1];
      dir_data[d * 3 + 2] = dirs[d][2];
    }
    at::Tensor dir_cpu = at::from_blob(
        dir_data, {NUM_DIRS, 3}, at::kFloat).clone();
    dir_tensor.copy_(dir_cpu);
  }

  at::Tensor all_directions =
      dir_tensor.unsqueeze(0)
          .expand({num_queries, NUM_DIRS, 3})
          .reshape({total_rays, 3})
          .contiguous();

  // Allocate intersection outputs
  auto float_opts = query_points.options().dtype(at::kFloat);
  at::Tensor t_out = at::full({total_rays}, 1e20f, float_opts);
  at::Tensor hit_out = at::zeros({total_rays}, float_opts.dtype(at::kInt));
  at::Tensor geom_id_out = at::full({total_rays}, -1, float_opts.dtype(at::kLong));
  at::Tensor prim_id_out = at::full({total_rays}, -1, float_opts.dtype(at::kLong));
  at::Tensor u_out = at::zeros({total_rays}, float_opts);
  at::Tensor v_out = at::zeros({total_rays}, float_opts);

  // Launch intersection rays
  LaunchParams params = {};
  params.mode = LaunchMode::INTERSECT;
  params.ray_origins = all_origins.data_ptr<float>();
  params.ray_directions = all_directions.data_ptr<float>();
  params.num_rays = static_cast<int>(total_rays);
  params.traversable = gas.traversable;
  params.t_out = t_out.data_ptr<float>();
  params.hit_out = hit_out.data_ptr<int>();
  params.geometry_id_out = geom_id_out.data_ptr<int64_t>();
  params.primitive_id_out = prim_id_out.data_ptr<int64_t>();
  params.u_out = u_out.data_ptr<float>();
  params.v_out = v_out.data_ptr<float>();

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
      static_cast<unsigned int>(total_rays), 1, 1));

  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  cudaFree(reinterpret_cast<void*>(d_params));

  // Reshape to [num_queries, NUM_DIRS] and find minimum t per query
  at::Tensor t_reshaped = t_out.reshape({num_queries, NUM_DIRS});
  auto [min_t, min_idx] = t_reshaped.min(/*dim=*/1);

  // Gather results at the minimum-t direction index
  at::Tensor flat_idx = at::arange(num_queries, float_opts.dtype(at::kLong)) *
                             NUM_DIRS +
                         min_idx;

  at::Tensor best_t = t_out.index_select(0, flat_idx);
  at::Tensor best_geom = geom_id_out.index_select(0, flat_idx);
  at::Tensor best_prim = prim_id_out.index_select(0, flat_idx);
  at::Tensor best_u = u_out.index_select(0, flat_idx);
  at::Tensor best_v = v_out.index_select(0, flat_idx);

  // Compute closest points: origin + t * direction
  at::Tensor best_dirs = all_directions.reshape({num_queries, NUM_DIRS, 3});
  // Gather the best direction for each query
  at::Tensor dir_idx = min_idx.unsqueeze(-1).unsqueeze(-1).expand({num_queries, 1, 3});
  at::Tensor best_dir = best_dirs.gather(1, dir_idx).squeeze(1);

  at::Tensor closest_pts =
      queries_flat + best_t.unsqueeze(-1) * best_dir;

  // Build output shapes
  std::vector<int64_t> point_shape = batch_shape;
  point_shape.push_back(3);

  return std::make_tuple(
      closest_pts.reshape(point_shape),
      best_t.reshape(batch_shape),
      best_geom.reshape(batch_shape),
      best_prim.reshape(batch_shape),
      best_u.reshape(batch_shape),
      best_v.reshape(batch_shape));
}

}  // namespace torchscience::optix::geometry

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
  m.impl(
      "bvh_closest_point",
      torchscience::optix::geometry::bvh_closest_point);
}

#endif  // TORCHSCIENCE_OPTIX
