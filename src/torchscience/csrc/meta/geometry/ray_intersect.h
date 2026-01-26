// src/torchscience/csrc/meta/geometry/ray_intersect.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry {

/**
 * Meta implementation of BVH ray intersection for shape inference.
 *
 * @param scene_handle BVH scene handle
 * @param origins Ray origins, shape (..., 3)
 * @param directions Ray directions, shape (..., 3)
 * @return Tuple of (t, hit, geometry_id, primitive_id, u, v)
 */
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

  // Compute batch shape (all dims except last)
  std::vector<int64_t> batch_shape;
  for (int64_t i = 0; i < origins.dim() - 1; ++i) {
    batch_shape.push_back(origins.size(i));
  }

  auto float_options = origins.options().dtype(at::kFloat);

  return std::make_tuple(
      at::empty(batch_shape, float_options),
      at::empty(batch_shape, origins.options().dtype(at::kBool)),
      at::empty(batch_shape, origins.options().dtype(at::kLong)),
      at::empty(batch_shape, origins.options().dtype(at::kLong)),
      at::empty(batch_shape, float_options),
      at::empty(batch_shape, float_options));
}

}  // namespace torchscience::meta::geometry

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl(
      "bvh_ray_intersect",
      torchscience::meta::geometry::bvh_ray_intersect);
}
