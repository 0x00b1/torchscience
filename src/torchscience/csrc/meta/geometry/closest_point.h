// src/torchscience/csrc/meta/geometry/closest_point.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry {

/**
 * Meta implementation of BVH closest point query for shape inference.
 *
 * @param scene_handle BVH scene handle
 * @param query_points Query points, shape (..., 3)
 * @return Tuple of (point, distance, geometry_id, primitive_id, u, v)
 */
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

  // Compute batch shape (all dims except last)
  std::vector<int64_t> batch_shape;
  for (int64_t i = 0; i < query_points.dim() - 1; ++i) {
    batch_shape.push_back(query_points.size(i));
  }

  // Point shape: batch_shape + [3]
  std::vector<int64_t> point_shape = batch_shape;
  point_shape.push_back(3);

  auto float_options = query_points.options().dtype(at::kFloat);

  return std::make_tuple(
      at::empty(point_shape, float_options),
      at::empty(batch_shape, float_options),
      at::empty(batch_shape, query_points.options().dtype(at::kLong)),
      at::empty(batch_shape, query_points.options().dtype(at::kLong)),
      at::empty(batch_shape, float_options),
      at::empty(batch_shape, float_options));
}

}  // namespace torchscience::meta::geometry

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl(
      "bvh_closest_point",
      torchscience::meta::geometry::bvh_closest_point);
}
