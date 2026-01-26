// src/torchscience/csrc/meta/geometry/intersection/ray_plane.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::intersection {

/**
 * Meta implementation of ray-plane intersection for shape inference.
 *
 * @param origins Ray origins, shape (..., 3)
 * @param directions Ray directions, shape (..., 3)
 * @param plane_normals Plane normals, shape (..., 3)
 * @param plane_offsets Plane offsets, shape (...)
 * @return Tuple of (t, hit_point, normal, uv, hit)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_plane(
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& plane_normals,
    const at::Tensor& plane_offsets) {
  // Input validation
  TORCH_CHECK(
      origins.dim() >= 1 && origins.size(-1) == 3,
      "ray_plane: origins must have shape (..., 3), got ",
      origins.sizes());
  TORCH_CHECK(
      directions.dim() >= 1 && directions.size(-1) == 3,
      "ray_plane: directions must have shape (..., 3), got ",
      directions.sizes());
  TORCH_CHECK(
      plane_normals.dim() >= 1 && plane_normals.size(-1) == 3,
      "ray_plane: plane_normals must have shape (..., 3), got ",
      plane_normals.sizes());
  TORCH_CHECK(
      plane_offsets.dim() >= 1,
      "ray_plane: plane_offsets must have at least 1 dimension, got ",
      plane_offsets.dim());

  // Compute ray batch shape (all dims except last)
  std::vector<int64_t> ray_batch;
  for (int64_t i = 0; i < origins.dim() - 1; ++i) {
    ray_batch.push_back(origins.size(i));
  }

  // Compute plane batch shape
  std::vector<int64_t> plane_batch;
  for (int64_t i = 0; i < plane_offsets.dim(); ++i) {
    plane_batch.push_back(plane_offsets.size(i));
  }

  // Output batch shape: ray_batch + plane_batch
  std::vector<int64_t> out_batch = ray_batch;
  out_batch.insert(out_batch.end(), plane_batch.begin(), plane_batch.end());

  // t shape: out_batch
  std::vector<int64_t> t_shape = out_batch;

  // hit_point, normal shape: out_batch + [3]
  std::vector<int64_t> point_shape = out_batch;
  point_shape.push_back(3);

  // uv shape: out_batch + [2]
  std::vector<int64_t> uv_shape = out_batch;
  uv_shape.push_back(2);

  auto options = origins.options();

  return std::make_tuple(
      at::empty(t_shape, options),
      at::empty(point_shape, options),
      at::empty(point_shape, options),
      at::empty(uv_shape, options),
      at::empty(t_shape, options.dtype(at::kBool)));
}

/**
 * Meta implementation of ray-plane intersection backward pass for shape inference.
 *
 * @param grad_t Gradient of loss w.r.t. t
 * @param grad_hit_point Gradient of loss w.r.t. hit_point
 * @param grad_normal Gradient of loss w.r.t. normal
 * @param grad_uv Gradient of loss w.r.t. uv
 * @param origins Ray origins (saved from forward)
 * @param directions Ray directions (saved from forward)
 * @param plane_normals Plane normals (saved from forward)
 * @param plane_offsets Plane offsets (saved from forward)
 * @param t Intersection parameter (saved from forward)
 * @param hit Hit flags (saved from forward)
 * @return Tuple of (grad_origins, grad_directions, grad_plane_normals, grad_plane_offsets)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_plane_backward(
    const at::Tensor& grad_t,
    const at::Tensor& grad_hit_point,
    const at::Tensor& grad_normal,
    const at::Tensor& grad_uv,
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& plane_normals,
    const at::Tensor& plane_offsets,
    const at::Tensor& t,
    const at::Tensor& hit) {
  return std::make_tuple(
      at::empty_like(origins),
      at::empty_like(directions),
      at::empty_like(plane_normals),
      at::empty_like(plane_offsets));
}

}  // namespace torchscience::meta::geometry::intersection

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("ray_plane", torchscience::meta::geometry::intersection::ray_plane);
  m.impl(
      "ray_plane_backward",
      torchscience::meta::geometry::intersection::ray_plane_backward);
}
