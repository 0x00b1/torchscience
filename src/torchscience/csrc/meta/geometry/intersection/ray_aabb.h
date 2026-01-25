// src/torchscience/csrc/meta/geometry/intersection/ray_aabb.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::intersection {

/**
 * Meta implementation of ray-AABB intersection for shape inference.
 *
 * @param origins Ray origins, shape (*, 3)
 * @param directions Ray directions, shape (*, 3)
 * @param box_min AABB minimum corners, shape (*, 3)
 * @param box_max AABB maximum corners, shape (*, 3)
 * @return Tuple of (t, hit_point, normal, uv, hit)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_aabb(
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& box_min,
    const at::Tensor& box_max) {
  // Input validation
  TORCH_CHECK(
      origins.dim() >= 1 && origins.size(-1) == 3,
      "ray_aabb: origins must have shape (*, 3), got ",
      origins.sizes());
  TORCH_CHECK(
      directions.dim() >= 1 && directions.size(-1) == 3,
      "ray_aabb: directions must have shape (*, 3), got ",
      directions.sizes());
  TORCH_CHECK(
      box_min.dim() >= 1 && box_min.size(-1) == 3,
      "ray_aabb: box_min must have shape (*, 3), got ",
      box_min.sizes());
  TORCH_CHECK(
      box_max.dim() >= 1 && box_max.size(-1) == 3,
      "ray_aabb: box_max must have shape (*, 3), got ",
      box_max.sizes());

  // Compute ray batch shape (all dims except last)
  std::vector<int64_t> ray_batch;
  for (int64_t i = 0; i < origins.dim() - 1; ++i) {
    ray_batch.push_back(origins.size(i));
  }

  // Compute box batch shape (all dims except last)
  std::vector<int64_t> box_batch;
  for (int64_t i = 0; i < box_min.dim() - 1; ++i) {
    box_batch.push_back(box_min.size(i));
  }

  // Output batch shape: ray_batch + box_batch
  std::vector<int64_t> out_batch = ray_batch;
  out_batch.insert(out_batch.end(), box_batch.begin(), box_batch.end());

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
 * Meta implementation of ray-AABB intersection backward pass for shape inference.
 *
 * @param grad_t Gradient of loss w.r.t. t
 * @param grad_hit_point Gradient of loss w.r.t. hit_point
 * @param grad_normal Gradient of loss w.r.t. normal
 * @param grad_uv Gradient of loss w.r.t. uv
 * @param origins Ray origins (saved from forward)
 * @param directions Ray directions (saved from forward)
 * @param box_min AABB minimum corners (saved from forward)
 * @param box_max AABB maximum corners (saved from forward)
 * @param t Intersection parameter (saved from forward)
 * @param hit Hit flags (saved from forward)
 * @return Tuple of (grad_origins, grad_directions, grad_box_min, grad_box_max)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_aabb_backward(
    const at::Tensor& grad_t,
    const at::Tensor& grad_hit_point,
    const at::Tensor& grad_normal,
    const at::Tensor& grad_uv,
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& box_min,
    const at::Tensor& box_max,
    const at::Tensor& t,
    const at::Tensor& hit) {
  return std::make_tuple(
      at::empty_like(origins),
      at::empty_like(directions),
      at::empty_like(box_min),
      at::empty_like(box_max));
}

}  // namespace torchscience::meta::geometry::intersection

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("ray_aabb", torchscience::meta::geometry::intersection::ray_aabb);
  m.impl(
      "ray_aabb_backward",
      torchscience::meta::geometry::intersection::ray_aabb_backward);
}
