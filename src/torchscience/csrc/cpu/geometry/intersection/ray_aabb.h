// src/torchscience/csrc/cpu/geometry/intersection/ray_aabb.h
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/intersection/ray_aabb.h"
#include "../../../kernel/geometry/intersection/ray_aabb_backward.h"

namespace torchscience::cpu::geometry::intersection {

/**
 * CPU implementation of ray-AABB intersection.
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
      "ray_aabb: origins must have shape (..., 3), got ",
      origins.sizes());
  TORCH_CHECK(
      directions.dim() >= 1 && directions.size(-1) == 3,
      "ray_aabb: directions must have shape (..., 3), got ",
      directions.sizes());
  TORCH_CHECK(
      box_min.dim() >= 1 && box_min.size(-1) == 3,
      "ray_aabb: box_min must have shape (..., 3), got ",
      box_min.sizes());
  TORCH_CHECK(
      box_max.dim() >= 1 && box_max.size(-1) == 3,
      "ray_aabb: box_max must have shape (..., 3), got ",
      box_max.sizes());
  TORCH_CHECK(
      origins.sizes() == directions.sizes(),
      "ray_aabb: origins and directions must have matching shapes");
  TORCH_CHECK(
      box_min.sizes() == box_max.sizes(),
      "ray_aabb: box_min and box_max must have matching shapes");

  // Flatten ray batch dimensions
  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  // Flatten box batch dimensions (all dims except last)
  auto box_batch_shape = box_min.sizes().slice(0, box_min.dim() - 1);
  int64_t num_boxes = 1;
  for (auto s : box_batch_shape) {
    num_boxes *= s;
  }

  // Make contiguous
  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor box_min_flat = box_min.reshape({num_boxes, 3}).contiguous();
  at::Tensor box_max_flat = box_max.reshape({num_boxes, 3}).contiguous();

  // Compute output batch size (broadcast rays x boxes)
  int64_t batch_size = num_rays * num_boxes;

  // Allocate output tensors
  auto options = origins.options();
  at::Tensor t_out = at::empty({batch_size}, options);
  at::Tensor hit_point_out = at::empty({batch_size, 3}, options);
  at::Tensor normal_out = at::empty({batch_size, 3}, options);
  at::Tensor uv_out = at::empty({batch_size, 2}, options);
  at::Tensor hit_out = at::empty({batch_size}, options.dtype(at::kBool));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      origins.scalar_type(),
      "ray_aabb_cpu",
      [&]() {
        const scalar_t* origins_ptr = origins_flat.data_ptr<scalar_t>();
        const scalar_t* directions_ptr = directions_flat.data_ptr<scalar_t>();
        const scalar_t* box_min_ptr = box_min_flat.data_ptr<scalar_t>();
        const scalar_t* box_max_ptr = box_max_flat.data_ptr<scalar_t>();

        scalar_t* t_ptr = t_out.data_ptr<scalar_t>();
        scalar_t* hit_point_ptr = hit_point_out.data_ptr<scalar_t>();
        scalar_t* normal_ptr = normal_out.data_ptr<scalar_t>();
        scalar_t* uv_ptr = uv_out.data_ptr<scalar_t>();
        bool* hit_ptr = hit_out.data_ptr<bool>();

        at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            // Compute ray and box indices from flat index
            int64_t ray_idx = idx / num_boxes;
            int64_t box_idx = idx % num_boxes;

            // Get ray data
            scalar_t ox = origins_ptr[ray_idx * 3 + 0];
            scalar_t oy = origins_ptr[ray_idx * 3 + 1];
            scalar_t oz = origins_ptr[ray_idx * 3 + 2];
            scalar_t dx = directions_ptr[ray_idx * 3 + 0];
            scalar_t dy = directions_ptr[ray_idx * 3 + 1];
            scalar_t dz = directions_ptr[ray_idx * 3 + 2];

            // Get box data
            scalar_t bmin_x = box_min_ptr[box_idx * 3 + 0];
            scalar_t bmin_y = box_min_ptr[box_idx * 3 + 1];
            scalar_t bmin_z = box_min_ptr[box_idx * 3 + 2];
            scalar_t bmax_x = box_max_ptr[box_idx * 3 + 0];
            scalar_t bmax_y = box_max_ptr[box_idx * 3 + 1];
            scalar_t bmax_z = box_max_ptr[box_idx * 3 + 2];

            // Output variables
            scalar_t t, hx, hy, hz, out_nx, out_ny, out_nz, u, v, hit_val;

            // Call kernel
            kernel::geometry::intersection::ray_aabb_kernel<scalar_t>(
                ox, oy, oz, dx, dy, dz,
                bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z,
                t, hx, hy, hz, out_nx, out_ny, out_nz, u, v, hit_val);

            // Store outputs
            t_ptr[idx] = t;
            hit_point_ptr[idx * 3 + 0] = hx;
            hit_point_ptr[idx * 3 + 1] = hy;
            hit_point_ptr[idx * 3 + 2] = hz;
            normal_ptr[idx * 3 + 0] = out_nx;
            normal_ptr[idx * 3 + 1] = out_ny;
            normal_ptr[idx * 3 + 2] = out_nz;
            uv_ptr[idx * 2 + 0] = u;
            uv_ptr[idx * 2 + 1] = v;
            hit_ptr[idx] = hit_val > scalar_t(0.5);
          }
        });
      });

  // Compute output shape: ray_batch + box_batch
  std::vector<int64_t> out_batch_shape;
  for (auto s : ray_batch_shape) {
    out_batch_shape.push_back(s);
  }
  for (auto s : box_batch_shape) {
    out_batch_shape.push_back(s);
  }

  // Reshape outputs
  std::vector<int64_t> t_shape = out_batch_shape;
  std::vector<int64_t> hit_point_shape = out_batch_shape;
  hit_point_shape.push_back(3);
  std::vector<int64_t> normal_shape = out_batch_shape;
  normal_shape.push_back(3);
  std::vector<int64_t> uv_shape = out_batch_shape;
  uv_shape.push_back(2);

  return std::make_tuple(
      t_out.reshape(t_shape),
      hit_point_out.reshape(hit_point_shape),
      normal_out.reshape(normal_shape),
      uv_out.reshape(uv_shape),
      hit_out.reshape(t_shape));
}

/**
 * CPU implementation of ray-AABB intersection backward pass.
 *
 * @param grad_t Gradient of loss w.r.t. t, shape (batch,)
 * @param grad_hit_point Gradient of loss w.r.t. hit_point, shape (batch, 3)
 * @param grad_normal Gradient of loss w.r.t. normal, shape (batch, 3)
 * @param grad_uv Gradient of loss w.r.t. uv, shape (batch, 2)
 * @param origins Ray origins (saved from forward), shape (*, 3)
 * @param directions Ray directions (saved from forward), shape (*, 3)
 * @param box_min AABB minimum corners (saved from forward), shape (*, 3)
 * @param box_max AABB maximum corners (saved from forward), shape (*, 3)
 * @param t Intersection parameter (saved from forward), shape (batch,)
 * @param hit Hit flags (saved from forward), shape (batch,)
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
  // Flatten ray batch dimensions
  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  // Flatten box batch dimensions
  auto box_batch_shape = box_min.sizes().slice(0, box_min.dim() - 1);
  int64_t num_boxes = 1;
  for (auto s : box_batch_shape) {
    num_boxes *= s;
  }

  int64_t batch_size = num_rays * num_boxes;

  // Make inputs contiguous
  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor box_min_flat = box_min.reshape({num_boxes, 3}).contiguous();
  at::Tensor box_max_flat = box_max.reshape({num_boxes, 3}).contiguous();

  at::Tensor grad_t_flat = grad_t.reshape({batch_size}).contiguous();
  at::Tensor grad_hit_point_flat = grad_hit_point.reshape({batch_size, 3}).contiguous();
  at::Tensor grad_normal_flat = grad_normal.reshape({batch_size, 3}).contiguous();
  at::Tensor grad_uv_flat = grad_uv.reshape({batch_size, 2}).contiguous();
  at::Tensor t_flat = t.reshape({batch_size}).contiguous();
  at::Tensor hit_flat = hit.reshape({batch_size}).contiguous();

  // Allocate output gradient tensors (accumulated over broadcast)
  auto options = origins.options();
  at::Tensor grad_origins = at::zeros({num_rays, 3}, options);
  at::Tensor grad_directions = at::zeros({num_rays, 3}, options);
  at::Tensor grad_box_min_out = at::zeros({num_boxes, 3}, options);
  at::Tensor grad_box_max_out = at::zeros({num_boxes, 3}, options);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      origins.scalar_type(),
      "ray_aabb_backward_cpu",
      [&]() {
        const scalar_t* origins_ptr = origins_flat.data_ptr<scalar_t>();
        const scalar_t* directions_ptr = directions_flat.data_ptr<scalar_t>();
        const scalar_t* box_min_ptr = box_min_flat.data_ptr<scalar_t>();
        const scalar_t* box_max_ptr = box_max_flat.data_ptr<scalar_t>();

        const scalar_t* grad_t_ptr = grad_t_flat.data_ptr<scalar_t>();
        const scalar_t* grad_hit_point_ptr = grad_hit_point_flat.data_ptr<scalar_t>();
        const scalar_t* grad_normal_ptr = grad_normal_flat.data_ptr<scalar_t>();
        const scalar_t* grad_uv_ptr = grad_uv_flat.data_ptr<scalar_t>();
        const scalar_t* t_ptr = t_flat.data_ptr<scalar_t>();

        // Need to handle hit tensor which is bool
        const bool* hit_ptr = hit_flat.data_ptr<bool>();

        // For thread-safe accumulation, we compute per-element then reduce
        // Allocate per-element gradient arrays
        at::Tensor grad_origins_elem = at::zeros({batch_size, 3}, options);
        at::Tensor grad_directions_elem = at::zeros({batch_size, 3}, options);
        at::Tensor grad_box_min_elem = at::zeros({batch_size, 3}, options);
        at::Tensor grad_box_max_elem = at::zeros({batch_size, 3}, options);

        scalar_t* grad_origins_elem_ptr = grad_origins_elem.data_ptr<scalar_t>();
        scalar_t* grad_directions_elem_ptr = grad_directions_elem.data_ptr<scalar_t>();
        scalar_t* grad_box_min_elem_ptr = grad_box_min_elem.data_ptr<scalar_t>();
        scalar_t* grad_box_max_elem_ptr = grad_box_max_elem.data_ptr<scalar_t>();

        at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            // Compute ray and box indices
            int64_t ray_idx = idx / num_boxes;
            int64_t box_idx = idx % num_boxes;

            // Get ray data
            scalar_t ox = origins_ptr[ray_idx * 3 + 0];
            scalar_t oy = origins_ptr[ray_idx * 3 + 1];
            scalar_t oz = origins_ptr[ray_idx * 3 + 2];
            scalar_t dx = directions_ptr[ray_idx * 3 + 0];
            scalar_t dy = directions_ptr[ray_idx * 3 + 1];
            scalar_t dz = directions_ptr[ray_idx * 3 + 2];

            // Get box data
            scalar_t bmin_x = box_min_ptr[box_idx * 3 + 0];
            scalar_t bmin_y = box_min_ptr[box_idx * 3 + 1];
            scalar_t bmin_z = box_min_ptr[box_idx * 3 + 2];
            scalar_t bmax_x = box_max_ptr[box_idx * 3 + 0];
            scalar_t bmax_y = box_max_ptr[box_idx * 3 + 1];
            scalar_t bmax_z = box_max_ptr[box_idx * 3 + 2];

            // Get upstream gradients
            scalar_t gt = grad_t_ptr[idx];
            scalar_t ghx = grad_hit_point_ptr[idx * 3 + 0];
            scalar_t ghy = grad_hit_point_ptr[idx * 3 + 1];
            scalar_t ghz = grad_hit_point_ptr[idx * 3 + 2];
            scalar_t gnx = grad_normal_ptr[idx * 3 + 0];
            scalar_t gny = grad_normal_ptr[idx * 3 + 1];
            scalar_t gnz = grad_normal_ptr[idx * 3 + 2];
            scalar_t gu = grad_uv_ptr[idx * 2 + 0];
            scalar_t gv = grad_uv_ptr[idx * 2 + 1];

            // Get saved tensors
            scalar_t t_val = t_ptr[idx];
            scalar_t hit_val = hit_ptr[idx] ? scalar_t(1) : scalar_t(0);

            // Output gradient variables
            scalar_t go_x, go_y, go_z;
            scalar_t gd_x, gd_y, gd_z;
            scalar_t gbmin_x, gbmin_y, gbmin_z;
            scalar_t gbmax_x, gbmax_y, gbmax_z;

            // Call backward kernel
            kernel::geometry::intersection::ray_aabb_backward_kernel<scalar_t>(
                gt, ghx, ghy, ghz, gnx, gny, gnz, gu, gv,
                ox, oy, oz, dx, dy, dz,
                bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z,
                t_val, hit_val,
                go_x, go_y, go_z, gd_x, gd_y, gd_z,
                gbmin_x, gbmin_y, gbmin_z, gbmax_x, gbmax_y, gbmax_z);

            // Store per-element gradients
            grad_origins_elem_ptr[idx * 3 + 0] = go_x;
            grad_origins_elem_ptr[idx * 3 + 1] = go_y;
            grad_origins_elem_ptr[idx * 3 + 2] = go_z;
            grad_directions_elem_ptr[idx * 3 + 0] = gd_x;
            grad_directions_elem_ptr[idx * 3 + 1] = gd_y;
            grad_directions_elem_ptr[idx * 3 + 2] = gd_z;
            grad_box_min_elem_ptr[idx * 3 + 0] = gbmin_x;
            grad_box_min_elem_ptr[idx * 3 + 1] = gbmin_y;
            grad_box_min_elem_ptr[idx * 3 + 2] = gbmin_z;
            grad_box_max_elem_ptr[idx * 3 + 0] = gbmax_x;
            grad_box_max_elem_ptr[idx * 3 + 1] = gbmax_y;
            grad_box_max_elem_ptr[idx * 3 + 2] = gbmax_z;
          }
        });

        // Reduce gradients: sum over broadcast dimension
        // grad_origins: sum over boxes for each ray
        // grad_box_min/max: sum over rays for each box
        auto grad_origins_acc = grad_origins.accessor<scalar_t, 2>();
        auto grad_directions_acc = grad_directions.accessor<scalar_t, 2>();
        auto grad_box_min_acc = grad_box_min_out.accessor<scalar_t, 2>();
        auto grad_box_max_acc = grad_box_max_out.accessor<scalar_t, 2>();

        // Sum gradients (sequential to avoid race conditions)
        for (int64_t idx = 0; idx < batch_size; ++idx) {
          int64_t ray_idx = idx / num_boxes;
          int64_t box_idx = idx % num_boxes;

          // Accumulate ray gradients
          grad_origins_acc[ray_idx][0] += grad_origins_elem_ptr[idx * 3 + 0];
          grad_origins_acc[ray_idx][1] += grad_origins_elem_ptr[idx * 3 + 1];
          grad_origins_acc[ray_idx][2] += grad_origins_elem_ptr[idx * 3 + 2];
          grad_directions_acc[ray_idx][0] += grad_directions_elem_ptr[idx * 3 + 0];
          grad_directions_acc[ray_idx][1] += grad_directions_elem_ptr[idx * 3 + 1];
          grad_directions_acc[ray_idx][2] += grad_directions_elem_ptr[idx * 3 + 2];

          // Accumulate box gradients
          grad_box_min_acc[box_idx][0] += grad_box_min_elem_ptr[idx * 3 + 0];
          grad_box_min_acc[box_idx][1] += grad_box_min_elem_ptr[idx * 3 + 1];
          grad_box_min_acc[box_idx][2] += grad_box_min_elem_ptr[idx * 3 + 2];
          grad_box_max_acc[box_idx][0] += grad_box_max_elem_ptr[idx * 3 + 0];
          grad_box_max_acc[box_idx][1] += grad_box_max_elem_ptr[idx * 3 + 1];
          grad_box_max_acc[box_idx][2] += grad_box_max_elem_ptr[idx * 3 + 2];
        }
      });

  // Reshape to original input shapes
  std::vector<int64_t> origins_shape(ray_batch_shape.begin(), ray_batch_shape.end());
  origins_shape.push_back(3);

  std::vector<int64_t> box_min_shape(box_batch_shape.begin(), box_batch_shape.end());
  box_min_shape.push_back(3);

  return std::make_tuple(
      grad_origins.reshape(origins_shape),
      grad_directions.reshape(origins_shape),
      grad_box_min_out.reshape(box_min_shape),
      grad_box_max_out.reshape(box_min_shape));
}

}  // namespace torchscience::cpu::geometry::intersection

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("ray_aabb", torchscience::cpu::geometry::intersection::ray_aabb);
  m.impl(
      "ray_aabb_backward",
      torchscience::cpu::geometry::intersection::ray_aabb_backward);
}
