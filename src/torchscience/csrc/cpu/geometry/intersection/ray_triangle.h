// src/torchscience/csrc/cpu/geometry/intersection/ray_triangle.h
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/intersection/ray_triangle.h"
#include "../../../kernel/geometry/intersection/ray_triangle_backward.h"

namespace torchscience::cpu::geometry::intersection {

/**
 * CPU implementation of ray-triangle intersection (Moller-Trumbore algorithm).
 *
 * @param origins Ray origins, shape (*, 3)
 * @param directions Ray directions, shape (*, 3)
 * @param v0 Triangle vertex V0, shape (*, 3)
 * @param v1 Triangle vertex V1, shape (*, 3)
 * @param v2 Triangle vertex V2, shape (*, 3)
 * @return Tuple of (t, hit_point, normal, uv, hit)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_triangle(
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& v0,
    const at::Tensor& v1,
    const at::Tensor& v2) {
  // Input validation
  TORCH_CHECK(
      origins.dim() >= 1 && origins.size(-1) == 3,
      "ray_triangle: origins must have shape (..., 3), got ",
      origins.sizes());
  TORCH_CHECK(
      directions.dim() >= 1 && directions.size(-1) == 3,
      "ray_triangle: directions must have shape (..., 3), got ",
      directions.sizes());
  TORCH_CHECK(
      origins.sizes() == directions.sizes(),
      "ray_triangle: origins and directions must have matching shapes");
  TORCH_CHECK(
      v0.dim() >= 1 && v0.size(-1) == 3,
      "ray_triangle: v0 must have shape (..., 3), got ",
      v0.sizes());
  TORCH_CHECK(
      v1.dim() >= 1 && v1.size(-1) == 3,
      "ray_triangle: v1 must have shape (..., 3), got ",
      v1.sizes());
  TORCH_CHECK(
      v2.dim() >= 1 && v2.size(-1) == 3,
      "ray_triangle: v2 must have shape (..., 3), got ",
      v2.sizes());
  TORCH_CHECK(
      v0.sizes() == v1.sizes() && v1.sizes() == v2.sizes(),
      "ray_triangle: v0, v1, and v2 must have matching shapes");

  // Flatten ray batch dimensions
  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  // Flatten triangle batch dimensions
  auto triangle_batch_shape = v0.sizes().slice(0, v0.dim() - 1);
  int64_t num_triangles = 1;
  for (auto s : triangle_batch_shape) {
    num_triangles *= s;
  }

  // Make contiguous
  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor v0_flat = v0.reshape({num_triangles, 3}).contiguous();
  at::Tensor v1_flat = v1.reshape({num_triangles, 3}).contiguous();
  at::Tensor v2_flat = v2.reshape({num_triangles, 3}).contiguous();

  // Compute output batch size (broadcast rays x triangles)
  int64_t batch_size = num_rays * num_triangles;

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
      "ray_triangle_cpu",
      [&]() {
        const scalar_t* origins_ptr = origins_flat.data_ptr<scalar_t>();
        const scalar_t* directions_ptr = directions_flat.data_ptr<scalar_t>();
        const scalar_t* v0_ptr = v0_flat.data_ptr<scalar_t>();
        const scalar_t* v1_ptr = v1_flat.data_ptr<scalar_t>();
        const scalar_t* v2_ptr = v2_flat.data_ptr<scalar_t>();

        scalar_t* t_ptr = t_out.data_ptr<scalar_t>();
        scalar_t* hit_point_ptr = hit_point_out.data_ptr<scalar_t>();
        scalar_t* normal_ptr = normal_out.data_ptr<scalar_t>();
        scalar_t* uv_ptr = uv_out.data_ptr<scalar_t>();
        bool* hit_ptr = hit_out.data_ptr<bool>();

        at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            // Compute ray and triangle indices from flat index
            int64_t ray_idx = idx / num_triangles;
            int64_t tri_idx = idx % num_triangles;

            // Get ray data
            scalar_t ox = origins_ptr[ray_idx * 3 + 0];
            scalar_t oy = origins_ptr[ray_idx * 3 + 1];
            scalar_t oz = origins_ptr[ray_idx * 3 + 2];
            scalar_t dx = directions_ptr[ray_idx * 3 + 0];
            scalar_t dy = directions_ptr[ray_idx * 3 + 1];
            scalar_t dz = directions_ptr[ray_idx * 3 + 2];

            // Get triangle vertex data
            scalar_t v0x = v0_ptr[tri_idx * 3 + 0];
            scalar_t v0y = v0_ptr[tri_idx * 3 + 1];
            scalar_t v0z = v0_ptr[tri_idx * 3 + 2];
            scalar_t v1x = v1_ptr[tri_idx * 3 + 0];
            scalar_t v1y = v1_ptr[tri_idx * 3 + 1];
            scalar_t v1z = v1_ptr[tri_idx * 3 + 2];
            scalar_t v2x = v2_ptr[tri_idx * 3 + 0];
            scalar_t v2y = v2_ptr[tri_idx * 3 + 1];
            scalar_t v2z = v2_ptr[tri_idx * 3 + 2];

            // Output variables
            scalar_t t, hx, hy, hz, out_nx, out_ny, out_nz, u, v, hit_val;

            // Call kernel
            kernel::geometry::intersection::ray_triangle_kernel<scalar_t>(
                ox, oy, oz, dx, dy, dz,
                v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z,
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

  // Compute output shape: ray_batch + triangle_batch
  std::vector<int64_t> out_batch_shape;
  for (auto s : ray_batch_shape) {
    out_batch_shape.push_back(s);
  }
  for (auto s : triangle_batch_shape) {
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
 * CPU implementation of ray-triangle intersection backward pass.
 *
 * @param grad_t Gradient of loss w.r.t. t, shape (batch,)
 * @param grad_hit_point Gradient of loss w.r.t. hit_point, shape (batch, 3)
 * @param grad_normal Gradient of loss w.r.t. normal, shape (batch, 3)
 * @param grad_uv Gradient of loss w.r.t. uv, shape (batch, 2)
 * @param origins Ray origins (saved from forward), shape (*, 3)
 * @param directions Ray directions (saved from forward), shape (*, 3)
 * @param v0 Triangle vertex V0 (saved from forward), shape (*, 3)
 * @param v1 Triangle vertex V1 (saved from forward), shape (*, 3)
 * @param v2 Triangle vertex V2 (saved from forward), shape (*, 3)
 * @param t Intersection parameter (saved from forward), shape (batch,)
 * @param hit Hit flags (saved from forward), shape (batch,)
 * @return Tuple of (grad_origins, grad_directions, grad_v0, grad_v1, grad_v2)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_triangle_backward(
    const at::Tensor& grad_t,
    const at::Tensor& grad_hit_point,
    const at::Tensor& grad_normal,
    const at::Tensor& grad_uv,
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& v0,
    const at::Tensor& v1,
    const at::Tensor& v2,
    const at::Tensor& t,
    const at::Tensor& hit) {
  // Flatten ray batch dimensions
  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  // Flatten triangle batch dimensions
  auto triangle_batch_shape = v0.sizes().slice(0, v0.dim() - 1);
  int64_t num_triangles = 1;
  for (auto s : triangle_batch_shape) {
    num_triangles *= s;
  }

  int64_t batch_size = num_rays * num_triangles;

  // Make inputs contiguous
  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor v0_flat = v0.reshape({num_triangles, 3}).contiguous();
  at::Tensor v1_flat = v1.reshape({num_triangles, 3}).contiguous();
  at::Tensor v2_flat = v2.reshape({num_triangles, 3}).contiguous();

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
  at::Tensor grad_v0_out = at::zeros({num_triangles, 3}, options);
  at::Tensor grad_v1_out = at::zeros({num_triangles, 3}, options);
  at::Tensor grad_v2_out = at::zeros({num_triangles, 3}, options);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      origins.scalar_type(),
      "ray_triangle_backward_cpu",
      [&]() {
        const scalar_t* origins_ptr = origins_flat.data_ptr<scalar_t>();
        const scalar_t* directions_ptr = directions_flat.data_ptr<scalar_t>();
        const scalar_t* v0_ptr = v0_flat.data_ptr<scalar_t>();
        const scalar_t* v1_ptr = v1_flat.data_ptr<scalar_t>();
        const scalar_t* v2_ptr = v2_flat.data_ptr<scalar_t>();

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
        at::Tensor grad_v0_elem = at::zeros({batch_size, 3}, options);
        at::Tensor grad_v1_elem = at::zeros({batch_size, 3}, options);
        at::Tensor grad_v2_elem = at::zeros({batch_size, 3}, options);

        scalar_t* grad_origins_elem_ptr = grad_origins_elem.data_ptr<scalar_t>();
        scalar_t* grad_directions_elem_ptr = grad_directions_elem.data_ptr<scalar_t>();
        scalar_t* grad_v0_elem_ptr = grad_v0_elem.data_ptr<scalar_t>();
        scalar_t* grad_v1_elem_ptr = grad_v1_elem.data_ptr<scalar_t>();
        scalar_t* grad_v2_elem_ptr = grad_v2_elem.data_ptr<scalar_t>();

        at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            // Compute ray and triangle indices
            int64_t ray_idx = idx / num_triangles;
            int64_t tri_idx = idx % num_triangles;

            // Get ray data
            scalar_t ox = origins_ptr[ray_idx * 3 + 0];
            scalar_t oy = origins_ptr[ray_idx * 3 + 1];
            scalar_t oz = origins_ptr[ray_idx * 3 + 2];
            scalar_t dx = directions_ptr[ray_idx * 3 + 0];
            scalar_t dy = directions_ptr[ray_idx * 3 + 1];
            scalar_t dz = directions_ptr[ray_idx * 3 + 2];

            // Get triangle vertex data
            scalar_t v0x = v0_ptr[tri_idx * 3 + 0];
            scalar_t v0y = v0_ptr[tri_idx * 3 + 1];
            scalar_t v0z = v0_ptr[tri_idx * 3 + 2];
            scalar_t v1x = v1_ptr[tri_idx * 3 + 0];
            scalar_t v1y = v1_ptr[tri_idx * 3 + 1];
            scalar_t v1z = v1_ptr[tri_idx * 3 + 2];
            scalar_t v2x = v2_ptr[tri_idx * 3 + 0];
            scalar_t v2y = v2_ptr[tri_idx * 3 + 1];
            scalar_t v2z = v2_ptr[tri_idx * 3 + 2];

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
            scalar_t gv0_x, gv0_y, gv0_z;
            scalar_t gv1_x, gv1_y, gv1_z;
            scalar_t gv2_x, gv2_y, gv2_z;

            // Call backward kernel
            kernel::geometry::intersection::ray_triangle_backward_kernel<scalar_t>(
                gt, ghx, ghy, ghz, gnx, gny, gnz, gu, gv,
                ox, oy, oz, dx, dy, dz,
                v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z,
                t_val, hit_val,
                go_x, go_y, go_z, gd_x, gd_y, gd_z,
                gv0_x, gv0_y, gv0_z, gv1_x, gv1_y, gv1_z,
                gv2_x, gv2_y, gv2_z);

            // Store per-element gradients
            grad_origins_elem_ptr[idx * 3 + 0] = go_x;
            grad_origins_elem_ptr[idx * 3 + 1] = go_y;
            grad_origins_elem_ptr[idx * 3 + 2] = go_z;
            grad_directions_elem_ptr[idx * 3 + 0] = gd_x;
            grad_directions_elem_ptr[idx * 3 + 1] = gd_y;
            grad_directions_elem_ptr[idx * 3 + 2] = gd_z;
            grad_v0_elem_ptr[idx * 3 + 0] = gv0_x;
            grad_v0_elem_ptr[idx * 3 + 1] = gv0_y;
            grad_v0_elem_ptr[idx * 3 + 2] = gv0_z;
            grad_v1_elem_ptr[idx * 3 + 0] = gv1_x;
            grad_v1_elem_ptr[idx * 3 + 1] = gv1_y;
            grad_v1_elem_ptr[idx * 3 + 2] = gv1_z;
            grad_v2_elem_ptr[idx * 3 + 0] = gv2_x;
            grad_v2_elem_ptr[idx * 3 + 1] = gv2_y;
            grad_v2_elem_ptr[idx * 3 + 2] = gv2_z;
          }
        });

        // Reduce gradients: sum over broadcast dimension
        // grad_origins: sum over triangles for each ray
        // grad_v0/v1/v2: sum over rays for each triangle
        auto grad_origins_acc = grad_origins.accessor<scalar_t, 2>();
        auto grad_directions_acc = grad_directions.accessor<scalar_t, 2>();
        auto grad_v0_acc = grad_v0_out.accessor<scalar_t, 2>();
        auto grad_v1_acc = grad_v1_out.accessor<scalar_t, 2>();
        auto grad_v2_acc = grad_v2_out.accessor<scalar_t, 2>();

        // Sum gradients (sequential to avoid race conditions)
        for (int64_t idx = 0; idx < batch_size; ++idx) {
          int64_t ray_idx = idx / num_triangles;
          int64_t tri_idx = idx % num_triangles;

          // Accumulate ray gradients
          grad_origins_acc[ray_idx][0] += grad_origins_elem_ptr[idx * 3 + 0];
          grad_origins_acc[ray_idx][1] += grad_origins_elem_ptr[idx * 3 + 1];
          grad_origins_acc[ray_idx][2] += grad_origins_elem_ptr[idx * 3 + 2];
          grad_directions_acc[ray_idx][0] += grad_directions_elem_ptr[idx * 3 + 0];
          grad_directions_acc[ray_idx][1] += grad_directions_elem_ptr[idx * 3 + 1];
          grad_directions_acc[ray_idx][2] += grad_directions_elem_ptr[idx * 3 + 2];

          // Accumulate triangle vertex gradients
          grad_v0_acc[tri_idx][0] += grad_v0_elem_ptr[idx * 3 + 0];
          grad_v0_acc[tri_idx][1] += grad_v0_elem_ptr[idx * 3 + 1];
          grad_v0_acc[tri_idx][2] += grad_v0_elem_ptr[idx * 3 + 2];
          grad_v1_acc[tri_idx][0] += grad_v1_elem_ptr[idx * 3 + 0];
          grad_v1_acc[tri_idx][1] += grad_v1_elem_ptr[idx * 3 + 1];
          grad_v1_acc[tri_idx][2] += grad_v1_elem_ptr[idx * 3 + 2];
          grad_v2_acc[tri_idx][0] += grad_v2_elem_ptr[idx * 3 + 0];
          grad_v2_acc[tri_idx][1] += grad_v2_elem_ptr[idx * 3 + 1];
          grad_v2_acc[tri_idx][2] += grad_v2_elem_ptr[idx * 3 + 2];
        }
      });

  // Reshape to original input shapes
  std::vector<int64_t> origins_shape(ray_batch_shape.begin(), ray_batch_shape.end());
  origins_shape.push_back(3);

  std::vector<int64_t> v_shape(triangle_batch_shape.begin(), triangle_batch_shape.end());
  v_shape.push_back(3);

  return std::make_tuple(
      grad_origins.reshape(origins_shape),
      grad_directions.reshape(origins_shape),
      grad_v0_out.reshape(v_shape),
      grad_v1_out.reshape(v_shape),
      grad_v2_out.reshape(v_shape));
}

}  // namespace torchscience::cpu::geometry::intersection

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("ray_triangle", torchscience::cpu::geometry::intersection::ray_triangle);
  m.impl(
      "ray_triangle_backward",
      torchscience::cpu::geometry::intersection::ray_triangle_backward);
}
