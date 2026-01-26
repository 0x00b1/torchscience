// src/torchscience/csrc/cpu/geometry/intersection/ray_sphere.h
#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/geometry/intersection/ray_sphere.h"
#include "../../../kernel/geometry/intersection/ray_sphere_backward.h"

namespace torchscience::cpu::geometry::intersection {

/**
 * CPU implementation of ray-sphere intersection.
 *
 * @param origins Ray origins, shape (*, 3)
 * @param directions Ray directions, shape (*, 3)
 * @param centers Sphere centers, shape (*, 3)
 * @param radii Sphere radii, shape (*)
 * @return Tuple of (t, hit_point, normal, uv, hit)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_sphere(
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& centers,
    const at::Tensor& radii) {
  // Input validation
  TORCH_CHECK(
      origins.dim() >= 1 && origins.size(-1) == 3,
      "ray_sphere: origins must have shape (..., 3), got ",
      origins.sizes());
  TORCH_CHECK(
      directions.dim() >= 1 && directions.size(-1) == 3,
      "ray_sphere: directions must have shape (..., 3), got ",
      directions.sizes());
  TORCH_CHECK(
      centers.dim() >= 1 && centers.size(-1) == 3,
      "ray_sphere: centers must have shape (..., 3), got ",
      centers.sizes());
  TORCH_CHECK(
      radii.dim() >= 1,
      "ray_sphere: radii must have at least 1 dimension, got ",
      radii.dim());
  TORCH_CHECK(
      origins.sizes() == directions.sizes(),
      "ray_sphere: origins and directions must have matching shapes");

  // Flatten ray batch dimensions
  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  // Flatten sphere batch dimensions
  auto sphere_batch_shape = radii.sizes();
  int64_t num_spheres = 1;
  for (auto s : sphere_batch_shape) {
    num_spheres *= s;
  }

  // Check centers batch matches radii
  auto centers_batch = centers.sizes().slice(0, centers.dim() - 1);
  int64_t num_centers = 1;
  for (auto s : centers_batch) {
    num_centers *= s;
  }
  TORCH_CHECK(
      num_spheres == num_centers,
      "ray_sphere: centers and radii batch sizes must match");

  // Make contiguous
  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor centers_flat = centers.reshape({num_spheres, 3}).contiguous();
  at::Tensor radii_flat = radii.reshape({num_spheres}).contiguous();

  // Compute output batch size (broadcast rays x spheres)
  int64_t batch_size = num_rays * num_spheres;

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
      "ray_sphere_cpu",
      [&]() {
        const scalar_t* origins_ptr = origins_flat.data_ptr<scalar_t>();
        const scalar_t* directions_ptr = directions_flat.data_ptr<scalar_t>();
        const scalar_t* centers_ptr = centers_flat.data_ptr<scalar_t>();
        const scalar_t* radii_ptr = radii_flat.data_ptr<scalar_t>();

        scalar_t* t_ptr = t_out.data_ptr<scalar_t>();
        scalar_t* hit_point_ptr = hit_point_out.data_ptr<scalar_t>();
        scalar_t* normal_ptr = normal_out.data_ptr<scalar_t>();
        scalar_t* uv_ptr = uv_out.data_ptr<scalar_t>();
        bool* hit_ptr = hit_out.data_ptr<bool>();

        at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            // Compute ray and sphere indices from flat index
            int64_t ray_idx = idx / num_spheres;
            int64_t sphere_idx = idx % num_spheres;

            // Get ray data
            scalar_t ox = origins_ptr[ray_idx * 3 + 0];
            scalar_t oy = origins_ptr[ray_idx * 3 + 1];
            scalar_t oz = origins_ptr[ray_idx * 3 + 2];
            scalar_t dx = directions_ptr[ray_idx * 3 + 0];
            scalar_t dy = directions_ptr[ray_idx * 3 + 1];
            scalar_t dz = directions_ptr[ray_idx * 3 + 2];

            // Get sphere data
            scalar_t cx = centers_ptr[sphere_idx * 3 + 0];
            scalar_t cy = centers_ptr[sphere_idx * 3 + 1];
            scalar_t cz = centers_ptr[sphere_idx * 3 + 2];
            scalar_t r = radii_ptr[sphere_idx];

            // Output variables
            scalar_t t, hx, hy, hz, out_nx, out_ny, out_nz, u, v, hit_val;

            // Call kernel
            kernel::geometry::intersection::ray_sphere_kernel<scalar_t>(
                ox, oy, oz, dx, dy, dz, cx, cy, cz, r,
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

  // Compute output shape: ray_batch + sphere_batch
  std::vector<int64_t> out_batch_shape;
  for (auto s : ray_batch_shape) {
    out_batch_shape.push_back(s);
  }
  for (auto s : sphere_batch_shape) {
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
 * CPU implementation of ray-sphere intersection backward pass.
 *
 * @param grad_t Gradient of loss w.r.t. t, shape (batch,)
 * @param grad_hit_point Gradient of loss w.r.t. hit_point, shape (batch, 3)
 * @param grad_normal Gradient of loss w.r.t. normal, shape (batch, 3)
 * @param grad_uv Gradient of loss w.r.t. uv, shape (batch, 2)
 * @param origins Ray origins (saved from forward), shape (*, 3)
 * @param directions Ray directions (saved from forward), shape (*, 3)
 * @param centers Sphere centers (saved from forward), shape (*, 3)
 * @param radii Sphere radii (saved from forward), shape (*)
 * @param t Intersection parameter (saved from forward), shape (batch,)
 * @param hit Hit flags (saved from forward), shape (batch,)
 * @return Tuple of (grad_origins, grad_directions, grad_centers, grad_radii)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_sphere_backward(
    const at::Tensor& grad_t,
    const at::Tensor& grad_hit_point,
    const at::Tensor& grad_normal,
    const at::Tensor& grad_uv,
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& centers,
    const at::Tensor& radii,
    const at::Tensor& t,
    const at::Tensor& hit) {
  // Flatten ray batch dimensions
  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  // Flatten sphere batch dimensions
  auto sphere_batch_shape = radii.sizes();
  int64_t num_spheres = 1;
  for (auto s : sphere_batch_shape) {
    num_spheres *= s;
  }

  int64_t batch_size = num_rays * num_spheres;

  // Make inputs contiguous
  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor centers_flat = centers.reshape({num_spheres, 3}).contiguous();
  at::Tensor radii_flat = radii.reshape({num_spheres}).contiguous();

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
  at::Tensor grad_centers = at::zeros({num_spheres, 3}, options);
  at::Tensor grad_radii = at::zeros({num_spheres}, options);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      origins.scalar_type(),
      "ray_sphere_backward_cpu",
      [&]() {
        const scalar_t* origins_ptr = origins_flat.data_ptr<scalar_t>();
        const scalar_t* directions_ptr = directions_flat.data_ptr<scalar_t>();
        const scalar_t* centers_ptr = centers_flat.data_ptr<scalar_t>();
        const scalar_t* radii_ptr = radii_flat.data_ptr<scalar_t>();

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
        at::Tensor grad_centers_elem = at::zeros({batch_size, 3}, options);
        at::Tensor grad_radii_elem = at::zeros({batch_size}, options);

        scalar_t* grad_origins_elem_ptr = grad_origins_elem.data_ptr<scalar_t>();
        scalar_t* grad_directions_elem_ptr = grad_directions_elem.data_ptr<scalar_t>();
        scalar_t* grad_centers_elem_ptr = grad_centers_elem.data_ptr<scalar_t>();
        scalar_t* grad_radii_elem_ptr = grad_radii_elem.data_ptr<scalar_t>();

        at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
          for (int64_t idx = begin; idx < end; ++idx) {
            // Compute ray and sphere indices
            int64_t ray_idx = idx / num_spheres;
            int64_t sphere_idx = idx % num_spheres;

            // Get ray data
            scalar_t ox = origins_ptr[ray_idx * 3 + 0];
            scalar_t oy = origins_ptr[ray_idx * 3 + 1];
            scalar_t oz = origins_ptr[ray_idx * 3 + 2];
            scalar_t dx = directions_ptr[ray_idx * 3 + 0];
            scalar_t dy = directions_ptr[ray_idx * 3 + 1];
            scalar_t dz = directions_ptr[ray_idx * 3 + 2];

            // Get sphere data
            scalar_t cx = centers_ptr[sphere_idx * 3 + 0];
            scalar_t cy = centers_ptr[sphere_idx * 3 + 1];
            scalar_t cz = centers_ptr[sphere_idx * 3 + 2];
            scalar_t r = radii_ptr[sphere_idx];

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
            scalar_t gc_x, gc_y, gc_z;
            scalar_t gr;

            // Call backward kernel
            kernel::geometry::intersection::ray_sphere_backward_kernel<scalar_t>(
                gt, ghx, ghy, ghz, gnx, gny, gnz, gu, gv,
                ox, oy, oz, dx, dy, dz, cx, cy, cz, r,
                t_val, hit_val,
                go_x, go_y, go_z, gd_x, gd_y, gd_z,
                gc_x, gc_y, gc_z, gr);

            // Store per-element gradients
            grad_origins_elem_ptr[idx * 3 + 0] = go_x;
            grad_origins_elem_ptr[idx * 3 + 1] = go_y;
            grad_origins_elem_ptr[idx * 3 + 2] = go_z;
            grad_directions_elem_ptr[idx * 3 + 0] = gd_x;
            grad_directions_elem_ptr[idx * 3 + 1] = gd_y;
            grad_directions_elem_ptr[idx * 3 + 2] = gd_z;
            grad_centers_elem_ptr[idx * 3 + 0] = gc_x;
            grad_centers_elem_ptr[idx * 3 + 1] = gc_y;
            grad_centers_elem_ptr[idx * 3 + 2] = gc_z;
            grad_radii_elem_ptr[idx] = gr;
          }
        });

        // Reduce gradients: sum over broadcast dimension
        // grad_origins: sum over spheres for each ray
        // grad_centers: sum over rays for each sphere
        auto grad_origins_acc = grad_origins.accessor<scalar_t, 2>();
        auto grad_directions_acc = grad_directions.accessor<scalar_t, 2>();
        auto grad_centers_acc = grad_centers.accessor<scalar_t, 2>();
        auto grad_radii_acc = grad_radii.accessor<scalar_t, 1>();

        // Sum gradients (sequential to avoid race conditions)
        for (int64_t idx = 0; idx < batch_size; ++idx) {
          int64_t ray_idx = idx / num_spheres;
          int64_t sphere_idx = idx % num_spheres;

          // Accumulate ray gradients
          grad_origins_acc[ray_idx][0] += grad_origins_elem_ptr[idx * 3 + 0];
          grad_origins_acc[ray_idx][1] += grad_origins_elem_ptr[idx * 3 + 1];
          grad_origins_acc[ray_idx][2] += grad_origins_elem_ptr[idx * 3 + 2];
          grad_directions_acc[ray_idx][0] += grad_directions_elem_ptr[idx * 3 + 0];
          grad_directions_acc[ray_idx][1] += grad_directions_elem_ptr[idx * 3 + 1];
          grad_directions_acc[ray_idx][2] += grad_directions_elem_ptr[idx * 3 + 2];

          // Accumulate sphere gradients
          grad_centers_acc[sphere_idx][0] += grad_centers_elem_ptr[idx * 3 + 0];
          grad_centers_acc[sphere_idx][1] += grad_centers_elem_ptr[idx * 3 + 1];
          grad_centers_acc[sphere_idx][2] += grad_centers_elem_ptr[idx * 3 + 2];
          grad_radii_acc[sphere_idx] += grad_radii_elem_ptr[idx];
        }
      });

  // Reshape to original input shapes
  std::vector<int64_t> origins_shape(ray_batch_shape.begin(), ray_batch_shape.end());
  origins_shape.push_back(3);

  std::vector<int64_t> centers_shape(sphere_batch_shape.begin(), sphere_batch_shape.end());
  centers_shape.push_back(3);

  std::vector<int64_t> radii_shape(sphere_batch_shape.begin(), sphere_batch_shape.end());

  return std::make_tuple(
      grad_origins.reshape(origins_shape),
      grad_directions.reshape(origins_shape),
      grad_centers.reshape(centers_shape),
      grad_radii.reshape(radii_shape));
}

}  // namespace torchscience::cpu::geometry::intersection

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("ray_sphere", torchscience::cpu::geometry::intersection::ray_sphere);
  m.impl(
      "ray_sphere_backward",
      torchscience::cpu::geometry::intersection::ray_sphere_backward);
}
