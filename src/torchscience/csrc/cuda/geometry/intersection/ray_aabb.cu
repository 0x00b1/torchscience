// src/torchscience/csrc/cuda/geometry/intersection/ray_aabb.cu
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../../../kernel/geometry/intersection/ray_aabb.h"
#include "../../../kernel/geometry/intersection/ray_aabb_backward.h"

namespace torchscience::cuda::geometry::intersection {

template <typename scalar_t>
__global__ void ray_aabb_forward_kernel(
    const scalar_t* __restrict__ origins,
    const scalar_t* __restrict__ directions,
    const scalar_t* __restrict__ box_min,
    const scalar_t* __restrict__ box_max,
    scalar_t* __restrict__ t_out,
    scalar_t* __restrict__ hit_point_out,
    scalar_t* __restrict__ normal_out,
    scalar_t* __restrict__ uv_out,
    bool* __restrict__ hit_out,
    int64_t batch_size,
    int64_t num_boxes) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    int64_t ray_idx = idx / num_boxes;
    int64_t box_idx = idx % num_boxes;

    scalar_t t, hx, hy, hz, nx, ny, nz, u, v, hit_val;

    kernel::geometry::intersection::ray_aabb_kernel<scalar_t>(
        origins[ray_idx * 3 + 0],
        origins[ray_idx * 3 + 1],
        origins[ray_idx * 3 + 2],
        directions[ray_idx * 3 + 0],
        directions[ray_idx * 3 + 1],
        directions[ray_idx * 3 + 2],
        box_min[box_idx * 3 + 0],
        box_min[box_idx * 3 + 1],
        box_min[box_idx * 3 + 2],
        box_max[box_idx * 3 + 0],
        box_max[box_idx * 3 + 1],
        box_max[box_idx * 3 + 2],
        t, hx, hy, hz, nx, ny, nz, u, v, hit_val);

    t_out[idx] = t;
    hit_point_out[idx * 3 + 0] = hx;
    hit_point_out[idx * 3 + 1] = hy;
    hit_point_out[idx * 3 + 2] = hz;
    normal_out[idx * 3 + 0] = nx;
    normal_out[idx * 3 + 1] = ny;
    normal_out[idx * 3 + 2] = nz;
    uv_out[idx * 2 + 0] = u;
    uv_out[idx * 2 + 1] = v;
    hit_out[idx] = hit_val > scalar_t(0.5);
  }
}

template <typename scalar_t>
__global__ void ray_aabb_backward_cuda_kernel(
    const scalar_t* __restrict__ grad_t,
    const scalar_t* __restrict__ grad_hit_point,
    const scalar_t* __restrict__ grad_normal,
    const scalar_t* __restrict__ grad_uv,
    const scalar_t* __restrict__ origins,
    const scalar_t* __restrict__ directions,
    const scalar_t* __restrict__ box_min,
    const scalar_t* __restrict__ box_max,
    const scalar_t* __restrict__ t,
    const bool* __restrict__ hit,
    scalar_t* __restrict__ grad_origins_out,
    scalar_t* __restrict__ grad_directions_out,
    scalar_t* __restrict__ grad_box_min_out,
    scalar_t* __restrict__ grad_box_max_out,
    int64_t batch_size,
    int64_t num_boxes) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    int64_t ray_idx = idx / num_boxes;
    int64_t box_idx = idx % num_boxes;

    scalar_t hit_val = hit[idx] ? scalar_t(1) : scalar_t(0);

    scalar_t go_x, go_y, go_z, gd_x, gd_y, gd_z;
    scalar_t gbmin_x, gbmin_y, gbmin_z;
    scalar_t gbmax_x, gbmax_y, gbmax_z;

    kernel::geometry::intersection::ray_aabb_backward_kernel<scalar_t>(
        grad_t[idx],
        grad_hit_point[idx * 3 + 0],
        grad_hit_point[idx * 3 + 1],
        grad_hit_point[idx * 3 + 2],
        grad_normal[idx * 3 + 0],
        grad_normal[idx * 3 + 1],
        grad_normal[idx * 3 + 2],
        grad_uv[idx * 2 + 0],
        grad_uv[idx * 2 + 1],
        origins[ray_idx * 3 + 0],
        origins[ray_idx * 3 + 1],
        origins[ray_idx * 3 + 2],
        directions[ray_idx * 3 + 0],
        directions[ray_idx * 3 + 1],
        directions[ray_idx * 3 + 2],
        box_min[box_idx * 3 + 0],
        box_min[box_idx * 3 + 1],
        box_min[box_idx * 3 + 2],
        box_max[box_idx * 3 + 0],
        box_max[box_idx * 3 + 1],
        box_max[box_idx * 3 + 2],
        t[idx],
        hit_val,
        go_x, go_y, go_z, gd_x, gd_y, gd_z,
        gbmin_x, gbmin_y, gbmin_z, gbmax_x, gbmax_y, gbmax_z);

    grad_origins_out[idx * 3 + 0] = go_x;
    grad_origins_out[idx * 3 + 1] = go_y;
    grad_origins_out[idx * 3 + 2] = go_z;
    grad_directions_out[idx * 3 + 0] = gd_x;
    grad_directions_out[idx * 3 + 1] = gd_y;
    grad_directions_out[idx * 3 + 2] = gd_z;
    grad_box_min_out[idx * 3 + 0] = gbmin_x;
    grad_box_min_out[idx * 3 + 1] = gbmin_y;
    grad_box_min_out[idx * 3 + 2] = gbmin_z;
    grad_box_max_out[idx * 3 + 0] = gbmax_x;
    grad_box_max_out[idx * 3 + 1] = gbmax_y;
    grad_box_max_out[idx * 3 + 2] = gbmax_z;
  }
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_aabb(
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& box_min,
    const at::Tensor& box_max) {
  c10::cuda::CUDAGuard device_guard(origins.device());

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

  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  auto box_batch_shape = box_min.sizes().slice(0, box_min.dim() - 1);
  int64_t num_boxes = 1;
  for (auto s : box_batch_shape) {
    num_boxes *= s;
  }

  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor box_min_flat = box_min.reshape({num_boxes, 3}).contiguous();
  at::Tensor box_max_flat = box_max.reshape({num_boxes, 3}).contiguous();

  int64_t batch_size = num_rays * num_boxes;

  auto options = origins.options();
  at::Tensor t_out = at::empty({batch_size}, options);
  at::Tensor hit_point_out = at::empty({batch_size, 3}, options);
  at::Tensor normal_out = at::empty({batch_size, 3}, options);
  at::Tensor uv_out = at::empty({batch_size, 2}, options);
  at::Tensor hit_out = at::empty({batch_size}, options.dtype(at::kBool));

  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      origins.scalar_type(),
      "ray_aabb_cuda",
      [&]() {
        ray_aabb_forward_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                origins_flat.data_ptr<scalar_t>(),
                directions_flat.data_ptr<scalar_t>(),
                box_min_flat.data_ptr<scalar_t>(),
                box_max_flat.data_ptr<scalar_t>(),
                t_out.data_ptr<scalar_t>(),
                hit_point_out.data_ptr<scalar_t>(),
                normal_out.data_ptr<scalar_t>(),
                uv_out.data_ptr<scalar_t>(),
                hit_out.data_ptr<bool>(),
                batch_size,
                num_boxes);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  std::vector<int64_t> out_batch_shape;
  for (auto s : ray_batch_shape) {
    out_batch_shape.push_back(s);
  }
  for (auto s : box_batch_shape) {
    out_batch_shape.push_back(s);
  }

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
  c10::cuda::CUDAGuard device_guard(origins.device());

  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  auto box_batch_shape = box_min.sizes().slice(0, box_min.dim() - 1);
  int64_t num_boxes = 1;
  for (auto s : box_batch_shape) {
    num_boxes *= s;
  }

  int64_t batch_size = num_rays * num_boxes;

  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor box_min_flat = box_min.reshape({num_boxes, 3}).contiguous();
  at::Tensor box_max_flat = box_max.reshape({num_boxes, 3}).contiguous();

  at::Tensor grad_t_flat = grad_t.reshape({batch_size}).contiguous();
  at::Tensor grad_hit_point_flat =
      grad_hit_point.reshape({batch_size, 3}).contiguous();
  at::Tensor grad_normal_flat =
      grad_normal.reshape({batch_size, 3}).contiguous();
  at::Tensor grad_uv_flat = grad_uv.reshape({batch_size, 2}).contiguous();
  at::Tensor t_flat = t.reshape({batch_size}).contiguous();
  at::Tensor hit_flat = hit.reshape({batch_size}).contiguous();

  auto options = origins.options();
  at::Tensor grad_origins_elem = at::zeros({batch_size, 3}, options);
  at::Tensor grad_directions_elem = at::zeros({batch_size, 3}, options);
  at::Tensor grad_box_min_elem = at::zeros({batch_size, 3}, options);
  at::Tensor grad_box_max_elem = at::zeros({batch_size, 3}, options);

  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      origins.scalar_type(),
      "ray_aabb_backward_cuda",
      [&]() {
        ray_aabb_backward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_t_flat.data_ptr<scalar_t>(),
                grad_hit_point_flat.data_ptr<scalar_t>(),
                grad_normal_flat.data_ptr<scalar_t>(),
                grad_uv_flat.data_ptr<scalar_t>(),
                origins_flat.data_ptr<scalar_t>(),
                directions_flat.data_ptr<scalar_t>(),
                box_min_flat.data_ptr<scalar_t>(),
                box_max_flat.data_ptr<scalar_t>(),
                t_flat.data_ptr<scalar_t>(),
                hit_flat.data_ptr<bool>(),
                grad_origins_elem.data_ptr<scalar_t>(),
                grad_directions_elem.data_ptr<scalar_t>(),
                grad_box_min_elem.data_ptr<scalar_t>(),
                grad_box_max_elem.data_ptr<scalar_t>(),
                batch_size,
                num_boxes);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // Reduce gradients over broadcast dimensions
  auto grad_origins_elem_r =
      grad_origins_elem.reshape({num_rays, num_boxes, 3});
  auto grad_directions_elem_r =
      grad_directions_elem.reshape({num_rays, num_boxes, 3});
  auto grad_box_min_elem_r =
      grad_box_min_elem.reshape({num_rays, num_boxes, 3});
  auto grad_box_max_elem_r =
      grad_box_max_elem.reshape({num_rays, num_boxes, 3});

  at::Tensor grad_origins = grad_origins_elem_r.sum(1);
  at::Tensor grad_directions = grad_directions_elem_r.sum(1);
  at::Tensor grad_box_min_out = grad_box_min_elem_r.sum(0);
  at::Tensor grad_box_max_out = grad_box_max_elem_r.sum(0);

  std::vector<int64_t> origins_shape(
      ray_batch_shape.begin(), ray_batch_shape.end());
  origins_shape.push_back(3);

  std::vector<int64_t> box_shape(
      box_batch_shape.begin(), box_batch_shape.end());
  box_shape.push_back(3);

  return std::make_tuple(
      grad_origins.reshape(origins_shape),
      grad_directions.reshape(origins_shape),
      grad_box_min_out.reshape(box_shape),
      grad_box_max_out.reshape(box_shape));
}

}  // namespace torchscience::cuda::geometry::intersection

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
  m.impl(
      "ray_aabb",
      torchscience::cuda::geometry::intersection::ray_aabb);
  m.impl(
      "ray_aabb_backward",
      torchscience::cuda::geometry::intersection::ray_aabb_backward);
}
