// src/torchscience/csrc/cuda/geometry/intersection/ray_sphere.cu
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../../../kernel/geometry/intersection/ray_sphere.h"
#include "../../../kernel/geometry/intersection/ray_sphere_backward.h"

namespace torchscience::cuda::geometry::intersection {

template <typename scalar_t>
__global__ void ray_sphere_forward_kernel(
    const scalar_t* __restrict__ origins,
    const scalar_t* __restrict__ directions,
    const scalar_t* __restrict__ centers,
    const scalar_t* __restrict__ radii,
    scalar_t* __restrict__ t_out,
    scalar_t* __restrict__ hit_point_out,
    scalar_t* __restrict__ normal_out,
    scalar_t* __restrict__ uv_out,
    bool* __restrict__ hit_out,
    int64_t batch_size,
    int64_t num_spheres) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    int64_t ray_idx = idx / num_spheres;
    int64_t sphere_idx = idx % num_spheres;

    scalar_t t, hx, hy, hz, nx, ny, nz, u, v, hit_val;

    kernel::geometry::intersection::ray_sphere_kernel<scalar_t>(
        origins[ray_idx * 3 + 0],
        origins[ray_idx * 3 + 1],
        origins[ray_idx * 3 + 2],
        directions[ray_idx * 3 + 0],
        directions[ray_idx * 3 + 1],
        directions[ray_idx * 3 + 2],
        centers[sphere_idx * 3 + 0],
        centers[sphere_idx * 3 + 1],
        centers[sphere_idx * 3 + 2],
        radii[sphere_idx],
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
__global__ void ray_sphere_backward_cuda_kernel(
    const scalar_t* __restrict__ grad_t,
    const scalar_t* __restrict__ grad_hit_point,
    const scalar_t* __restrict__ grad_normal,
    const scalar_t* __restrict__ grad_uv,
    const scalar_t* __restrict__ origins,
    const scalar_t* __restrict__ directions,
    const scalar_t* __restrict__ centers,
    const scalar_t* __restrict__ radii,
    const scalar_t* __restrict__ t,
    const bool* __restrict__ hit,
    scalar_t* __restrict__ grad_origins_out,
    scalar_t* __restrict__ grad_directions_out,
    scalar_t* __restrict__ grad_centers_out,
    scalar_t* __restrict__ grad_radii_out,
    int64_t batch_size,
    int64_t num_spheres) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    int64_t ray_idx = idx / num_spheres;
    int64_t sphere_idx = idx % num_spheres;

    scalar_t hit_val = hit[idx] ? scalar_t(1) : scalar_t(0);

    scalar_t go_x, go_y, go_z, gd_x, gd_y, gd_z;
    scalar_t gc_x, gc_y, gc_z, gr;

    kernel::geometry::intersection::ray_sphere_backward_kernel<scalar_t>(
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
        centers[sphere_idx * 3 + 0],
        centers[sphere_idx * 3 + 1],
        centers[sphere_idx * 3 + 2],
        radii[sphere_idx],
        t[idx],
        hit_val,
        go_x, go_y, go_z, gd_x, gd_y, gd_z,
        gc_x, gc_y, gc_z, gr);

    grad_origins_out[idx * 3 + 0] = go_x;
    grad_origins_out[idx * 3 + 1] = go_y;
    grad_origins_out[idx * 3 + 2] = go_z;
    grad_directions_out[idx * 3 + 0] = gd_x;
    grad_directions_out[idx * 3 + 1] = gd_y;
    grad_directions_out[idx * 3 + 2] = gd_z;
    grad_centers_out[idx * 3 + 0] = gc_x;
    grad_centers_out[idx * 3 + 1] = gc_y;
    grad_centers_out[idx * 3 + 2] = gc_z;
    grad_radii_out[idx] = gr;
  }
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_sphere(
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& centers,
    const at::Tensor& radii) {
  c10::cuda::CUDAGuard device_guard(origins.device());

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

  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  auto sphere_batch_shape = radii.sizes();
  int64_t num_spheres = 1;
  for (auto s : sphere_batch_shape) {
    num_spheres *= s;
  }

  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor centers_flat = centers.reshape({num_spheres, 3}).contiguous();
  at::Tensor radii_flat = radii.reshape({num_spheres}).contiguous();

  int64_t batch_size = num_rays * num_spheres;

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
      "ray_sphere_cuda",
      [&]() {
        ray_sphere_forward_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                origins_flat.data_ptr<scalar_t>(),
                directions_flat.data_ptr<scalar_t>(),
                centers_flat.data_ptr<scalar_t>(),
                radii_flat.data_ptr<scalar_t>(),
                t_out.data_ptr<scalar_t>(),
                hit_point_out.data_ptr<scalar_t>(),
                normal_out.data_ptr<scalar_t>(),
                uv_out.data_ptr<scalar_t>(),
                hit_out.data_ptr<bool>(),
                batch_size,
                num_spheres);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  std::vector<int64_t> out_batch_shape;
  for (auto s : ray_batch_shape) {
    out_batch_shape.push_back(s);
  }
  for (auto s : sphere_batch_shape) {
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
  c10::cuda::CUDAGuard device_guard(origins.device());

  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  auto sphere_batch_shape = radii.sizes();
  int64_t num_spheres = 1;
  for (auto s : sphere_batch_shape) {
    num_spheres *= s;
  }

  int64_t batch_size = num_rays * num_spheres;

  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor centers_flat = centers.reshape({num_spheres, 3}).contiguous();
  at::Tensor radii_flat = radii.reshape({num_spheres}).contiguous();

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
  at::Tensor grad_centers_elem = at::zeros({batch_size, 3}, options);
  at::Tensor grad_radii_elem = at::zeros({batch_size}, options);

  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      origins.scalar_type(),
      "ray_sphere_backward_cuda",
      [&]() {
        ray_sphere_backward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_t_flat.data_ptr<scalar_t>(),
                grad_hit_point_flat.data_ptr<scalar_t>(),
                grad_normal_flat.data_ptr<scalar_t>(),
                grad_uv_flat.data_ptr<scalar_t>(),
                origins_flat.data_ptr<scalar_t>(),
                directions_flat.data_ptr<scalar_t>(),
                centers_flat.data_ptr<scalar_t>(),
                radii_flat.data_ptr<scalar_t>(),
                t_flat.data_ptr<scalar_t>(),
                hit_flat.data_ptr<bool>(),
                grad_origins_elem.data_ptr<scalar_t>(),
                grad_directions_elem.data_ptr<scalar_t>(),
                grad_centers_elem.data_ptr<scalar_t>(),
                grad_radii_elem.data_ptr<scalar_t>(),
                batch_size,
                num_spheres);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // Reduce gradients over broadcast dimensions
  auto grad_origins_elem_r =
      grad_origins_elem.reshape({num_rays, num_spheres, 3});
  auto grad_directions_elem_r =
      grad_directions_elem.reshape({num_rays, num_spheres, 3});
  auto grad_centers_elem_r =
      grad_centers_elem.reshape({num_rays, num_spheres, 3});
  auto grad_radii_elem_r = grad_radii_elem.reshape({num_rays, num_spheres});

  at::Tensor grad_origins = grad_origins_elem_r.sum(1);
  at::Tensor grad_directions = grad_directions_elem_r.sum(1);
  at::Tensor grad_centers = grad_centers_elem_r.sum(0);
  at::Tensor grad_radii = grad_radii_elem_r.sum(0);

  std::vector<int64_t> origins_shape(
      ray_batch_shape.begin(), ray_batch_shape.end());
  origins_shape.push_back(3);

  std::vector<int64_t> centers_shape(
      sphere_batch_shape.begin(), sphere_batch_shape.end());
  centers_shape.push_back(3);

  std::vector<int64_t> radii_shape(
      sphere_batch_shape.begin(), sphere_batch_shape.end());

  return std::make_tuple(
      grad_origins.reshape(origins_shape),
      grad_directions.reshape(origins_shape),
      grad_centers.reshape(centers_shape),
      grad_radii.reshape(radii_shape));
}

}  // namespace torchscience::cuda::geometry::intersection

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
  m.impl(
      "ray_sphere",
      torchscience::cuda::geometry::intersection::ray_sphere);
  m.impl(
      "ray_sphere_backward",
      torchscience::cuda::geometry::intersection::ray_sphere_backward);
}
