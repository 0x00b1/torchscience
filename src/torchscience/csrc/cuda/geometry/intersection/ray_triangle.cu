// src/torchscience/csrc/cuda/geometry/intersection/ray_triangle.cu
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../../../kernel/geometry/intersection/ray_triangle.h"
#include "../../../kernel/geometry/intersection/ray_triangle_backward.h"

namespace torchscience::cuda::geometry::intersection {

template <typename scalar_t>
__global__ void ray_triangle_forward_kernel(
    const scalar_t* __restrict__ origins,
    const scalar_t* __restrict__ directions,
    const scalar_t* __restrict__ v0,
    const scalar_t* __restrict__ v1,
    const scalar_t* __restrict__ v2,
    scalar_t* __restrict__ t_out,
    scalar_t* __restrict__ hit_point_out,
    scalar_t* __restrict__ normal_out,
    scalar_t* __restrict__ uv_out,
    bool* __restrict__ hit_out,
    int64_t batch_size,
    int64_t num_triangles) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    int64_t ray_idx = idx / num_triangles;
    int64_t tri_idx = idx % num_triangles;

    scalar_t t, hx, hy, hz, nx, ny, nz, u, v, hit_val;

    kernel::geometry::intersection::ray_triangle_kernel<scalar_t>(
        origins[ray_idx * 3 + 0],
        origins[ray_idx * 3 + 1],
        origins[ray_idx * 3 + 2],
        directions[ray_idx * 3 + 0],
        directions[ray_idx * 3 + 1],
        directions[ray_idx * 3 + 2],
        v0[tri_idx * 3 + 0],
        v0[tri_idx * 3 + 1],
        v0[tri_idx * 3 + 2],
        v1[tri_idx * 3 + 0],
        v1[tri_idx * 3 + 1],
        v1[tri_idx * 3 + 2],
        v2[tri_idx * 3 + 0],
        v2[tri_idx * 3 + 1],
        v2[tri_idx * 3 + 2],
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
__global__ void ray_triangle_backward_cuda_kernel(
    const scalar_t* __restrict__ grad_t,
    const scalar_t* __restrict__ grad_hit_point,
    const scalar_t* __restrict__ grad_normal,
    const scalar_t* __restrict__ grad_uv,
    const scalar_t* __restrict__ origins,
    const scalar_t* __restrict__ directions,
    const scalar_t* __restrict__ v0,
    const scalar_t* __restrict__ v1,
    const scalar_t* __restrict__ v2,
    const scalar_t* __restrict__ t,
    const bool* __restrict__ hit,
    scalar_t* __restrict__ grad_origins_out,
    scalar_t* __restrict__ grad_directions_out,
    scalar_t* __restrict__ grad_v0_out,
    scalar_t* __restrict__ grad_v1_out,
    scalar_t* __restrict__ grad_v2_out,
    int64_t batch_size,
    int64_t num_triangles) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size) {
    int64_t ray_idx = idx / num_triangles;
    int64_t tri_idx = idx % num_triangles;

    scalar_t hit_val = hit[idx] ? scalar_t(1) : scalar_t(0);

    scalar_t go_x, go_y, go_z, gd_x, gd_y, gd_z;
    scalar_t gv0_x, gv0_y, gv0_z;
    scalar_t gv1_x, gv1_y, gv1_z;
    scalar_t gv2_x, gv2_y, gv2_z;

    kernel::geometry::intersection::ray_triangle_backward_kernel<scalar_t>(
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
        v0[tri_idx * 3 + 0],
        v0[tri_idx * 3 + 1],
        v0[tri_idx * 3 + 2],
        v1[tri_idx * 3 + 0],
        v1[tri_idx * 3 + 1],
        v1[tri_idx * 3 + 2],
        v2[tri_idx * 3 + 0],
        v2[tri_idx * 3 + 1],
        v2[tri_idx * 3 + 2],
        t[idx],
        hit_val,
        go_x, go_y, go_z, gd_x, gd_y, gd_z,
        gv0_x, gv0_y, gv0_z, gv1_x, gv1_y, gv1_z,
        gv2_x, gv2_y, gv2_z);

    grad_origins_out[idx * 3 + 0] = go_x;
    grad_origins_out[idx * 3 + 1] = go_y;
    grad_origins_out[idx * 3 + 2] = go_z;
    grad_directions_out[idx * 3 + 0] = gd_x;
    grad_directions_out[idx * 3 + 1] = gd_y;
    grad_directions_out[idx * 3 + 2] = gd_z;
    grad_v0_out[idx * 3 + 0] = gv0_x;
    grad_v0_out[idx * 3 + 1] = gv0_y;
    grad_v0_out[idx * 3 + 2] = gv0_z;
    grad_v1_out[idx * 3 + 0] = gv1_x;
    grad_v1_out[idx * 3 + 1] = gv1_y;
    grad_v1_out[idx * 3 + 2] = gv1_z;
    grad_v2_out[idx * 3 + 0] = gv2_x;
    grad_v2_out[idx * 3 + 1] = gv2_y;
    grad_v2_out[idx * 3 + 2] = gv2_z;
  }
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_triangle(
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& v0,
    const at::Tensor& v1,
    const at::Tensor& v2) {
  c10::cuda::CUDAGuard device_guard(origins.device());

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

  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  auto triangle_batch_shape = v0.sizes().slice(0, v0.dim() - 1);
  int64_t num_triangles = 1;
  for (auto s : triangle_batch_shape) {
    num_triangles *= s;
  }

  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor v0_flat = v0.reshape({num_triangles, 3}).contiguous();
  at::Tensor v1_flat = v1.reshape({num_triangles, 3}).contiguous();
  at::Tensor v2_flat = v2.reshape({num_triangles, 3}).contiguous();

  int64_t batch_size = num_rays * num_triangles;

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
      "ray_triangle_cuda",
      [&]() {
        ray_triangle_forward_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                origins_flat.data_ptr<scalar_t>(),
                directions_flat.data_ptr<scalar_t>(),
                v0_flat.data_ptr<scalar_t>(),
                v1_flat.data_ptr<scalar_t>(),
                v2_flat.data_ptr<scalar_t>(),
                t_out.data_ptr<scalar_t>(),
                hit_point_out.data_ptr<scalar_t>(),
                normal_out.data_ptr<scalar_t>(),
                uv_out.data_ptr<scalar_t>(),
                hit_out.data_ptr<bool>(),
                batch_size,
                num_triangles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  std::vector<int64_t> out_batch_shape;
  for (auto s : ray_batch_shape) {
    out_batch_shape.push_back(s);
  }
  for (auto s : triangle_batch_shape) {
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
  c10::cuda::CUDAGuard device_guard(origins.device());

  auto ray_batch_shape = origins.sizes().slice(0, origins.dim() - 1);
  int64_t num_rays = 1;
  for (auto s : ray_batch_shape) {
    num_rays *= s;
  }

  auto triangle_batch_shape = v0.sizes().slice(0, v0.dim() - 1);
  int64_t num_triangles = 1;
  for (auto s : triangle_batch_shape) {
    num_triangles *= s;
  }

  int64_t batch_size = num_rays * num_triangles;

  at::Tensor origins_flat = origins.reshape({num_rays, 3}).contiguous();
  at::Tensor directions_flat = directions.reshape({num_rays, 3}).contiguous();
  at::Tensor v0_flat = v0.reshape({num_triangles, 3}).contiguous();
  at::Tensor v1_flat = v1.reshape({num_triangles, 3}).contiguous();
  at::Tensor v2_flat = v2.reshape({num_triangles, 3}).contiguous();

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
  at::Tensor grad_v0_elem = at::zeros({batch_size, 3}, options);
  at::Tensor grad_v1_elem = at::zeros({batch_size, 3}, options);
  at::Tensor grad_v2_elem = at::zeros({batch_size, 3}, options);

  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      origins.scalar_type(),
      "ray_triangle_backward_cuda",
      [&]() {
        ray_triangle_backward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_t_flat.data_ptr<scalar_t>(),
                grad_hit_point_flat.data_ptr<scalar_t>(),
                grad_normal_flat.data_ptr<scalar_t>(),
                grad_uv_flat.data_ptr<scalar_t>(),
                origins_flat.data_ptr<scalar_t>(),
                directions_flat.data_ptr<scalar_t>(),
                v0_flat.data_ptr<scalar_t>(),
                v1_flat.data_ptr<scalar_t>(),
                v2_flat.data_ptr<scalar_t>(),
                t_flat.data_ptr<scalar_t>(),
                hit_flat.data_ptr<bool>(),
                grad_origins_elem.data_ptr<scalar_t>(),
                grad_directions_elem.data_ptr<scalar_t>(),
                grad_v0_elem.data_ptr<scalar_t>(),
                grad_v1_elem.data_ptr<scalar_t>(),
                grad_v2_elem.data_ptr<scalar_t>(),
                batch_size,
                num_triangles);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // Reduce gradients over broadcast dimensions
  auto grad_origins_elem_r =
      grad_origins_elem.reshape({num_rays, num_triangles, 3});
  auto grad_directions_elem_r =
      grad_directions_elem.reshape({num_rays, num_triangles, 3});
  auto grad_v0_elem_r = grad_v0_elem.reshape({num_rays, num_triangles, 3});
  auto grad_v1_elem_r = grad_v1_elem.reshape({num_rays, num_triangles, 3});
  auto grad_v2_elem_r = grad_v2_elem.reshape({num_rays, num_triangles, 3});

  at::Tensor grad_origins = grad_origins_elem_r.sum(1);
  at::Tensor grad_directions = grad_directions_elem_r.sum(1);
  at::Tensor grad_v0_out = grad_v0_elem_r.sum(0);
  at::Tensor grad_v1_out = grad_v1_elem_r.sum(0);
  at::Tensor grad_v2_out = grad_v2_elem_r.sum(0);

  std::vector<int64_t> origins_shape(
      ray_batch_shape.begin(), ray_batch_shape.end());
  origins_shape.push_back(3);

  std::vector<int64_t> v_shape(
      triangle_batch_shape.begin(), triangle_batch_shape.end());
  v_shape.push_back(3);

  return std::make_tuple(
      grad_origins.reshape(origins_shape),
      grad_directions.reshape(origins_shape),
      grad_v0_out.reshape(v_shape),
      grad_v1_out.reshape(v_shape),
      grad_v2_out.reshape(v_shape));
}

}  // namespace torchscience::cuda::geometry::intersection

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
  m.impl(
      "ray_triangle",
      torchscience::cuda::geometry::intersection::ray_triangle);
  m.impl(
      "ray_triangle_backward",
      torchscience::cuda::geometry::intersection::ray_triangle_backward);
}
