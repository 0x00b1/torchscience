// src/torchscience/csrc/cuda/graphics/shading/cook_torrance.cu
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../../../impl/graphics/shading/cook_torrance.h"
#include "../../../impl/graphics/shading/cook_torrance_backward.h"
#include "../../../impl/graphics/shading/cook_torrance_backward_backward.h"

namespace torchscience::cuda::graphics::shading {

using namespace torchscience::impl::graphics::shading;

/**
 * CUDA kernel for Cook-Torrance forward pass (scalar f0).
 * Uses strided access for roughness and f0 to avoid unnecessary memory allocation
 * when these are scalar values broadcasted across the batch.
 */
template <typename scalar_t>
__global__ void cook_torrance_forward_kernel(
    const scalar_t* __restrict__ normal,
    const scalar_t* __restrict__ view,
    const scalar_t* __restrict__ light,
    const scalar_t* __restrict__ roughness,
    const scalar_t* __restrict__ f0,
    scalar_t* __restrict__ output,
    int64_t numel,
    int64_t roughness_stride,
    int64_t f0_stride
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        const scalar_t* n = normal + idx * 3;
        const scalar_t* v = view + idx * 3;
        const scalar_t* l = light + idx * 3;
        scalar_t r = roughness[idx * roughness_stride];
        scalar_t f = f0[idx * f0_stride];

        output[idx] = cook_torrance_scalar<scalar_t>(n, v, l, r, f);
    }
}

/**
 * CUDA kernel for Cook-Torrance forward pass (RGB f0).
 * Uses strided access for roughness and f0 to avoid unnecessary memory allocation.
 */
template <typename scalar_t>
__global__ void cook_torrance_forward_rgb_kernel(
    const scalar_t* __restrict__ normal,
    const scalar_t* __restrict__ view,
    const scalar_t* __restrict__ light,
    const scalar_t* __restrict__ roughness,
    const scalar_t* __restrict__ f0,
    scalar_t* __restrict__ output,
    int64_t numel,
    int64_t roughness_stride,
    int64_t f0_stride
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        const scalar_t* n = normal + idx * 3;
        const scalar_t* v = view + idx * 3;
        const scalar_t* l = light + idx * 3;
        scalar_t r = roughness[idx * roughness_stride];
        const scalar_t* f = f0 + idx * f0_stride * 3;

        for (int c = 0; c < 3; ++c) {
            output[idx * 3 + c] = cook_torrance_scalar<scalar_t>(n, v, l, r, f[c]);
        }
    }
}

/**
 * CUDA kernel for Cook-Torrance backward pass (scalar f0).
 * Uses strided access for roughness and f0 to avoid unnecessary memory allocation.
 */
template <typename scalar_t>
__global__ void cook_torrance_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ normal,
    const scalar_t* __restrict__ view,
    const scalar_t* __restrict__ light,
    const scalar_t* __restrict__ roughness,
    const scalar_t* __restrict__ f0,
    scalar_t* __restrict__ grad_normal,
    scalar_t* __restrict__ grad_view,
    scalar_t* __restrict__ grad_light,
    scalar_t* __restrict__ grad_roughness,
    scalar_t* __restrict__ grad_f0,
    int64_t numel,
    int64_t roughness_stride,
    int64_t f0_stride
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        const scalar_t* n = normal + idx * 3;
        const scalar_t* v = view + idx * 3;
        const scalar_t* l = light + idx * 3;
        scalar_t r = roughness[idx * roughness_stride];
        scalar_t f = f0[idx * f0_stride];
        scalar_t g = grad_output[idx];

        scalar_t* gn = grad_normal + idx * 3;
        scalar_t* gv = grad_view + idx * 3;
        scalar_t* gl = grad_light + idx * 3;

        cook_torrance_backward_scalar<scalar_t>(
            g, n, v, l, r, f,
            gn, gv, gl, grad_roughness + idx, grad_f0 + idx
        );
    }
}

/**
 * CUDA kernel for Cook-Torrance backward pass (RGB f0).
 * Uses strided access for roughness and f0 to avoid unnecessary memory allocation.
 */
template <typename scalar_t>
__global__ void cook_torrance_backward_rgb_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ normal,
    const scalar_t* __restrict__ view,
    const scalar_t* __restrict__ light,
    const scalar_t* __restrict__ roughness,
    const scalar_t* __restrict__ f0,
    scalar_t* __restrict__ grad_normal,
    scalar_t* __restrict__ grad_view,
    scalar_t* __restrict__ grad_light,
    scalar_t* __restrict__ grad_roughness,
    scalar_t* __restrict__ grad_f0,
    int64_t numel,
    int64_t roughness_stride,
    int64_t f0_stride
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        const scalar_t* n = normal + idx * 3;
        const scalar_t* v = view + idx * 3;
        const scalar_t* l = light + idx * 3;
        scalar_t r = roughness[idx * roughness_stride];
        const scalar_t* f = f0 + idx * f0_stride * 3;
        const scalar_t* g = grad_output + idx * 3;

        scalar_t* gn = grad_normal + idx * 3;
        scalar_t* gv = grad_view + idx * 3;
        scalar_t* gl = grad_light + idx * 3;
        scalar_t* gr = grad_roughness + idx;
        scalar_t* gf = grad_f0 + idx * 3;

        // Initialize gradients
        gn[0] = gn[1] = gn[2] = scalar_t(0);
        gv[0] = gv[1] = gv[2] = scalar_t(0);
        gl[0] = gl[1] = gl[2] = scalar_t(0);
        *gr = scalar_t(0);

        // Accumulate gradients from each channel
        for (int c = 0; c < 3; ++c) {
            scalar_t temp_gn[3], temp_gv[3], temp_gl[3];
            scalar_t temp_gr, temp_gf;

            cook_torrance_backward_scalar<scalar_t>(
                g[c], n, v, l, r, f[c],
                temp_gn, temp_gv, temp_gl, &temp_gr, &temp_gf
            );

            for (int i = 0; i < 3; ++i) {
                gn[i] += temp_gn[i];
                gv[i] += temp_gv[i];
                gl[i] += temp_gl[i];
            }
            *gr += temp_gr;
            gf[c] = temp_gf;
        }
    }
}

/**
 * CUDA kernel for Cook-Torrance backward_backward pass (scalar f0).
 * Uses strided access for roughness and f0 to avoid unnecessary memory allocation.
 */
template <typename scalar_t>
__global__ void cook_torrance_backward_backward_kernel(
    const scalar_t* __restrict__ gg_normal,
    const scalar_t* __restrict__ gg_view,
    const scalar_t* __restrict__ gg_light,
    const scalar_t* __restrict__ gg_roughness,
    const scalar_t* __restrict__ gg_f0,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ normal,
    const scalar_t* __restrict__ view,
    const scalar_t* __restrict__ light,
    const scalar_t* __restrict__ roughness,
    const scalar_t* __restrict__ f0,
    scalar_t* __restrict__ grad_grad_output,
    scalar_t* __restrict__ grad2_normal,
    scalar_t* __restrict__ grad2_view,
    scalar_t* __restrict__ grad2_light,
    scalar_t* __restrict__ grad2_roughness,
    scalar_t* __restrict__ grad2_f0,
    int64_t numel,
    int64_t roughness_stride,
    int64_t gg_roughness_stride,
    int64_t f0_stride,
    int64_t gg_f0_stride
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        const scalar_t* n = normal + idx * 3;
        const scalar_t* v = view + idx * 3;
        const scalar_t* l = light + idx * 3;
        scalar_t r = roughness[idx * roughness_stride];
        scalar_t f = f0[idx * f0_stride];
        scalar_t g = grad_output[idx];

        const scalar_t* gg_n = gg_normal + idx * 3;
        const scalar_t* gg_v = gg_view + idx * 3;
        const scalar_t* gg_l = gg_light + idx * 3;
        scalar_t gg_r = gg_roughness[idx * gg_roughness_stride];
        scalar_t gg_f = gg_f0[idx * gg_f0_stride];

        cook_torrance_backward_backward_scalar<scalar_t>(
            gg_n[0], gg_n[1], gg_n[2],
            gg_v[0], gg_v[1], gg_v[2],
            gg_l[0], gg_l[1], gg_l[2],
            gg_r,
            gg_f,
            g,
            n, v, l, r, f,
            grad_grad_output + idx,
            grad2_normal + idx * 3,
            grad2_view + idx * 3,
            grad2_light + idx * 3,
            grad2_roughness + idx,
            grad2_f0 + idx
        );
    }
}

/**
 * CUDA kernel for Cook-Torrance backward_backward pass (RGB f0).
 * Uses strided access for roughness and f0 to avoid unnecessary memory allocation.
 */
template <typename scalar_t>
__global__ void cook_torrance_backward_backward_rgb_kernel(
    const scalar_t* __restrict__ gg_normal,
    const scalar_t* __restrict__ gg_view,
    const scalar_t* __restrict__ gg_light,
    const scalar_t* __restrict__ gg_roughness,
    const scalar_t* __restrict__ gg_f0,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ normal,
    const scalar_t* __restrict__ view,
    const scalar_t* __restrict__ light,
    const scalar_t* __restrict__ roughness,
    const scalar_t* __restrict__ f0,
    scalar_t* __restrict__ grad_grad_output,
    scalar_t* __restrict__ grad2_normal,
    scalar_t* __restrict__ grad2_view,
    scalar_t* __restrict__ grad2_light,
    scalar_t* __restrict__ grad2_roughness,
    scalar_t* __restrict__ grad2_f0,
    int64_t numel,
    int64_t roughness_stride,
    int64_t gg_roughness_stride,
    int64_t f0_stride,
    int64_t gg_f0_stride
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        const scalar_t* n = normal + idx * 3;
        const scalar_t* v = view + idx * 3;
        const scalar_t* l = light + idx * 3;
        scalar_t r = roughness[idx * roughness_stride];
        const scalar_t* f = f0 + idx * f0_stride * 3;

        const scalar_t* gg_n = gg_normal + idx * 3;
        const scalar_t* gg_v = gg_view + idx * 3;
        const scalar_t* gg_l = gg_light + idx * 3;
        scalar_t gg_r = gg_roughness[idx * gg_roughness_stride];
        const scalar_t* gg_f = gg_f0 + idx * gg_f0_stride * 3;
        const scalar_t* g = grad_output + idx * 3;

        scalar_t* ggo = grad_grad_output + idx * 3;
        scalar_t* g2n = grad2_normal + idx * 3;
        scalar_t* g2v = grad2_view + idx * 3;
        scalar_t* g2l = grad2_light + idx * 3;
        scalar_t* g2r = grad2_roughness + idx;
        scalar_t* g2f = grad2_f0 + idx * 3;

        // Initialize outputs
        ggo[0] = ggo[1] = ggo[2] = scalar_t(0);
        g2n[0] = g2n[1] = g2n[2] = scalar_t(0);
        g2v[0] = g2v[1] = g2v[2] = scalar_t(0);
        g2l[0] = g2l[1] = g2l[2] = scalar_t(0);
        *g2r = scalar_t(0);

        // Process each channel
        for (int c = 0; c < 3; ++c) {
            scalar_t temp_ggo, temp_g2n[3], temp_g2v[3], temp_g2l[3], temp_g2r, temp_g2f;

            cook_torrance_backward_backward_scalar<scalar_t>(
                gg_n[0], gg_n[1], gg_n[2],
                gg_v[0], gg_v[1], gg_v[2],
                gg_l[0], gg_l[1], gg_l[2],
                gg_r,
                gg_f[c],
                g[c],
                n, v, l, r, f[c],
                &temp_ggo, temp_g2n, temp_g2v, temp_g2l, &temp_g2r, &temp_g2f
            );

            ggo[c] += temp_ggo;
            for (int i = 0; i < 3; ++i) {
                g2n[i] += temp_g2n[i];
                g2v[i] += temp_g2v[i];
                g2l[i] += temp_g2l[i];
            }
            *g2r += temp_g2r;
            g2f[c] = temp_g2f;
        }
    }
}

namespace {

/**
 * Helper to compute broadcast batch shape.
 */
inline std::vector<int64_t> compute_batch_shape(
    c10::IntArrayRef normal_batch,
    c10::IntArrayRef view_batch,
    c10::IntArrayRef light_batch,
    c10::IntArrayRef roughness_batch,
    c10::IntArrayRef f0_batch
) {
    auto max_batch_dim = std::max({
        (int64_t)normal_batch.size(),
        (int64_t)view_batch.size(),
        (int64_t)light_batch.size(),
        (int64_t)roughness_batch.size(),
        (int64_t)f0_batch.size()
    });

    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < max_batch_dim; ++i) {
        int64_t dim = 1;
        auto get_dim = [&](c10::IntArrayRef shape, int64_t offset) -> int64_t {
            int64_t idx = (int64_t)shape.size() - max_batch_dim + offset;
            return idx >= 0 ? shape[idx] : 1;
        };
        dim = std::max(dim, get_dim(normal_batch, i));
        dim = std::max(dim, get_dim(view_batch, i));
        dim = std::max(dim, get_dim(light_batch, i));
        dim = std::max(dim, get_dim(roughness_batch, i));
        dim = std::max(dim, get_dim(f0_batch, i));
        batch_shape.push_back(dim);
    }
    return batch_shape;
}

}  // anonymous namespace

/**
 * CUDA implementation of Cook-Torrance forward pass.
 */
inline at::Tensor cook_torrance(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    TORCH_CHECK(normal.size(-1) == 3, "cook_torrance: normal must have last dimension 3");
    TORCH_CHECK(view.size(-1) == 3, "cook_torrance: view must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "cook_torrance: light must have last dimension 3");

    c10::cuda::CUDAGuard device_guard(normal.device());

    bool f0_is_rgb = f0.dim() > 0 && f0.size(-1) == 3;

    auto normal_batch = normal.sizes().slice(0, normal.dim() - 1);
    auto view_batch = view.sizes().slice(0, view.dim() - 1);
    auto light_batch = light.sizes().slice(0, light.dim() - 1);
    auto roughness_batch = roughness.sizes();
    auto f0_batch = f0_is_rgb ? f0.sizes().slice(0, f0.dim() - 1) : f0.sizes();

    auto batch_shape = compute_batch_shape(normal_batch, view_batch, light_batch, roughness_batch, f0_batch);

    int64_t numel = 1;
    for (auto d : batch_shape) {
        numel *= d;
    }

    std::vector<int64_t> vec_shape = batch_shape;
    vec_shape.push_back(3);

    at::Tensor normal_expanded = normal.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor view_expanded = view.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor light_expanded = light.expand(vec_shape).contiguous().view({numel, 3});

    // Optimize: avoid full expansion for scalar/broadcasted roughness and f0
    bool roughness_is_scalar = roughness.numel() == 1;
    bool f0_is_scalar = !f0_is_rgb && f0.numel() == 1;

    at::Tensor roughness_data = roughness_is_scalar ? roughness.contiguous().view({1}) : roughness.expand(batch_shape).contiguous().view({numel});
    int64_t roughness_stride = roughness_is_scalar ? 0 : 1;

    at::Tensor f0_data;
    int64_t f0_stride;
    if (f0_is_rgb) {
        bool f0_rgb_is_scalar = f0.numel() == 3;
        f0_data = f0_rgb_is_scalar ? f0.contiguous().view({1, 3}) : f0.expand(vec_shape).contiguous().view({numel, 3});
        f0_stride = f0_rgb_is_scalar ? 0 : 1;
    } else {
        f0_data = f0_is_scalar ? f0.contiguous().view({1}) : f0.expand(batch_shape).contiguous().view({numel});
        f0_stride = f0_is_scalar ? 0 : 1;
    }

    at::Tensor output;
    if (f0_is_rgb) {
        output = at::empty({numel, 3}, normal.options());
    } else {
        output = at::empty({numel}, normal.options());
    }

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        normal.scalar_type(),
        "cook_torrance_cuda",
        [&]() {
            if (f0_is_rgb) {
                cook_torrance_forward_rgb_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    normal_expanded.data_ptr<scalar_t>(),
                    view_expanded.data_ptr<scalar_t>(),
                    light_expanded.data_ptr<scalar_t>(),
                    roughness_data.data_ptr<scalar_t>(),
                    f0_data.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    numel,
                    roughness_stride,
                    f0_stride
                );
            } else {
                cook_torrance_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    normal_expanded.data_ptr<scalar_t>(),
                    view_expanded.data_ptr<scalar_t>(),
                    light_expanded.data_ptr<scalar_t>(),
                    roughness_data.data_ptr<scalar_t>(),
                    f0_data.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    numel,
                    roughness_stride,
                    f0_stride
                );
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    if (f0_is_rgb) {
        std::vector<int64_t> out_shape = batch_shape;
        out_shape.push_back(3);
        return output.view(out_shape);
    } else {
        return output.view(batch_shape);
    }
}

/**
 * CUDA implementation of Cook-Torrance backward pass.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cook_torrance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    c10::cuda::CUDAGuard device_guard(normal.device());

    bool f0_is_rgb = f0.dim() > 0 && f0.size(-1) == 3;

    auto normal_batch = normal.sizes().slice(0, normal.dim() - 1);
    auto view_batch = view.sizes().slice(0, view.dim() - 1);
    auto light_batch = light.sizes().slice(0, light.dim() - 1);
    auto roughness_batch = roughness.sizes();
    auto f0_batch = f0_is_rgb ? f0.sizes().slice(0, f0.dim() - 1) : f0.sizes();

    auto batch_shape = compute_batch_shape(normal_batch, view_batch, light_batch, roughness_batch, f0_batch);

    int64_t numel = 1;
    for (auto d : batch_shape) {
        numel *= d;
    }

    std::vector<int64_t> vec_shape = batch_shape;
    vec_shape.push_back(3);

    at::Tensor normal_expanded = normal.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor view_expanded = view.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor light_expanded = light.expand(vec_shape).contiguous().view({numel, 3});

    // Optimize: avoid full expansion for scalar/broadcasted roughness and f0
    bool roughness_is_scalar = roughness.numel() == 1;
    bool f0_is_scalar = !f0_is_rgb && f0.numel() == 1;

    at::Tensor roughness_data = roughness_is_scalar ? roughness.contiguous().view({1}) : roughness.expand(batch_shape).contiguous().view({numel});
    int64_t roughness_stride = roughness_is_scalar ? 0 : 1;

    at::Tensor f0_data;
    at::Tensor grad_expanded;
    int64_t f0_stride;
    if (f0_is_rgb) {
        bool f0_rgb_is_scalar = f0.numel() == 3;
        f0_data = f0_rgb_is_scalar ? f0.contiguous().view({1, 3}) : f0.expand(vec_shape).contiguous().view({numel, 3});
        f0_stride = f0_rgb_is_scalar ? 0 : 1;
        grad_expanded = grad_output.contiguous().view({numel, 3});
    } else {
        f0_data = f0_is_scalar ? f0.contiguous().view({1}) : f0.expand(batch_shape).contiguous().view({numel});
        f0_stride = f0_is_scalar ? 0 : 1;
        grad_expanded = grad_output.contiguous().view({numel});
    }

    at::Tensor grad_normal = at::zeros({numel, 3}, normal.options());
    at::Tensor grad_view = at::zeros({numel, 3}, normal.options());
    at::Tensor grad_light = at::zeros({numel, 3}, normal.options());
    at::Tensor grad_roughness = at::zeros({numel}, normal.options());
    at::Tensor grad_f0;
    if (f0_is_rgb) {
        grad_f0 = at::zeros({numel, 3}, normal.options());
    } else {
        grad_f0 = at::zeros({numel}, normal.options());
    }

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        normal.scalar_type(),
        "cook_torrance_backward_cuda",
        [&]() {
            if (f0_is_rgb) {
                cook_torrance_backward_rgb_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    grad_expanded.data_ptr<scalar_t>(),
                    normal_expanded.data_ptr<scalar_t>(),
                    view_expanded.data_ptr<scalar_t>(),
                    light_expanded.data_ptr<scalar_t>(),
                    roughness_data.data_ptr<scalar_t>(),
                    f0_data.data_ptr<scalar_t>(),
                    grad_normal.data_ptr<scalar_t>(),
                    grad_view.data_ptr<scalar_t>(),
                    grad_light.data_ptr<scalar_t>(),
                    grad_roughness.data_ptr<scalar_t>(),
                    grad_f0.data_ptr<scalar_t>(),
                    numel,
                    roughness_stride,
                    f0_stride
                );
            } else {
                cook_torrance_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    grad_expanded.data_ptr<scalar_t>(),
                    normal_expanded.data_ptr<scalar_t>(),
                    view_expanded.data_ptr<scalar_t>(),
                    light_expanded.data_ptr<scalar_t>(),
                    roughness_data.data_ptr<scalar_t>(),
                    f0_data.data_ptr<scalar_t>(),
                    grad_normal.data_ptr<scalar_t>(),
                    grad_view.data_ptr<scalar_t>(),
                    grad_light.data_ptr<scalar_t>(),
                    grad_roughness.data_ptr<scalar_t>(),
                    grad_f0.data_ptr<scalar_t>(),
                    numel,
                    roughness_stride,
                    f0_stride
                );
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return std::make_tuple(
        grad_normal.view(vec_shape),
        grad_view.view(vec_shape),
        grad_light.view(vec_shape),
        grad_roughness.view(batch_shape),
        f0_is_rgb ? grad_f0.view(vec_shape) : grad_f0.view(batch_shape)
    );
}

/**
 * CUDA implementation of Cook-Torrance backward_backward pass.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cook_torrance_backward_backward(
    const at::Tensor& gg_normal,
    const at::Tensor& gg_view,
    const at::Tensor& gg_light,
    const at::Tensor& gg_roughness,
    const at::Tensor& gg_f0,
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    c10::cuda::CUDAGuard device_guard(normal.device());

    bool f0_is_rgb = f0.dim() > 0 && f0.size(-1) == 3;

    auto normal_batch = normal.sizes().slice(0, normal.dim() - 1);
    auto view_batch = view.sizes().slice(0, view.dim() - 1);
    auto light_batch = light.sizes().slice(0, light.dim() - 1);
    auto roughness_batch = roughness.sizes();
    auto f0_batch = f0_is_rgb ? f0.sizes().slice(0, f0.dim() - 1) : f0.sizes();

    auto batch_shape = compute_batch_shape(normal_batch, view_batch, light_batch, roughness_batch, f0_batch);

    int64_t numel = 1;
    for (auto d : batch_shape) {
        numel *= d;
    }

    std::vector<int64_t> vec_shape = batch_shape;
    vec_shape.push_back(3);

    at::Tensor normal_expanded = normal.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor view_expanded = view.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor light_expanded = light.expand(vec_shape).contiguous().view({numel, 3});

    at::Tensor gg_normal_expanded = gg_normal.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor gg_view_expanded = gg_view.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor gg_light_expanded = gg_light.expand(vec_shape).contiguous().view({numel, 3});

    // Optimize: avoid full expansion for scalar/broadcasted roughness and f0
    bool roughness_is_scalar = roughness.numel() == 1;
    bool gg_roughness_is_scalar = gg_roughness.numel() == 1;
    bool f0_is_scalar = !f0_is_rgb && f0.numel() == 1;

    at::Tensor roughness_data = roughness_is_scalar ? roughness.contiguous().view({1}) : roughness.expand(batch_shape).contiguous().view({numel});
    int64_t roughness_stride = roughness_is_scalar ? 0 : 1;

    at::Tensor gg_roughness_data = gg_roughness_is_scalar ? gg_roughness.contiguous().view({1}) : gg_roughness.expand(batch_shape).contiguous().view({numel});
    int64_t gg_roughness_stride = gg_roughness_is_scalar ? 0 : 1;

    at::Tensor f0_data;
    at::Tensor grad_expanded;
    at::Tensor gg_f0_data;
    int64_t f0_stride;
    int64_t gg_f0_stride;
    if (f0_is_rgb) {
        bool f0_rgb_is_scalar = f0.numel() == 3;
        bool gg_f0_rgb_is_scalar = gg_f0.numel() == 3;
        f0_data = f0_rgb_is_scalar ? f0.contiguous().view({1, 3}) : f0.expand(vec_shape).contiguous().view({numel, 3});
        f0_stride = f0_rgb_is_scalar ? 0 : 1;
        gg_f0_data = gg_f0_rgb_is_scalar ? gg_f0.contiguous().view({1, 3}) : gg_f0.expand(vec_shape).contiguous().view({numel, 3});
        gg_f0_stride = gg_f0_rgb_is_scalar ? 0 : 1;
        grad_expanded = grad_output.expand(vec_shape).contiguous().view({numel, 3});
    } else {
        bool gg_f0_is_scalar = gg_f0.numel() == 1;
        f0_data = f0_is_scalar ? f0.contiguous().view({1}) : f0.expand(batch_shape).contiguous().view({numel});
        f0_stride = f0_is_scalar ? 0 : 1;
        gg_f0_data = gg_f0_is_scalar ? gg_f0.contiguous().view({1}) : gg_f0.expand(batch_shape).contiguous().view({numel});
        gg_f0_stride = gg_f0_is_scalar ? 0 : 1;
        grad_expanded = grad_output.expand(batch_shape).contiguous().view({numel});
    }

    at::Tensor grad_grad_output;
    if (f0_is_rgb) {
        grad_grad_output = at::zeros({numel, 3}, normal.options());
    } else {
        grad_grad_output = at::zeros({numel}, normal.options());
    }
    at::Tensor grad2_normal = at::zeros({numel, 3}, normal.options());
    at::Tensor grad2_view = at::zeros({numel, 3}, normal.options());
    at::Tensor grad2_light = at::zeros({numel, 3}, normal.options());
    at::Tensor grad2_roughness = at::zeros({numel}, normal.options());
    at::Tensor grad2_f0;
    if (f0_is_rgb) {
        grad2_f0 = at::zeros({numel, 3}, normal.options());
    } else {
        grad2_f0 = at::zeros({numel}, normal.options());
    }

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        normal.scalar_type(),
        "cook_torrance_backward_backward_cuda",
        [&]() {
            if (f0_is_rgb) {
                cook_torrance_backward_backward_rgb_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    gg_normal_expanded.data_ptr<scalar_t>(),
                    gg_view_expanded.data_ptr<scalar_t>(),
                    gg_light_expanded.data_ptr<scalar_t>(),
                    gg_roughness_data.data_ptr<scalar_t>(),
                    gg_f0_data.data_ptr<scalar_t>(),
                    grad_expanded.data_ptr<scalar_t>(),
                    normal_expanded.data_ptr<scalar_t>(),
                    view_expanded.data_ptr<scalar_t>(),
                    light_expanded.data_ptr<scalar_t>(),
                    roughness_data.data_ptr<scalar_t>(),
                    f0_data.data_ptr<scalar_t>(),
                    grad_grad_output.data_ptr<scalar_t>(),
                    grad2_normal.data_ptr<scalar_t>(),
                    grad2_view.data_ptr<scalar_t>(),
                    grad2_light.data_ptr<scalar_t>(),
                    grad2_roughness.data_ptr<scalar_t>(),
                    grad2_f0.data_ptr<scalar_t>(),
                    numel,
                    roughness_stride,
                    gg_roughness_stride,
                    f0_stride,
                    gg_f0_stride
                );
            } else {
                cook_torrance_backward_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    gg_normal_expanded.data_ptr<scalar_t>(),
                    gg_view_expanded.data_ptr<scalar_t>(),
                    gg_light_expanded.data_ptr<scalar_t>(),
                    gg_roughness_data.data_ptr<scalar_t>(),
                    gg_f0_data.data_ptr<scalar_t>(),
                    grad_expanded.data_ptr<scalar_t>(),
                    normal_expanded.data_ptr<scalar_t>(),
                    view_expanded.data_ptr<scalar_t>(),
                    light_expanded.data_ptr<scalar_t>(),
                    roughness_data.data_ptr<scalar_t>(),
                    f0_data.data_ptr<scalar_t>(),
                    grad_grad_output.data_ptr<scalar_t>(),
                    grad2_normal.data_ptr<scalar_t>(),
                    grad2_view.data_ptr<scalar_t>(),
                    grad2_light.data_ptr<scalar_t>(),
                    grad2_roughness.data_ptr<scalar_t>(),
                    grad2_f0.data_ptr<scalar_t>(),
                    numel,
                    roughness_stride,
                    gg_roughness_stride,
                    f0_stride,
                    gg_f0_stride
                );
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return std::make_tuple(
        f0_is_rgb ? grad_grad_output.view(vec_shape) : grad_grad_output.view(batch_shape),
        grad2_normal.view(vec_shape),
        grad2_view.view(vec_shape),
        grad2_light.view(vec_shape),
        grad2_roughness.view(batch_shape),
        f0_is_rgb ? grad2_f0.view(vec_shape) : grad2_f0.view(batch_shape)
    );
}

}  // namespace torchscience::cuda::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("cook_torrance", &torchscience::cuda::graphics::shading::cook_torrance);
    module.impl("cook_torrance_backward", &torchscience::cuda::graphics::shading::cook_torrance_backward);
    module.impl("cook_torrance_backward_backward", &torchscience::cuda::graphics::shading::cook_torrance_backward_backward);
}
