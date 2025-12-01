#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "../../impl/optimization/test_functions.h"

namespace torchscience::cuda::test_functions {

using torchscience::impl::optimization::test_functions::check_rosenbrock_input;

/**
 * CUDA kernel for rosenbrock function.
 */
template <typename scalar_t>
__global__ void rosenbrock_kernel(
    const scalar_t* __restrict__ x,
    scalar_t a,
    scalar_t b,
    scalar_t* __restrict__ output,
    int64_t batch_size,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        const scalar_t* x_ptr = x + idx * n;
        scalar_t sum = scalar_t(0);

        for (int64_t i = 0; i < n - 1; ++i) {
            scalar_t x_i = x_ptr[i];
            scalar_t x_i_plus_1 = x_ptr[i + 1];
            scalar_t x_i_sq = x_i * x_i;
            scalar_t term1 = (a - x_i) * (a - x_i);
            scalar_t diff = x_i_plus_1 - x_i_sq;
            scalar_t term2 = b * diff * diff;
            sum += term1 + term2;
        }

        output[idx] = sum;
    }
}

/**
 * CUDA kernel for rosenbrock gradient (internal).
 */
template <typename scalar_t>
__global__ void rosenbrock_gradient_kernel(
    const scalar_t* __restrict__ x,
    scalar_t a,
    scalar_t b,
    scalar_t* __restrict__ grad,
    int64_t batch_size,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        const scalar_t* x_ptr = x + idx * n;
        scalar_t* grad_ptr = grad + idx * n;

        for (int64_t i = 0; i < n; ++i) {
            scalar_t grad_i = scalar_t(0);

            if (i < n - 1) {
                scalar_t x_i = x_ptr[i];
                scalar_t x_i_plus_1 = x_ptr[i + 1];
                scalar_t x_i_sq = x_i * x_i;
                scalar_t diff = x_i_plus_1 - x_i_sq;
                grad_i += scalar_t(-2) * (a - x_i);
                grad_i += scalar_t(-4) * b * x_i * diff;
            }

            if (i > 0) {
                scalar_t x_i_minus_1 = x_ptr[i - 1];
                scalar_t x_i = x_ptr[i];
                scalar_t x_i_minus_1_sq = x_i_minus_1 * x_i_minus_1;
                grad_i += scalar_t(2) * b * (x_i - x_i_minus_1_sq);
            }

            grad_ptr[i] = grad_i;
        }
    }
}

/**
 * CUDA kernel for rosenbrock Hessian (internal).
 */
template <typename scalar_t>
__global__ void rosenbrock_hessian_kernel(
    const scalar_t* __restrict__ x,
    scalar_t b,
    scalar_t* __restrict__ H,
    int64_t batch_size,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        const scalar_t* x_ptr = x + idx * n;
        scalar_t* H_ptr = H + idx * n * n;

        for (int64_t i = 0; i < n * n; ++i) {
            H_ptr[i] = scalar_t(0);
        }

        for (int64_t i = 0; i < n; ++i) {
            scalar_t diag_val;

            if (i == 0) {
                scalar_t x_0 = x_ptr[0];
                scalar_t x_1 = x_ptr[1];
                diag_val = scalar_t(2) + scalar_t(12) * b * x_0 * x_0
                           - scalar_t(4) * b * x_1;
            } else if (i == n - 1) {
                diag_val = scalar_t(2) * b;
            } else {
                scalar_t x_i = x_ptr[i];
                scalar_t x_i_plus_1 = x_ptr[i + 1];
                diag_val = scalar_t(2) + scalar_t(2) * b
                           + scalar_t(12) * b * x_i * x_i
                           - scalar_t(4) * b * x_i_plus_1;
            }

            H_ptr[i * n + i] = diag_val;

            if (i < n - 1) {
                scalar_t off_diag_val = scalar_t(-4) * b * x_ptr[i];
                H_ptr[i * n + (i + 1)] = off_diag_val;
                H_ptr[(i + 1) * n + i] = off_diag_val;
            }
        }
    }
}

namespace {

// Internal helper to compute gradient
inline at::Tensor compute_gradient(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t n = x.size(-1);
    const int64_t batch_size = x.numel() / n;

    at::Tensor x_contig = x.contiguous();
    at::Tensor x_flat = x_contig.view({batch_size, n});

    bool a_is_scalar = (a.numel() == 1);
    bool b_is_scalar = (b.numel() == 1);

    if (!a_is_scalar || !b_is_scalar) {
        at::Tensor x_i = x.narrow(-1, 0, n - 1);
        at::Tensor x_i_plus_1 = x.narrow(-1, 1, n - 1);
        at::Tensor x_i_sq = at::pow(x_i, 2);
        at::Tensor diff = x_i_plus_1 - x_i_sq;

        at::Tensor grad = at::zeros_like(x);
        at::Tensor term1 = -2 * (a - x_i) - 4 * b * x_i * diff;
        grad.narrow(-1, 0, n - 1).add_(term1);

        at::Tensor x_prev_sq = at::pow(x.narrow(-1, 0, n - 1), 2);
        at::Tensor term2 = 2 * b * (x.narrow(-1, 1, n - 1) - x_prev_sq);
        grad.narrow(-1, 1, n - 1).add_(term2);

        return grad;
    }

    at::Tensor output = at::zeros({batch_size, n}, x.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        x.scalar_type(),
        "rosenbrock_gradient_cuda",
        [&]() {
            scalar_t a_val = a.item<scalar_t>();
            scalar_t b_val = b.item<scalar_t>();

            rosenbrock_gradient_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                x_flat.data_ptr<scalar_t>(),
                a_val,
                b_val,
                output.data_ptr<scalar_t>(),
                batch_size,
                n
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return output.view_as(x);
}

// Internal helper to compute Hessian
inline at::Tensor compute_hessian(
    const at::Tensor& x,
    const at::Tensor& b
) {
    const int64_t n = x.size(-1);
    const int64_t batch_size = x.numel() / n;

    std::vector<int64_t> output_shape(x.sizes().begin(), x.sizes().end() - 1);
    output_shape.push_back(n);
    output_shape.push_back(n);

    at::Tensor x_contig = x.contiguous();
    at::Tensor x_flat = x_contig.view({batch_size, n});

    bool b_is_scalar = (b.numel() == 1);

    if (!b_is_scalar) {
        at::Tensor H = at::zeros(output_shape, x.options());
        at::Tensor x_slice = x.narrow(-1, 0, n - 1);

        at::Tensor x_0 = x.select(-1, 0);
        at::Tensor x_1 = x.select(-1, 1);
        at::Tensor diag_0 = 2 + 12 * b * at::pow(x_0, 2) - 4 * b * x_1;

        at::Tensor x_mid = x.narrow(-1, 1, n - 2);
        at::Tensor x_mid_next = x.narrow(-1, 2, n - 2);
        at::Tensor diag_mid = 2 + 2 * b + 12 * b * at::pow(x_mid, 2) - 4 * b * x_mid_next;

        at::Tensor diag_last = 2 * b * at::ones_like(x.select(-1, -1));

        H.index_put_({at::indexing::Ellipsis, 0, 0}, diag_0);
        for (int64_t i = 1; i < n - 1; ++i) {
            H.index_put_({at::indexing::Ellipsis, i, i}, diag_mid.select(-1, i - 1));
        }
        H.index_put_({at::indexing::Ellipsis, n - 1, n - 1}, diag_last);

        at::Tensor off_diag = -4 * b * x_slice;
        for (int64_t i = 0; i < n - 1; ++i) {
            at::Tensor val = off_diag.select(-1, i);
            H.index_put_({at::indexing::Ellipsis, i, i + 1}, val);
            H.index_put_({at::indexing::Ellipsis, i + 1, i}, val);
        }

        return H;
    }

    at::Tensor output = at::zeros({batch_size, n, n}, x.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        x.scalar_type(),
        "rosenbrock_hessian_cuda",
        [&]() {
            scalar_t b_val = b.item<scalar_t>();

            rosenbrock_hessian_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                x_flat.data_ptr<scalar_t>(),
                b_val,
                output.data_ptr<scalar_t>(),
                batch_size,
                n
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return output.view(output_shape);
}

}  // anonymous namespace

/**
 * CUDA implementation of rosenbrock.
 */
inline at::Tensor rosenbrock(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    check_rosenbrock_input(x, "rosenbrock");

    c10::cuda::CUDAGuard device_guard(x.device());

    const int64_t n = x.size(-1);
    const int64_t batch_size = x.numel() / n;

    std::vector<int64_t> output_shape(x.sizes().begin(), x.sizes().end() - 1);
    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    at::Tensor x_contig = x.contiguous();
    at::Tensor x_flat = x_contig.view({batch_size, n});

    bool a_is_scalar = (a.numel() == 1);
    bool b_is_scalar = (b.numel() == 1);

    if (!a_is_scalar || !b_is_scalar) {
        at::Tensor x_i = x.narrow(-1, 0, n - 1);
        at::Tensor x_i_plus_1 = x.narrow(-1, 1, n - 1);
        at::Tensor term1 = at::pow(a - x_i, 2);
        at::Tensor term2 = b * at::pow(x_i_plus_1 - at::pow(x_i, 2), 2);
        return at::sum(term1 + term2, -1);
    }

    at::Tensor output = at::empty({batch_size}, x.options());

    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        x.scalar_type(),
        "rosenbrock_cuda",
        [&]() {
            scalar_t a_val = a.item<scalar_t>();
            scalar_t b_val = b.item<scalar_t>();

            rosenbrock_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                x_flat.data_ptr<scalar_t>(),
                a_val,
                b_val,
                output.data_ptr<scalar_t>(),
                batch_size,
                n
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    if (x.dim() == 1) {
        return output.squeeze(0);
    }
    return output.view(output_shape);
}

/**
 * CUDA implementation of rosenbrock_backward.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> rosenbrock_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t n = x.size(-1);

    at::Tensor grad_x_local = compute_gradient(x, a, b);
    at::Tensor grad_x = grad_output.unsqueeze(-1) * grad_x_local;

    at::Tensor x_i = x.narrow(-1, 0, n - 1);
    at::Tensor df_da = at::sum(2 * (a - x_i), -1);
    at::Tensor grad_a = grad_output * df_da;

    while (grad_a.dim() > a.dim()) {
        grad_a = grad_a.sum(0);
    }
    for (int64_t i = 0; i < a.dim(); ++i) {
        if (a.size(i) == 1 && grad_a.size(i) > 1) {
            grad_a = grad_a.sum(i, true);
        }
    }

    at::Tensor x_i_plus_1 = x.narrow(-1, 1, n - 1);
    at::Tensor diff = x_i_plus_1 - at::pow(x_i, 2);
    at::Tensor df_db = at::sum(at::pow(diff, 2), -1);
    at::Tensor grad_b = grad_output * df_db;

    while (grad_b.dim() > b.dim()) {
        grad_b = grad_b.sum(0);
    }
    for (int64_t i = 0; i < b.dim(); ++i) {
        if (b.size(i) == 1 && grad_b.size(i) > 1) {
            grad_b = grad_b.sum(i, true);
        }
    }

    return std::make_tuple(grad_x, grad_a, grad_b);
}

/**
 * CUDA implementation of rosenbrock_backward_backward.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rosenbrock_backward_backward(
    const at::Tensor& grad_grad_x,
    const at::Tensor& grad_grad_a,
    const at::Tensor& grad_grad_b,
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t n = x.size(-1);

    at::Tensor grad_grad_output = at::zeros_like(grad_output);
    at::Tensor grad_x = at::zeros_like(x);
    at::Tensor grad_a = at::zeros_like(a);
    at::Tensor grad_b = at::zeros_like(b);

    if (grad_grad_x.defined()) {
        at::Tensor H = compute_hessian(x, b);
        at::Tensor Hv = at::matmul(H, grad_grad_x.unsqueeze(-1)).squeeze(-1);
        grad_x = grad_x + grad_output.unsqueeze(-1) * Hv;

        at::Tensor grad_x_local = compute_gradient(x, a, b);
        grad_grad_output = grad_grad_output + at::sum(grad_grad_x * grad_x_local, -1);
    }

    if (grad_grad_a.defined()) {
        at::Tensor d2f_dxda = at::full({n - 1}, -2.0, x.options());
        at::Tensor contrib = grad_output * grad_grad_a;
        grad_x.narrow(-1, 0, n - 1).add_(contrib.unsqueeze(-1) * d2f_dxda);

        at::Tensor x_i = x.narrow(-1, 0, n - 1);
        at::Tensor df_da = at::sum(2 * (a - x_i), -1);
        grad_grad_output = grad_grad_output + df_da * grad_grad_a;
    }

    if (grad_grad_b.defined()) {
        at::Tensor x_i = x.narrow(-1, 0, n - 1);
        at::Tensor x_i_plus_1 = x.narrow(-1, 1, n - 1);
        at::Tensor diff = x_i_plus_1 - at::pow(x_i, 2);

        at::Tensor d2f_dxdb_i = -4 * x_i * diff;
        at::Tensor d2f_dxdb_i_plus_1 = 2 * diff;

        at::Tensor contrib = grad_output * grad_grad_b;
        grad_x.narrow(-1, 0, n - 1).add_(contrib.unsqueeze(-1) * d2f_dxdb_i);
        grad_x.narrow(-1, 1, n - 1).add_(contrib.unsqueeze(-1) * d2f_dxdb_i_plus_1);

        at::Tensor df_db = at::sum(at::pow(diff, 2), -1);
        grad_grad_output = grad_grad_output + df_db * grad_grad_b;
    }

    return std::make_tuple(grad_grad_output, grad_x, grad_a, grad_b);
}

}  // namespace torchscience::cuda::test_functions

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl("rosenbrock", &torchscience::cuda::test_functions::rosenbrock);
    module.impl("rosenbrock_backward", &torchscience::cuda::test_functions::rosenbrock_backward);
    module.impl("rosenbrock_backward_backward", &torchscience::cuda::test_functions::rosenbrock_backward_backward);
}
