#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_multiply.h"
#include "../../../kernel/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_multiply_backward.h"
#include "../../../kernel/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_multiply_backward_backward.h"

namespace torchscience::cpu::polynomial {

// Forward: a (B, N) x b (B, M) -> output (B, N + M - 1)
inline at::Tensor chebyshev_polynomial_v_multiply(
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(a.dim() >= 1, "a must have at least 1 dimension");
    TORCH_CHECK(b.dim() >= 1, "b must have at least 1 dimension");

    const int64_t N = a.size(-1);
    const int64_t M = b.size(-1);
    const int64_t B_a = a.numel() / N;
    const int64_t B_b = b.numel() / M;

    TORCH_CHECK(B_a == B_b || B_a == 1 || B_b == 1,
        "Batch sizes must match or be broadcastable");

    const int64_t B = std::max(B_a, B_b);
    const int64_t output_N = (N > 0 && M > 0) ? (N + M - 1) : 1;

    auto a_flat = a.reshape({B_a, N}).contiguous();
    auto b_flat = b.reshape({B_b, M}).contiguous();
    auto output = at::empty({B, output_N}, a.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        a.scalar_type(),
        "chebyshev_polynomial_v_multiply",
        [&] {
            const scalar_t* a_ptr = a_flat.data_ptr<scalar_t>();
            const scalar_t* b_ptr = b_flat.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t batch = start; batch < end; ++batch) {
                    const int64_t a_batch = (B_a == 1) ? 0 : batch;
                    const int64_t b_batch = (B_b == 1) ? 0 : batch;

                    kernel::polynomial::chebyshev_polynomial_v_multiply(
                        out_ptr + batch * output_N,
                        a_ptr + a_batch * N,
                        b_ptr + b_batch * M,
                        N,
                        M
                    );
                }
            });
        }
    );

    return output;
}

// Backward: returns (grad_a, grad_b)
inline std::tuple<at::Tensor, at::Tensor> chebyshev_polynomial_v_multiply_backward(
    const at::Tensor& grad_output,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t N = a.size(-1);
    const int64_t M = b.size(-1);
    const int64_t B_a = a.numel() / N;
    const int64_t B_b = b.numel() / M;
    const int64_t B = std::max(B_a, B_b);
    const int64_t output_N = grad_output.size(-1);

    auto grad_output_flat = grad_output.reshape({B, output_N}).contiguous();
    auto a_flat = a.reshape({B_a, N}).contiguous();
    auto b_flat = b.reshape({B_b, M}).contiguous();

    auto grad_a = at::zeros({B, N}, a.options());
    auto grad_b = at::zeros({B, M}, b.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        a.scalar_type(),
        "chebyshev_polynomial_v_multiply_backward",
        [&] {
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            const scalar_t* a_ptr = a_flat.data_ptr<scalar_t>();
            const scalar_t* b_ptr = b_flat.data_ptr<scalar_t>();
            scalar_t* grad_a_ptr = grad_a.data_ptr<scalar_t>();
            scalar_t* grad_b_ptr = grad_b.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t batch = start; batch < end; ++batch) {
                    const int64_t a_batch = (B_a == 1) ? 0 : batch;
                    const int64_t b_batch = (B_b == 1) ? 0 : batch;

                    kernel::polynomial::chebyshev_polynomial_v_multiply_backward(
                        grad_a_ptr + batch * N,
                        grad_b_ptr + batch * M,
                        grad_out_ptr + batch * output_N,
                        a_ptr + a_batch * N,
                        b_ptr + b_batch * M,
                        N,
                        M,
                        output_N
                    );
                }
            });
        }
    );

    // Handle broadcasting reduction
    if (B_a == 1 && B > 1) {
        grad_a = grad_a.sum(0, /*keepdim=*/true);
    }
    if (B_b == 1 && B > 1) {
        grad_b = grad_b.sum(0, /*keepdim=*/true);
    }

    return std::make_tuple(grad_a, grad_b);
}

// Backward backward: returns (grad_grad_output, grad_a, grad_b)
inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
chebyshev_polynomial_v_multiply_backward_backward(
    const at::Tensor& gg_a,
    const at::Tensor& gg_b,
    const at::Tensor& grad_output,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t N = a.size(-1);
    const int64_t M = b.size(-1);
    const int64_t B_a = a.numel() / N;
    const int64_t B_b = b.numel() / M;
    const int64_t B = std::max(B_a, B_b);
    const int64_t output_N = grad_output.size(-1);

    auto gg_a_flat = gg_a.reshape({B, N}).contiguous();
    auto gg_b_flat = gg_b.reshape({B, M}).contiguous();
    auto grad_output_flat = grad_output.reshape({B, output_N}).contiguous();
    auto a_flat = a.reshape({B_a, N}).contiguous();
    auto b_flat = b.reshape({B_b, M}).contiguous();

    auto grad_grad_output = at::zeros({B, output_N}, grad_output.options());
    auto grad_a_from_gg = at::zeros({B, N}, a.options());
    auto grad_b_from_gg = at::zeros({B, M}, b.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        a.scalar_type(),
        "chebyshev_polynomial_v_multiply_backward_backward",
        [&] {
            const scalar_t* gg_a_ptr = gg_a_flat.data_ptr<scalar_t>();
            const scalar_t* gg_b_ptr = gg_b_flat.data_ptr<scalar_t>();
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            const scalar_t* a_ptr = a_flat.data_ptr<scalar_t>();
            const scalar_t* b_ptr = b_flat.data_ptr<scalar_t>();
            scalar_t* ggo_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* grad_a_gg_ptr = grad_a_from_gg.data_ptr<scalar_t>();
            scalar_t* grad_b_gg_ptr = grad_b_from_gg.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t batch = start; batch < end; ++batch) {
                    const int64_t a_batch = (B_a == 1) ? 0 : batch;
                    const int64_t b_batch = (B_b == 1) ? 0 : batch;

                    kernel::polynomial::chebyshev_polynomial_v_multiply_backward_backward(
                        ggo_ptr + batch * output_N,
                        grad_a_gg_ptr + batch * N,
                        grad_b_gg_ptr + batch * M,
                        gg_a_ptr + batch * N,
                        gg_b_ptr + batch * M,
                        grad_out_ptr + batch * output_N,
                        a_ptr + a_batch * N,
                        b_ptr + b_batch * M,
                        N,
                        M,
                        output_N
                    );
                }
            });
        }
    );

    // Handle broadcasting reduction
    if (B_a == 1 && B > 1) {
        grad_a_from_gg = grad_a_from_gg.sum(0, /*keepdim=*/true);
    }
    if (B_b == 1 && B > 1) {
        grad_b_from_gg = grad_b_from_gg.sum(0, /*keepdim=*/true);
    }

    return std::make_tuple(grad_grad_output, grad_a_from_gg, grad_b_from_gg);
}

} // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("chebyshev_polynomial_v_multiply", torchscience::cpu::polynomial::chebyshev_polynomial_v_multiply);
    module.impl("chebyshev_polynomial_v_multiply_backward", torchscience::cpu::polynomial::chebyshev_polynomial_v_multiply_backward);
    module.impl("chebyshev_polynomial_v_multiply_backward_backward", torchscience::cpu::polynomial::chebyshev_polynomial_v_multiply_backward_backward);
}
