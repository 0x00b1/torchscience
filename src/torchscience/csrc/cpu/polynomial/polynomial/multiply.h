#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/complex.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/polynomial/polynomial_multiply.h"
#include "../../../kernel/polynomial/polynomial/polynomial_multiply_backward.h"
#include "../../../kernel/polynomial/polynomial/polynomial_multiply_backward_backward.h"

namespace torchscience::cpu::polynomial {

// Forward: p (B, N), q (B, M) -> output (B, N+M-1)
// Multiplies polynomials via discrete convolution
// Batch dimensions must match (Python handles broadcasting)
inline at::Tensor polynomial_multiply(
    const at::Tensor& p,
    const at::Tensor& q
) {
    TORCH_CHECK(p.dim() >= 1, "p must have at least 1 dimension");
    TORCH_CHECK(q.dim() >= 1, "q must have at least 1 dimension");

    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t K = N + M - 1;

    // Compute batch size (all dims except last)
    const int64_t B_p = p.numel() / N;
    const int64_t B_q = q.numel() / M;
    TORCH_CHECK(B_p == B_q, "Batch dimensions must match. Got ", B_p, " vs ", B_q);
    const int64_t B = B_p;

    // Handle empty polynomial case
    if (N == 0 || M == 0) {
        return at::zeros({B, 0}, p.options());
    }

    auto p_flat = p.reshape({B, N}).contiguous();
    auto q_flat = q.reshape({B, M}).contiguous();
    auto output = at::empty({B, K}, p.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_multiply",
        [&] {
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_flat.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_multiply(
                        out_ptr + b * K,
                        p_ptr + b * N,
                        q_ptr + b * M,
                        N,
                        M
                    );
                }
            });
        }
    );

    return output;
}

// Backward: grad_output (B, K), p (B, N), q (B, M) -> (grad_p (B, N), grad_q (B, M))
inline std::tuple<at::Tensor, at::Tensor> polynomial_multiply_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q
) {
    TORCH_CHECK(p.dim() >= 1, "p must have at least 1 dimension");
    TORCH_CHECK(q.dim() >= 1, "q must have at least 1 dimension");

    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t K = N + M - 1;
    const int64_t B = p.numel() / N;

    if (N == 0 || M == 0) {
        return {at::zeros_like(p), at::zeros_like(q)};
    }

    auto p_flat = p.reshape({B, N}).contiguous();
    auto q_flat = q.reshape({B, M}).contiguous();
    auto grad_output_flat = grad_output.reshape({B, K}).contiguous();

    auto grad_p = at::zeros({B, N}, p.options());
    auto grad_q = at::zeros({B, M}, q.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_multiply_backward",
        [&] {
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_flat.data_ptr<scalar_t>();
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            scalar_t* grad_p_ptr = grad_p.data_ptr<scalar_t>();
            scalar_t* grad_q_ptr = grad_q.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_multiply_backward(
                        grad_p_ptr + b * N,
                        grad_q_ptr + b * M,
                        grad_out_ptr + b * K,
                        p_ptr + b * N,
                        q_ptr + b * M,
                        N,
                        M
                    );
                }
            });
        }
    );

    return {grad_p, grad_q};
}

// Second-order backward
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> polynomial_multiply_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q
) {
    TORCH_CHECK(p.dim() >= 1, "p must have at least 1 dimension");
    TORCH_CHECK(q.dim() >= 1, "q must have at least 1 dimension");

    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t K = N + M - 1;
    const int64_t B = p.numel() / N;

    if (N == 0 || M == 0) {
        return {
            at::zeros({B, K}, grad_output.options()),
            at::zeros_like(p),
            at::zeros_like(q)
        };
    }

    auto p_flat = p.reshape({B, N}).contiguous();
    auto q_flat = q.reshape({B, M}).contiguous();
    auto grad_output_flat = grad_output.reshape({B, K}).contiguous();
    auto gg_p_flat = gg_p.defined() ? gg_p.reshape({B, N}).contiguous() : at::zeros({B, N}, p.options());
    auto gg_q_flat = gg_q.defined() ? gg_q.reshape({B, M}).contiguous() : at::zeros({B, M}, q.options());

    auto grad_grad_output = at::zeros({B, K}, grad_output.options());
    auto grad_p_out = at::zeros({B, N}, p.options());
    auto grad_q_out = at::zeros({B, M}, q.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_multiply_backward_backward",
        [&] {
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_flat.data_ptr<scalar_t>();
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            const scalar_t* gg_p_ptr = gg_p_flat.data_ptr<scalar_t>();
            const scalar_t* gg_q_ptr = gg_q_flat.data_ptr<scalar_t>();
            scalar_t* ggo_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* grad_p_out_ptr = grad_p_out.data_ptr<scalar_t>();
            scalar_t* grad_q_out_ptr = grad_q_out.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_multiply_backward_backward(
                        ggo_ptr + b * K,
                        grad_p_out_ptr + b * N,
                        grad_q_out_ptr + b * M,
                        gg_p_ptr + b * N,
                        gg_q_ptr + b * M,
                        grad_out_ptr + b * K,
                        p_ptr + b * N,
                        q_ptr + b * M,
                        N,
                        M
                    );
                }
            });
        }
    );

    return {grad_grad_output, grad_p_out, grad_q_out};
}

} // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("polynomial_multiply", torchscience::cpu::polynomial::polynomial_multiply);
    module.impl("polynomial_multiply_backward", torchscience::cpu::polynomial::polynomial_multiply_backward);
    module.impl("polynomial_multiply_backward_backward", torchscience::cpu::polynomial::polynomial_multiply_backward_backward);
}
