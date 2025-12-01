#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/complex.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/polynomial/polynomial_evaluate.h"
#include "../../../kernel/polynomial/polynomial/polynomial_evaluate_backward.h"
#include "../../../kernel/polynomial/polynomial/polynomial_evaluate_backward_backward.h"

namespace torchscience::cpu::polynomial {

// Forward: coeffs (B, N), x (M,) -> output (B, M)
// Evaluates p(x) for each batch of coefficients at each point
inline at::Tensor polynomial_evaluate(
    const at::Tensor& coeffs,
    const at::Tensor& x
) {
    TORCH_CHECK(coeffs.dim() >= 1, "coeffs must have at least 1 dimension");
    TORCH_CHECK(x.dim() == 1, "x must be 1-dimensional (flattened batch of points)");

    const int64_t B = coeffs.numel() / coeffs.size(-1);
    const int64_t N = coeffs.size(-1);
    const int64_t M = x.size(0);

    auto coeffs_flat = coeffs.reshape({B, N}).contiguous();
    auto x_contig = x.contiguous();
    auto output = at::empty({B, M}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "polynomial_evaluate",
        [&] {
            const scalar_t* coeffs_ptr = coeffs_flat.data_ptr<scalar_t>();
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, B * M, 1, [&](int64_t start, int64_t end) {
                for (int64_t idx = start; idx < end; ++idx) {
                    int64_t b = idx / M;
                    int64_t m = idx % M;
                    out_ptr[idx] = kernel::polynomial::polynomial_evaluate(
                        coeffs_ptr + b * N,
                        x_ptr[m],
                        N
                    );
                }
            });
        }
    );

    return output;
}

// Backward: grad_output (B, M), coeffs (B, N), x (M,) -> (grad_coeffs (B, N), grad_x (M,))
inline std::tuple<at::Tensor, at::Tensor> polynomial_evaluate_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& x
) {
    TORCH_CHECK(coeffs.dim() >= 1, "coeffs must have at least 1 dimension");
    TORCH_CHECK(x.dim() == 1, "x must be 1-dimensional");

    const int64_t B = coeffs.numel() / coeffs.size(-1);
    const int64_t N = coeffs.size(-1);
    const int64_t M = x.size(0);

    auto coeffs_flat = coeffs.reshape({B, N}).contiguous();
    auto x_contig = x.contiguous();
    auto grad_output_flat = grad_output.reshape({B, M}).contiguous();

    auto grad_coeffs = at::zeros({B, N}, coeffs.options());
    auto grad_x = at::zeros({M}, x.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "polynomial_evaluate_backward",
        [&] {
            const scalar_t* coeffs_ptr = coeffs_flat.data_ptr<scalar_t>();
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            scalar_t* grad_coeffs_ptr = grad_coeffs.data_ptr<scalar_t>();
            scalar_t* grad_x_ptr = grad_x.data_ptr<scalar_t>();

            // Process each (batch, point) combination
            // grad_coeffs accumulates across points, grad_x accumulates across batches
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t m = 0; m < M; ++m) {
                    // Temporary storage for per-point coefficient gradient
                    std::vector<scalar_t> temp_grad_coeffs(N);

                    scalar_t local_grad_x = kernel::polynomial::polynomial_evaluate_backward(
                        temp_grad_coeffs.data(),
                        grad_out_ptr[b * M + m],
                        coeffs_ptr + b * N,
                        x_ptr[m],
                        N
                    );

                    // Accumulate coefficient gradients
                    for (int64_t k = 0; k < N; ++k) {
                        grad_coeffs_ptr[b * N + k] += temp_grad_coeffs[k];
                    }

                    // Accumulate x gradient across batches
                    grad_x_ptr[m] += local_grad_x;
                }
            }
        }
    );

    return {grad_coeffs, grad_x};
}

// Second-order backward
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> polynomial_evaluate_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& gg_x,
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& x
) {
    TORCH_CHECK(coeffs.dim() >= 1, "coeffs must have at least 1 dimension");
    TORCH_CHECK(x.dim() == 1, "x must be 1-dimensional");

    const int64_t B = coeffs.numel() / coeffs.size(-1);
    const int64_t N = coeffs.size(-1);
    const int64_t M = x.size(0);

    auto coeffs_flat = coeffs.reshape({B, N}).contiguous();
    auto x_contig = x.contiguous();
    auto grad_output_flat = grad_output.reshape({B, M}).contiguous();
    auto gg_coeffs_flat = gg_coeffs.defined() ? gg_coeffs.reshape({B, N}).contiguous() : at::zeros({B, N}, coeffs.options());
    auto gg_x_contig = gg_x.defined() ? gg_x.contiguous() : at::zeros({M}, x.options());

    auto grad_grad_output = at::zeros({B, M}, grad_output.options());
    auto grad_coeffs = at::zeros({B, N}, coeffs.options());
    auto grad_x = at::zeros({M}, x.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "polynomial_evaluate_backward_backward",
        [&] {
            const scalar_t* coeffs_ptr = coeffs_flat.data_ptr<scalar_t>();
            const scalar_t* x_ptr = x_contig.data_ptr<scalar_t>();
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            const scalar_t* gg_coeffs_ptr = gg_coeffs_flat.data_ptr<scalar_t>();
            const scalar_t* gg_x_ptr = gg_x_contig.data_ptr<scalar_t>();
            scalar_t* ggo_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* grad_coeffs_ptr = grad_coeffs.data_ptr<scalar_t>();
            scalar_t* grad_x_ptr = grad_x.data_ptr<scalar_t>();

            for (int64_t b = 0; b < B; ++b) {
                for (int64_t m = 0; m < M; ++m) {
                    std::vector<scalar_t> temp_grad_coeffs(N);

                    auto [ggo, g_x] = kernel::polynomial::polynomial_evaluate_backward_backward(
                        temp_grad_coeffs.data(),
                        gg_coeffs_ptr + b * N,
                        gg_x_ptr[m],
                        grad_out_ptr[b * M + m],
                        coeffs_ptr + b * N,
                        x_ptr[m],
                        N
                    );

                    ggo_ptr[b * M + m] = ggo;

                    for (int64_t k = 0; k < N; ++k) {
                        grad_coeffs_ptr[b * N + k] += temp_grad_coeffs[k];
                    }

                    grad_x_ptr[m] += g_x;
                }
            }
        }
    );

    return {grad_grad_output, grad_coeffs, grad_x};
}

} // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("polynomial_evaluate", torchscience::cpu::polynomial::polynomial_evaluate);
    module.impl("polynomial_evaluate_backward", torchscience::cpu::polynomial::polynomial_evaluate_backward);
    module.impl("polynomial_evaluate_backward_backward", torchscience::cpu::polynomial::polynomial_evaluate_backward_backward);
}
