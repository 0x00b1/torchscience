#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_mulx.h"
#include "../../../kernel/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_mulx_backward.h"
#include "../../../kernel/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_mulx_backward_backward.h"

namespace torchscience::cpu::polynomial {

inline at::Tensor jacobi_polynomial_p_mulx(
    const at::Tensor& coeffs,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    TORCH_CHECK(coeffs.dim() >= 1, "coeffs must have at least 1 dimension");

    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = N + 1;

    auto coeffs_flat = coeffs.reshape({B, N}).contiguous();
    auto output = at::empty({B, output_N}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "jacobi_polynomial_p_mulx",
        [&] {
            const scalar_t* coeffs_ptr = coeffs_flat.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();
            scalar_t alpha_val = alpha.item<scalar_t>();
            scalar_t beta_val = beta.item<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::jacobi_polynomial_p_mulx(
                        out_ptr + b * output_N,
                        coeffs_ptr + b * N,
                        alpha_val,
                        beta_val,
                        N
                    );
                }
            });
        }
    );

    return output;
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> jacobi_polynomial_p_mulx_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = grad_output.size(-1);

    auto grad_output_flat = grad_output.reshape({B, output_N}).contiguous();
    auto coeffs_flat = coeffs.reshape({B, N}).contiguous();
    auto grad_coeffs = at::empty({B, N}, coeffs.options());
    auto grad_alpha_accum = at::zeros({B}, coeffs.options());
    auto grad_beta_accum = at::zeros({B}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "jacobi_polynomial_p_mulx_backward",
        [&] {
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            const scalar_t* coeffs_ptr = coeffs_flat.data_ptr<scalar_t>();
            scalar_t* grad_coeffs_ptr = grad_coeffs.data_ptr<scalar_t>();
            scalar_t* grad_alpha_ptr = grad_alpha_accum.data_ptr<scalar_t>();
            scalar_t* grad_beta_ptr = grad_beta_accum.data_ptr<scalar_t>();
            scalar_t alpha_val = alpha.item<scalar_t>();
            scalar_t beta_val = beta.item<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::jacobi_polynomial_p_mulx_backward(
                        grad_coeffs_ptr + b * N,
                        grad_alpha_ptr + b,
                        grad_beta_ptr + b,
                        grad_out_ptr + b * output_N,
                        coeffs_ptr + b * N,
                        alpha_val,
                        beta_val,
                        N,
                        output_N
                    );
                }
            });
        }
    );

    // Sum gradients over batch dimension
    at::Tensor grad_alpha = grad_alpha_accum.sum();
    at::Tensor grad_beta = grad_beta_accum.sum();

    return std::make_tuple(grad_coeffs, grad_alpha, grad_beta);
}

inline at::Tensor jacobi_polynomial_p_mulx_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& coeffs,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = N + 1;

    auto gg_coeffs_flat = gg_coeffs.reshape({B, N}).contiguous();
    auto grad_grad_output = at::empty({B, output_N}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "jacobi_polynomial_p_mulx_backward_backward",
        [&] {
            const scalar_t* gg_coeffs_ptr = gg_coeffs_flat.data_ptr<scalar_t>();
            scalar_t* ggo_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t alpha_val = alpha.item<scalar_t>();
            scalar_t beta_val = beta.item<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::jacobi_polynomial_p_mulx_backward_backward(
                        ggo_ptr + b * output_N,
                        gg_coeffs_ptr + b * N,
                        alpha_val,
                        beta_val,
                        N,
                        output_N
                    );
                }
            });
        }
    );

    return grad_grad_output;
}

} // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("jacobi_polynomial_p_mulx", torchscience::cpu::polynomial::jacobi_polynomial_p_mulx);
    module.impl("jacobi_polynomial_p_mulx_backward", torchscience::cpu::polynomial::jacobi_polynomial_p_mulx_backward);
    module.impl("jacobi_polynomial_p_mulx_backward_backward", torchscience::cpu::polynomial::jacobi_polynomial_p_mulx_backward_backward);
}
