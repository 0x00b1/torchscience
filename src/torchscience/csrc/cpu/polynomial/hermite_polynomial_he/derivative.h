#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/hermite_polynomial_he/hermite_polynomial_he_derivative.h"
#include "../../../kernel/polynomial/hermite_polynomial_he/hermite_polynomial_he_derivative_backward.h"
#include "../../../kernel/polynomial/hermite_polynomial_he/hermite_polynomial_he_derivative_backward_backward.h"

namespace torchscience::cpu::polynomial {

inline at::Tensor hermite_polynomial_he_derivative(const at::Tensor& coeffs) {
    TORCH_CHECK(coeffs.dim() >= 1, "coeffs must have at least 1 dimension");

    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = (N > 1) ? (N - 1) : 1;

    auto coeffs_flat = coeffs.reshape({B, N}).contiguous();
    auto output = at::empty({B, output_N}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "hermite_polynomial_he_derivative",
        [&] {
            const scalar_t* coeffs_ptr = coeffs_flat.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::hermite_polynomial_he_derivative(
                        out_ptr + b * output_N,
                        coeffs_ptr + b * N,
                        N
                    );
                }
            });
        }
    );

    return output;
}

inline at::Tensor hermite_polynomial_he_derivative_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = grad_output.size(-1);

    auto grad_output_flat = grad_output.reshape({B, output_N}).contiguous();
    auto grad_coeffs = at::empty({B, N}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "hermite_polynomial_he_derivative_backward",
        [&] {
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            scalar_t* grad_coeffs_ptr = grad_coeffs.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::hermite_polynomial_he_derivative_backward(
                        grad_coeffs_ptr + b * N,
                        grad_out_ptr + b * output_N,
                        N,
                        output_N
                    );
                }
            });
        }
    );

    return grad_coeffs;
}

inline at::Tensor hermite_polynomial_he_derivative_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = (N > 1) ? (N - 1) : 1;

    auto gg_coeffs_flat = gg_coeffs.reshape({B, N}).contiguous();
    auto grad_grad_output = at::empty({B, output_N}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "hermite_polynomial_he_derivative_backward_backward",
        [&] {
            const scalar_t* gg_coeffs_ptr = gg_coeffs_flat.data_ptr<scalar_t>();
            scalar_t* ggo_ptr = grad_grad_output.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::hermite_polynomial_he_derivative_backward_backward(
                        ggo_ptr + b * output_N,
                        gg_coeffs_ptr + b * N,
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
    module.impl("hermite_polynomial_he_derivative", torchscience::cpu::polynomial::hermite_polynomial_he_derivative);
    module.impl("hermite_polynomial_he_derivative_backward", torchscience::cpu::polynomial::hermite_polynomial_he_derivative_backward);
    module.impl("hermite_polynomial_he_derivative_backward_backward", torchscience::cpu::polynomial::hermite_polynomial_he_derivative_backward_backward);
}
