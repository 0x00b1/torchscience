#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/polynomial/polynomial_antiderivative.h"
#include "../../../kernel/polynomial/polynomial/polynomial_antiderivative_backward.h"
#include "../../../kernel/polynomial/polynomial/polynomial_antiderivative_backward_backward.h"

namespace torchscience::cpu::polynomial {

// Forward: coeffs (B, N), constant (B,) -> output (B, N+1)
inline at::Tensor polynomial_antiderivative(
    const at::Tensor& coeffs,
    const at::Tensor& constant
) {
    TORCH_CHECK(coeffs.dim() >= 1, "coeffs must have at least 1 dimension");
    TORCH_CHECK(constant.numel() > 0, "constant must not be empty");

    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = N + 1;

    // Broadcast constant to match batch size
    auto constant_expanded = constant.expand({B}).contiguous();
    auto coeffs_flat = coeffs.reshape({B, N}).contiguous();
    auto output = at::empty({B, output_N}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "polynomial_antiderivative",
        [&] {
            const scalar_t* coeffs_ptr = coeffs_flat.data_ptr<scalar_t>();
            const scalar_t* constant_ptr = constant_expanded.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_antiderivative(
                        out_ptr + b * output_N,
                        coeffs_ptr + b * N,
                        constant_ptr[b],
                        N
                    );
                }
            });
        }
    );

    return output;
}

// Backward: returns (grad_coeffs, grad_constant)
inline std::tuple<at::Tensor, at::Tensor> polynomial_antiderivative_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& constant
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = grad_output.size(-1);

    auto grad_output_flat = grad_output.reshape({B, output_N}).contiguous();
    auto grad_coeffs = at::empty({B, N}, coeffs.options());
    auto grad_constant = at::empty({B}, constant.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "polynomial_antiderivative_backward",
        [&] {
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            scalar_t* grad_coeffs_ptr = grad_coeffs.data_ptr<scalar_t>();
            scalar_t* grad_constant_ptr = grad_constant.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    // grad_constant = grad_output[0]
                    grad_constant_ptr[b] = grad_out_ptr[b * output_N];

                    // grad_coeffs[k] = grad_output[k+1] / (k+1)
                    kernel::polynomial::polynomial_antiderivative_backward(
                        grad_coeffs_ptr + b * N,
                        grad_out_ptr + b * output_N,
                        N
                    );
                }
            });
        }
    );

    return std::make_tuple(grad_coeffs, grad_constant);
}

// Backward backward: returns grad_grad_output
inline at::Tensor polynomial_antiderivative_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& gg_constant,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = N + 1;

    auto gg_coeffs_flat = gg_coeffs.reshape({B, N}).contiguous();
    auto gg_constant_expanded = gg_constant.expand({B}).contiguous();
    auto grad_grad_output = at::empty({B, output_N}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "polynomial_antiderivative_backward_backward",
        [&] {
            const scalar_t* gg_coeffs_ptr = gg_coeffs_flat.data_ptr<scalar_t>();
            const scalar_t* gg_constant_ptr = gg_constant_expanded.data_ptr<scalar_t>();
            scalar_t* ggo_ptr = grad_grad_output.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    // Compute grad_grad_output from gg_coeffs
                    kernel::polynomial::polynomial_antiderivative_backward_backward(
                        ggo_ptr + b * output_N,
                        gg_coeffs_ptr + b * N,
                        N,
                        output_N
                    );

                    // Add contribution from gg_constant
                    ggo_ptr[b * output_N] = gg_constant_ptr[b];
                }
            });
        }
    );

    return grad_grad_output;
}

} // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("polynomial_antiderivative", torchscience::cpu::polynomial::polynomial_antiderivative);
    module.impl("polynomial_antiderivative_backward", torchscience::cpu::polynomial::polynomial_antiderivative_backward);
    module.impl("polynomial_antiderivative_backward_backward", torchscience::cpu::polynomial::polynomial_antiderivative_backward_backward);
}
