#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_mulx.h"
#include "../../../kernel/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_mulx_backward.h"
#include "../../../kernel/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_mulx_backward_backward.h"

namespace torchscience::cpu::polynomial {

// Forward: coeffs (B, N) -> output (B, N + 1)
inline at::Tensor chebyshev_polynomial_t_mulx(
    const at::Tensor& coeffs
) {
    TORCH_CHECK(coeffs.dim() >= 1, "coeffs must have at least 1 dimension");

    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = (N > 0) ? (N + 1) : 1;

    auto coeffs_flat = coeffs.reshape({B, N}).contiguous();
    auto output = at::empty({B, output_N}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "chebyshev_polynomial_t_mulx",
        [&] {
            const scalar_t* coeffs_ptr = coeffs_flat.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t batch = start; batch < end; ++batch) {
                    kernel::polynomial::chebyshev_polynomial_t_mulx(
                        out_ptr + batch * output_N,
                        coeffs_ptr + batch * N,
                        N
                    );
                }
            });
        }
    );

    return output;
}

// Backward: returns grad_coeffs
inline at::Tensor chebyshev_polynomial_t_mulx_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = grad_output.size(-1);

    auto grad_output_flat = grad_output.reshape({B, output_N}).contiguous();
    auto coeffs_flat = coeffs.reshape({B, N}).contiguous();

    auto grad_coeffs = at::zeros({B, N}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "chebyshev_polynomial_t_mulx_backward",
        [&] {
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            const scalar_t* coeffs_ptr = coeffs_flat.data_ptr<scalar_t>();
            scalar_t* grad_coeffs_ptr = grad_coeffs.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t batch = start; batch < end; ++batch) {
                    kernel::polynomial::chebyshev_polynomial_t_mulx_backward(
                        grad_coeffs_ptr + batch * N,
                        grad_out_ptr + batch * output_N,
                        coeffs_ptr + batch * N,
                        N,
                        output_N
                    );
                }
            });
        }
    );

    return grad_coeffs;
}

// Backward backward: returns (grad_grad_output, grad_coeffs)
inline std::tuple<at::Tensor, at::Tensor>
chebyshev_polynomial_t_mulx_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& grad_output,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = grad_output.size(-1);

    auto gg_coeffs_flat = gg_coeffs.reshape({B, N}).contiguous();
    auto grad_output_flat = grad_output.reshape({B, output_N}).contiguous();
    auto coeffs_flat = coeffs.reshape({B, N}).contiguous();

    auto grad_grad_output = at::zeros({B, output_N}, grad_output.options());
    auto grad_coeffs_from_gg = at::zeros({B, N}, coeffs.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        coeffs.scalar_type(),
        "chebyshev_polynomial_t_mulx_backward_backward",
        [&] {
            const scalar_t* gg_coeffs_ptr = gg_coeffs_flat.data_ptr<scalar_t>();
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            const scalar_t* coeffs_ptr = coeffs_flat.data_ptr<scalar_t>();
            scalar_t* ggo_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* grad_coeffs_gg_ptr = grad_coeffs_from_gg.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t batch = start; batch < end; ++batch) {
                    kernel::polynomial::chebyshev_polynomial_t_mulx_backward_backward(
                        ggo_ptr + batch * output_N,
                        grad_coeffs_gg_ptr + batch * N,
                        gg_coeffs_ptr + batch * N,
                        grad_out_ptr + batch * output_N,
                        coeffs_ptr + batch * N,
                        N,
                        output_N
                    );
                }
            });
        }
    );

    return std::make_tuple(grad_grad_output, grad_coeffs_from_gg);
}

} // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("chebyshev_polynomial_t_mulx", torchscience::cpu::polynomial::chebyshev_polynomial_t_mulx);
    module.impl("chebyshev_polynomial_t_mulx_backward", torchscience::cpu::polynomial::chebyshev_polynomial_t_mulx_backward);
    module.impl("chebyshev_polynomial_t_mulx_backward_backward", torchscience::cpu::polynomial::chebyshev_polynomial_t_mulx_backward_backward);
}
