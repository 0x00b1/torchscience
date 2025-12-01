#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor jacobi_polynomial_p_mulx(
    const at::Tensor& coeffs,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    (void)alpha;
    (void)beta;
    const int64_t N = coeffs.size(-1);
    const int64_t output_N = N + 1;

    auto output_sizes = coeffs.sizes().vec();
    output_sizes.back() = output_N;

    return at::empty(output_sizes, coeffs.options());
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> jacobi_polynomial_p_mulx_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    (void)grad_output;
    return std::make_tuple(
        at::empty_like(coeffs),
        at::empty_like(alpha),
        at::empty_like(beta)
    );
}

inline at::Tensor jacobi_polynomial_p_mulx_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& coeffs,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    (void)gg_coeffs;
    (void)alpha;
    (void)beta;
    const int64_t N = coeffs.size(-1);
    const int64_t output_N = N + 1;

    auto output_sizes = coeffs.sizes().vec();
    output_sizes.back() = output_N;

    return at::empty(output_sizes, coeffs.options());
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("jacobi_polynomial_p_mulx", torchscience::meta::polynomial::jacobi_polynomial_p_mulx);
    module.impl("jacobi_polynomial_p_mulx_backward", torchscience::meta::polynomial::jacobi_polynomial_p_mulx_backward);
    module.impl("jacobi_polynomial_p_mulx_backward_backward", torchscience::meta::polynomial::jacobi_polynomial_p_mulx_backward_backward);
}
