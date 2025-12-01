#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor legendre_polynomial_p_derivative(const at::Tensor& coeffs) {
    const int64_t N = coeffs.size(-1);
    const int64_t output_N = (N > 1) ? (N - 1) : 1;

    auto output_sizes = coeffs.sizes().vec();
    output_sizes.back() = output_N;

    return at::empty(output_sizes, coeffs.options());
}

inline at::Tensor legendre_polynomial_p_derivative_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs
) {
    return at::empty_like(coeffs);
}

inline at::Tensor legendre_polynomial_p_derivative_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t output_N = (N > 1) ? (N - 1) : 1;

    auto output_sizes = coeffs.sizes().vec();
    output_sizes.back() = output_N;

    return at::empty(output_sizes, coeffs.options());
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("legendre_polynomial_p_derivative", torchscience::meta::polynomial::legendre_polynomial_p_derivative);
    module.impl("legendre_polynomial_p_derivative_backward", torchscience::meta::polynomial::legendre_polynomial_p_derivative_backward);
    module.impl("legendre_polynomial_p_derivative_backward_backward", torchscience::meta::polynomial::legendre_polynomial_p_derivative_backward_backward);
}
