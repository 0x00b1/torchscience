#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

// Meta implementation for chebyshev_polynomial_w_antiderivative
inline at::Tensor chebyshev_polynomial_w_antiderivative(
    const at::Tensor& coeffs
) {
    TORCH_CHECK(coeffs.dim() >= 1, "coeffs must have at least 1 dimension");

    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = N + 1;

    return at::empty({B, output_N}, coeffs.options());
}

// Meta implementation for chebyshev_polynomial_w_antiderivative_backward
inline at::Tensor chebyshev_polynomial_w_antiderivative_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;

    return at::empty({B, N}, coeffs.options());
}

// Meta implementation for chebyshev_polynomial_w_antiderivative_backward_backward
inline at::Tensor chebyshev_polynomial_w_antiderivative_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& grad_output,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = N + 1;

    return at::empty({B, output_N}, grad_output.options());
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("chebyshev_polynomial_w_antiderivative", torchscience::meta::polynomial::chebyshev_polynomial_w_antiderivative);
    module.impl("chebyshev_polynomial_w_antiderivative_backward", torchscience::meta::polynomial::chebyshev_polynomial_w_antiderivative_backward);
    module.impl("chebyshev_polynomial_w_antiderivative_backward_backward", torchscience::meta::polynomial::chebyshev_polynomial_w_antiderivative_backward_backward);
}
