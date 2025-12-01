#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor gegenbauer_polynomial_c_mulx(
    const at::Tensor& coeffs,
    const at::Tensor& alpha
) {
    (void)alpha;
    const int64_t N = coeffs.size(-1);
    const int64_t output_N = N + 1;

    auto output_sizes = coeffs.sizes().vec();
    output_sizes.back() = output_N;

    return at::empty(output_sizes, coeffs.options());
}

inline std::tuple<at::Tensor, at::Tensor> gegenbauer_polynomial_c_mulx_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& alpha
) {
    (void)grad_output;
    return std::make_tuple(at::empty_like(coeffs), at::empty_like(alpha));
}

inline at::Tensor gegenbauer_polynomial_c_mulx_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& coeffs,
    const at::Tensor& alpha
) {
    (void)gg_coeffs;
    (void)alpha;
    const int64_t N = coeffs.size(-1);
    const int64_t output_N = N + 1;

    auto output_sizes = coeffs.sizes().vec();
    output_sizes.back() = output_N;

    return at::empty(output_sizes, coeffs.options());
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("gegenbauer_polynomial_c_mulx", torchscience::meta::polynomial::gegenbauer_polynomial_c_mulx);
    module.impl("gegenbauer_polynomial_c_mulx_backward", torchscience::meta::polynomial::gegenbauer_polynomial_c_mulx_backward);
    module.impl("gegenbauer_polynomial_c_mulx_backward_backward", torchscience::meta::polynomial::gegenbauer_polynomial_c_mulx_backward_backward);
}
