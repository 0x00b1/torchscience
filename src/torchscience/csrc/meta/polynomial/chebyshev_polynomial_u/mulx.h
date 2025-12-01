#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor chebyshev_polynomial_u_mulx(
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = (N > 0) ? (N + 1) : 1;

    return at::empty({B, output_N}, coeffs.options());
}

inline at::Tensor chebyshev_polynomial_u_mulx_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;

    return at::empty({B, N}, coeffs.options());
}

inline std::tuple<at::Tensor, at::Tensor>
chebyshev_polynomial_u_mulx_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& grad_output,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t B = coeffs.numel() / N;
    const int64_t output_N = grad_output.size(-1);

    return std::make_tuple(
        at::empty({B, output_N}, grad_output.options()),
        at::empty({B, N}, coeffs.options())
    );
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("chebyshev_polynomial_u_mulx", torchscience::meta::polynomial::chebyshev_polynomial_u_mulx);
    module.impl("chebyshev_polynomial_u_mulx_backward", torchscience::meta::polynomial::chebyshev_polynomial_u_mulx_backward);
    module.impl("chebyshev_polynomial_u_mulx_backward_backward", torchscience::meta::polynomial::chebyshev_polynomial_u_mulx_backward_backward);
}
