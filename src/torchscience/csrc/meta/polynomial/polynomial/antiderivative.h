#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor polynomial_antiderivative(
    const at::Tensor& coeffs,
    const at::Tensor& constant
) {
    const int64_t N = coeffs.size(-1);
    const int64_t output_N = N + 1;

    auto output_sizes = coeffs.sizes().vec();
    output_sizes.back() = output_N;

    return at::empty(output_sizes, coeffs.options());
}

inline std::tuple<at::Tensor, at::Tensor> polynomial_antiderivative_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& constant
) {
    const int64_t B = coeffs.numel() / coeffs.size(-1);

    return std::make_tuple(
        at::empty_like(coeffs),
        at::empty({B}, constant.options())
    );
}

inline at::Tensor polynomial_antiderivative_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& gg_constant,
    const at::Tensor& coeffs
) {
    const int64_t N = coeffs.size(-1);
    const int64_t output_N = N + 1;

    auto output_sizes = coeffs.sizes().vec();
    output_sizes.back() = output_N;

    return at::empty(output_sizes, coeffs.options());
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("polynomial_antiderivative", torchscience::meta::polynomial::polynomial_antiderivative);
    module.impl("polynomial_antiderivative_backward", torchscience::meta::polynomial::polynomial_antiderivative_backward);
    module.impl("polynomial_antiderivative_backward_backward", torchscience::meta::polynomial::polynomial_antiderivative_backward_backward);
}
