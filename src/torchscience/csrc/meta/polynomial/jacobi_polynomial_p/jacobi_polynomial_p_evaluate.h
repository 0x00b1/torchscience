#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor jacobi_polynomial_p_evaluate(
    const at::Tensor& coeffs,
    const at::Tensor& x,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    TORCH_CHECK(coeffs.dim() >= 1, "coeffs must have at least 1 dimension");
    TORCH_CHECK(x.dim() == 1, "x must be 1-dimensional");

    const int64_t B = coeffs.numel() / coeffs.size(-1);
    const int64_t M = x.size(0);

    return at::empty({B, M}, coeffs.options());
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> jacobi_polynomial_p_evaluate_backward(
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& x,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    const int64_t B = coeffs.numel() / coeffs.size(-1);
    const int64_t N = coeffs.size(-1);
    const int64_t M = x.size(0);

    return {
        at::empty({B, N}, coeffs.options()),
        at::empty({M}, x.options()),
        at::empty({}, alpha.options()),
        at::empty({}, beta.options())
    };
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> jacobi_polynomial_p_evaluate_backward_backward(
    const at::Tensor& gg_coeffs,
    const at::Tensor& gg_x,
    const at::Tensor& gg_alpha,
    const at::Tensor& gg_beta,
    const at::Tensor& grad_output,
    const at::Tensor& coeffs,
    const at::Tensor& x,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    const int64_t B = coeffs.numel() / coeffs.size(-1);
    const int64_t N = coeffs.size(-1);
    const int64_t M = x.size(0);

    return {
        at::empty({B, M}, grad_output.options()),
        at::empty({B, N}, coeffs.options()),
        at::empty({M}, x.options()),
        at::empty({}, alpha.options()),
        at::empty({}, beta.options())
    };
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("jacobi_polynomial_p_evaluate", torchscience::meta::polynomial::jacobi_polynomial_p_evaluate);
    module.impl("jacobi_polynomial_p_evaluate_backward", torchscience::meta::polynomial::jacobi_polynomial_p_evaluate_backward);
    module.impl("jacobi_polynomial_p_evaluate_backward_backward", torchscience::meta::polynomial::jacobi_polynomial_p_evaluate_backward_backward);
}
