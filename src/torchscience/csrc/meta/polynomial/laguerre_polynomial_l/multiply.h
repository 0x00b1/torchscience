#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor laguerre_polynomial_l_multiply(
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t N = a.size(-1);
    const int64_t M = b.size(-1);
    const int64_t output_N = (N == 0 || M == 0) ? 1 : (N + M - 1);

    auto output_sizes = a.sizes().vec();
    output_sizes.back() = output_N;

    return at::empty(output_sizes, a.options());
}

inline std::tuple<at::Tensor, at::Tensor> laguerre_polynomial_l_multiply_backward(
    const at::Tensor& grad_output,
    const at::Tensor& a,
    const at::Tensor& b
) {
    return std::make_tuple(at::empty_like(a), at::empty_like(b));
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> laguerre_polynomial_l_multiply_backward_backward(
    const at::Tensor& gg_a,
    const at::Tensor& gg_b,
    const at::Tensor& grad_output,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t output_N = grad_output.size(-1);

    auto output_sizes = grad_output.sizes().vec();
    output_sizes.back() = output_N;

    return std::make_tuple(
        at::empty(output_sizes, grad_output.options()),
        at::empty_like(a),
        at::empty_like(b)
    );
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("laguerre_polynomial_l_multiply", torchscience::meta::polynomial::laguerre_polynomial_l_multiply);
    module.impl("laguerre_polynomial_l_multiply_backward", torchscience::meta::polynomial::laguerre_polynomial_l_multiply_backward);
    module.impl("laguerre_polynomial_l_multiply_backward_backward", torchscience::meta::polynomial::laguerre_polynomial_l_multiply_backward_backward);
}
