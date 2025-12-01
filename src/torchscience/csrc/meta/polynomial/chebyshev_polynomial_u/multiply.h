#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor chebyshev_polynomial_u_multiply(
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t N = a.size(-1);
    const int64_t M = b.size(-1);
    const int64_t B_a = a.numel() / N;
    const int64_t B_b = b.numel() / M;
    const int64_t B = std::max(B_a, B_b);
    const int64_t output_N = (N > 0 && M > 0) ? (N + M - 1) : 1;

    return at::empty({B, output_N}, a.options());
}

inline std::tuple<at::Tensor, at::Tensor> chebyshev_polynomial_u_multiply_backward(
    const at::Tensor& grad_output,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t N = a.size(-1);
    const int64_t M = b.size(-1);
    const int64_t B_a = a.numel() / N;
    const int64_t B_b = b.numel() / M;
    const int64_t B = std::max(B_a, B_b);

    // Note: for broadcasting, returned gradients may need reduction
    // This meta function returns the unreduced shape
    return std::make_tuple(
        at::empty({B, N}, a.options()),
        at::empty({B, M}, b.options())
    );
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
chebyshev_polynomial_u_multiply_backward_backward(
    const at::Tensor& gg_a,
    const at::Tensor& gg_b,
    const at::Tensor& grad_output,
    const at::Tensor& a,
    const at::Tensor& b
) {
    const int64_t N = a.size(-1);
    const int64_t M = b.size(-1);
    const int64_t B_a = a.numel() / N;
    const int64_t B_b = b.numel() / M;
    const int64_t B = std::max(B_a, B_b);
    const int64_t output_N = grad_output.size(-1);

    return std::make_tuple(
        at::empty({B, output_N}, grad_output.options()),
        at::empty({B, N}, a.options()),
        at::empty({B, M}, b.options())
    );
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("chebyshev_polynomial_u_multiply", torchscience::meta::polynomial::chebyshev_polynomial_u_multiply);
    module.impl("chebyshev_polynomial_u_multiply_backward", torchscience::meta::polynomial::chebyshev_polynomial_u_multiply_backward);
    module.impl("chebyshev_polynomial_u_multiply_backward_backward", torchscience::meta::polynomial::chebyshev_polynomial_u_multiply_backward_backward);
}
