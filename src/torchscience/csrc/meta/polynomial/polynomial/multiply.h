#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

// Meta implementation for polynomial_multiply
// Computes output shape without performing actual computation
// p (B, N), q (B, M) -> output (B, N+M-1)
inline at::Tensor polynomial_multiply(
    const at::Tensor& p,
    const at::Tensor& q
) {
    TORCH_CHECK(p.dim() >= 1, "p must have at least 1 dimension");
    TORCH_CHECK(q.dim() >= 1, "q must have at least 1 dimension");

    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t K = (N == 0 || M == 0) ? 0 : N + M - 1;
    const int64_t B = p.numel() / (N > 0 ? N : 1);

    return at::empty({B, K}, p.options());
}

// Meta implementation for polynomial_multiply_backward
// grad_output (B, K), p (B, N), q (B, M) -> (grad_p (B, N), grad_q (B, M))
inline std::tuple<at::Tensor, at::Tensor> polynomial_multiply_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q
) {
    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t B = p.numel() / (N > 0 ? N : 1);

    return {
        at::empty({B, N}, p.options()),
        at::empty({B, M}, q.options())
    };
}

// Meta implementation for polynomial_multiply_backward_backward
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> polynomial_multiply_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q
) {
    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t K = (N == 0 || M == 0) ? 0 : N + M - 1;
    const int64_t B = p.numel() / (N > 0 ? N : 1);

    return {
        at::empty({B, K}, grad_output.options()),
        at::empty({B, N}, p.options()),
        at::empty({B, M}, q.options())
    };
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("polynomial_multiply", torchscience::meta::polynomial::polynomial_multiply);
    module.impl("polynomial_multiply_backward", torchscience::meta::polynomial::polynomial_multiply_backward);
    module.impl("polynomial_multiply_backward_backward", torchscience::meta::polynomial::polynomial_multiply_backward_backward);
}
