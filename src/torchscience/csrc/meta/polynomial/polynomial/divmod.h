#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

// Meta implementation for polynomial_divmod
// Computes output shapes without performing actual computation
// p (B, N), q (B, M) -> (quotient (B, N-M+1), remainder (B, max(M-1, 1)))
inline std::tuple<at::Tensor, at::Tensor> polynomial_divmod(
    const at::Tensor& p,
    const at::Tensor& q
) {
    TORCH_CHECK(p.dim() >= 1, "p must have at least 1 dimension");
    TORCH_CHECK(q.dim() >= 1, "q must have at least 1 dimension");

    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t B = p.numel() / (N > 0 ? N : 1);

    TORCH_CHECK(N >= M, "Dividend degree must be >= divisor degree");

    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;

    return {
        at::empty({B, quot_len}, p.options()),
        at::empty({B, rem_len}, p.options())
    };
}

// Meta implementation for polynomial_divmod_backward
inline std::tuple<at::Tensor, at::Tensor> polynomial_divmod_backward(
    const at::Tensor& grad_Q,
    const at::Tensor& grad_R,
    const at::Tensor& Q,
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

// Meta implementation for polynomial_divmod_backward_backward
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
polynomial_divmod_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_Q,
    const at::Tensor& grad_R,
    const at::Tensor& Q,
    const at::Tensor& p,
    const at::Tensor& q
) {
    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t B = p.numel() / (N > 0 ? N : 1);
    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;

    return {
        at::empty({B, quot_len}, grad_Q.options()),  // grad_grad_Q
        at::empty({B, rem_len}, grad_R.options()),   // grad_grad_R
        at::empty({B, quot_len}, Q.options()),       // grad_Q_out
        at::empty({B, N}, p.options()),              // grad_p_out
        at::empty({B, M}, q.options())               // grad_q_out
    };
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("polynomial_divmod", torchscience::meta::polynomial::polynomial_divmod);
    module.impl("polynomial_divmod_backward", torchscience::meta::polynomial::polynomial_divmod_backward);
    module.impl("polynomial_divmod_backward_backward", torchscience::meta::polynomial::polynomial_divmod_backward_backward);
}
