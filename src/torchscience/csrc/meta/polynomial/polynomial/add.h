#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor polynomial_add(
    const at::Tensor& p,
    const at::Tensor& q
) {
    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t B_p = p.numel() / N;
    const int64_t B_q = q.numel() / M;
    const int64_t B = std::max(B_p, B_q);
    const int64_t K = std::max(N, M);

    return at::empty({B, K}, p.options());
}

inline std::tuple<at::Tensor, at::Tensor> polynomial_add_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q
) {
    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t B_p = p.numel() / N;
    const int64_t B_q = q.numel() / M;
    const int64_t B = std::max(B_p, B_q);

    // Note: for broadcasting, returned gradients may need reduction
    // This meta function returns the unreduced shape
    return std::make_tuple(
        at::empty({B, N}, p.options()),
        at::empty({B, M}, q.options())
    );
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
polynomial_add_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q
) {
    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t B_p = p.numel() / N;
    const int64_t B_q = q.numel() / M;
    const int64_t B = std::max(B_p, B_q);
    const int64_t K = grad_output.size(-1);

    return std::make_tuple(
        at::empty({B, K}, grad_output.options()),
        at::empty({B, N}, p.options()),
        at::empty({B, M}, q.options())
    );
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("polynomial_add", torchscience::meta::polynomial::polynomial_add);
    module.impl("polynomial_add_backward", torchscience::meta::polynomial::polynomial_add_backward);
    module.impl("polynomial_add_backward_backward", torchscience::meta::polynomial::polynomial_add_backward_backward);
}
