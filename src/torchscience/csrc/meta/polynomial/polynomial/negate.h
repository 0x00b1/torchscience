#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor polynomial_negate(
    const at::Tensor& p
) {
    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    return at::empty({B, N}, p.options());
}

inline at::Tensor polynomial_negate_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p
) {
    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    return at::empty({B, N}, p.options());
}

inline std::tuple<at::Tensor, at::Tensor>
polynomial_negate_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& grad_output,
    const at::Tensor& p
) {
    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    return std::make_tuple(
        at::empty({B, N}, grad_output.options()),
        at::empty({B, N}, p.options())
    );
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("polynomial_negate", torchscience::meta::polynomial::polynomial_negate);
    module.impl("polynomial_negate_backward", torchscience::meta::polynomial::polynomial_negate_backward);
    module.impl("polynomial_negate_backward_backward", torchscience::meta::polynomial::polynomial_negate_backward_backward);
}
