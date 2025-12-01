#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::polynomial {

inline at::Tensor polynomial_scale(
    const at::Tensor& p,
    const at::Tensor& c
) {
    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    return at::empty({B, N}, p.options());
}

inline std::tuple<at::Tensor, at::Tensor> polynomial_scale_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& c
) {
    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    return std::make_tuple(
        at::empty({B, N}, p.options()),  // grad_p
        at::empty({B}, c.options())       // grad_c
    );
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
polynomial_scale_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_c,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& c
) {
    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    return std::make_tuple(
        at::empty({B, N}, grad_output.options()),  // grad_grad_output
        at::empty({B, N}, p.options()),             // grad_p
        at::empty({B}, c.options())                 // grad_c
    );
}

} // namespace torchscience::meta::polynomial

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("polynomial_scale", torchscience::meta::polynomial::polynomial_scale);
    module.impl("polynomial_scale_backward", torchscience::meta::polynomial::polynomial_scale_backward);
    module.impl("polynomial_scale_backward_backward", torchscience::meta::polynomial::polynomial_scale_backward_backward);
}
