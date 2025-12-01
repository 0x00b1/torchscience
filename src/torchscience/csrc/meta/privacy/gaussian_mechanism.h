#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::privacy {

at::Tensor gaussian_mechanism(const at::Tensor& x, const at::Tensor& noise, double sigma) {
    return at::empty_like(x);
}

at::Tensor gaussian_mechanism_backward(const at::Tensor& grad_output) {
    return at::empty_like(grad_output);
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("gaussian_mechanism", &gaussian_mechanism);
    m.impl("gaussian_mechanism_backward", &gaussian_mechanism_backward);
}

}  // namespace torchscience::meta::privacy
