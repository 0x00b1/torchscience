#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::privacy {

at::Tensor laplace_mechanism(const at::Tensor& x, const at::Tensor& noise, double b) {
    return at::empty_like(x);
}

at::Tensor laplace_mechanism_backward(const at::Tensor& grad_output) {
    return at::empty_like(grad_output);
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("laplace_mechanism", &laplace_mechanism);
    m.impl("laplace_mechanism_backward", &laplace_mechanism_backward);
}

}  // namespace torchscience::meta::privacy
