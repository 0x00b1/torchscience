#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/privacy/laplace_mechanism.h"

namespace torchscience::cpu::privacy {

at::Tensor laplace_mechanism(const at::Tensor& x, const at::Tensor& noise, double b) {
    TORCH_CHECK(x.sizes() == noise.sizes(), "laplace_mechanism: x and noise must have same shape");
    auto output = at::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "laplace_mechanism", [&] {
        auto x_data = x.data_ptr<scalar_t>();
        auto noise_data = noise.data_ptr<scalar_t>();
        auto output_data = output.data_ptr<scalar_t>();
        scalar_t b_t = static_cast<scalar_t>(b);
        for (int64_t i = 0; i < x.numel(); i++) {
            output_data[i] = kernel::privacy::laplace_mechanism_forward(x_data[i], noise_data[i], b_t);
        }
    });
    return output;
}

at::Tensor laplace_mechanism_backward(const at::Tensor& grad_output) {
    return grad_output.clone();
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("laplace_mechanism", &laplace_mechanism);
    m.impl("laplace_mechanism_backward", &laplace_mechanism_backward);
}

}  // namespace torchscience::cpu::privacy
