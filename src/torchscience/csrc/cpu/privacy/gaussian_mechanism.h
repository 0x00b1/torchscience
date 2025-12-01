#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/privacy/gaussian_mechanism.h"

namespace torchscience::cpu::privacy {

at::Tensor gaussian_mechanism(const at::Tensor& x, const at::Tensor& noise, double sigma) {
    TORCH_CHECK(x.sizes() == noise.sizes(), "gaussian_mechanism: x and noise must have same shape");
    auto output = at::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "gaussian_mechanism", [&] {
        auto x_data = x.data_ptr<scalar_t>();
        auto noise_data = noise.data_ptr<scalar_t>();
        auto output_data = output.data_ptr<scalar_t>();
        scalar_t sigma_t = static_cast<scalar_t>(sigma);
        for (int64_t i = 0; i < x.numel(); i++) {
            output_data[i] = kernel::privacy::gaussian_mechanism_forward(x_data[i], noise_data[i], sigma_t);
        }
    });
    return output;
}

at::Tensor gaussian_mechanism_backward(const at::Tensor& grad_output) {
    return grad_output.clone();
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("gaussian_mechanism", &gaussian_mechanism);
    m.impl("gaussian_mechanism_backward", &gaussian_mechanism_backward);
}

}  // namespace torchscience::cpu::privacy
