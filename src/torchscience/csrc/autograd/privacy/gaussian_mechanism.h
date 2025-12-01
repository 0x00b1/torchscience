#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::autograd::privacy {

class GaussianMechanismFunction : public torch::autograd::Function<GaussianMechanismFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& x, const at::Tensor& noise, double sigma) {
        at::AutoDispatchBelowADInplaceOrView guard;
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gaussian_mechanism", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, double)>()
            .call(x, noise, sigma);
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
        auto grad_output = grad_outputs[0];
        at::AutoDispatchBelowADInplaceOrView guard;
        auto grad_x = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::gaussian_mechanism_backward", "")
            .typed<at::Tensor(const at::Tensor&)>()
            .call(grad_output);
        return {grad_x, at::Tensor(), at::Tensor()};
    }
};

at::Tensor gaussian_mechanism_autograd(const at::Tensor& x, const at::Tensor& noise, double sigma) {
    return GaussianMechanismFunction::apply(x, noise, sigma);
}

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("gaussian_mechanism", &gaussian_mechanism_autograd);
}

}  // namespace torchscience::autograd::privacy
