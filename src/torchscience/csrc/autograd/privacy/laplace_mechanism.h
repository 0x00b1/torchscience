#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::autograd::privacy {

class LaplaceMechanismFunction : public torch::autograd::Function<LaplaceMechanismFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& x, const at::Tensor& noise, double b) {
        at::AutoDispatchBelowADInplaceOrView guard;
        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::laplace_mechanism", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&, double)>()
            .call(x, noise, b);
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
        auto grad_output = grad_outputs[0];
        at::AutoDispatchBelowADInplaceOrView guard;
        auto grad_x = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::laplace_mechanism_backward", "")
            .typed<at::Tensor(const at::Tensor&)>()
            .call(grad_output);
        return {grad_x, at::Tensor(), at::Tensor()};
    }
};

at::Tensor laplace_mechanism_autograd(const at::Tensor& x, const at::Tensor& noise, double b) {
    return LaplaceMechanismFunction::apply(x, noise, b);
}

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("laplace_mechanism", &laplace_mechanism_autograd);
}

}  // namespace torchscience::autograd::privacy
