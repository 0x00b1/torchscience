#pragma once

#include <tuple>
#include <vector>

#include <ATen/core/Tensor.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::statistics::hypothesis_test {

/**
 * Autograd function for Jarque-Bera backward pass.
 *
 * Enables second-order gradients by wrapping the backward call.
 */
class JarqueBeraBackward
    : public torch::autograd::Function<JarqueBeraBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_statistic,
        const at::Tensor& input,
        bool input_requires_grad
    ) {
        ctx->save_for_backward({grad_statistic, input});
        ctx->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_input = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::jarque_bera_backward", "")
            .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
            .call(grad_statistic, input);

        return grad_input;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        // For now, return undefined tensors for second-order gradients.
        // Second-order gradients through hypothesis tests are rarely needed.
        return {
            at::Tensor(),  // grad_grad_statistic
            at::Tensor(),  // grad_input
            at::Tensor()   // grad_input_requires_grad
        };
    }
};

/**
 * Autograd function for Jarque-Bera test.
 *
 * Wraps the forward pass and implements backward pass for autograd.
 */
class JarqueBeraFunction
    : public torch::autograd::Function<JarqueBeraFunction> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& input
    ) {
        bool input_requires_grad = input.requires_grad() &&
            at::isFloatingType(input.scalar_type());
        ctx->saved_data["input_requires_grad"] = input_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [statistic, pvalue] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::jarque_bera", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&)>()
            .call(input);

        ctx->save_for_backward({input});

        return {statistic, pvalue};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor input = saved[0];

        // grad_outputs[0] = gradient w.r.t. statistic
        // grad_outputs[1] = gradient w.r.t. pvalue (we ignore this)
        at::Tensor grad_statistic = grad_outputs[0];

        bool input_requires_grad = ctx->saved_data["input_requires_grad"].toBool();

        if (!input_requires_grad || !grad_statistic.defined()) {
            return {at::Tensor()};
        }

        at::Tensor grad_input = JarqueBeraBackward::apply(
            grad_statistic,
            input,
            input_requires_grad
        );

        return {grad_input};
    }
};

inline std::tuple<at::Tensor, at::Tensor> jarque_bera(const at::Tensor& input) {
    auto results = JarqueBeraFunction::apply(input);
    return std::make_tuple(results[0], results[1]);
}

}  // namespace torchscience::autograd::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("jarque_bera", &torchscience::autograd::statistics::hypothesis_test::jarque_bera);
}
