#pragma once

#include <tuple>

#include <torch/extension.h>

namespace torchscience::autograd::filter {

/**
 * Backward function class for double-backward support.
 */
class ButterworthAnalogBandpassFilterBackward
    : public torch::autograd::Function<ButterworthAnalogBandpassFilterBackward> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* context,
        const at::Tensor& gradient_output,
        int64_t n,
        const at::Tensor& omega_p1,
        const at::Tensor& omega_p2,
        const bool omega_p1_requires_grad,
        const bool omega_p2_requires_grad
    ) {
        context->save_for_backward({gradient_output, omega_p1, omega_p2});
        context->saved_data["n"] = n;
        context->saved_data["omega_p1_requires_grad"] = omega_p1_requires_grad;
        context->saved_data["omega_p2_requires_grad"] = omega_p2_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [gradient_omega_p1, gradient_omega_p2] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::butterworth_analog_bandpass_filter_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                int64_t,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(gradient_output, n, omega_p1, omega_p2);

        return {gradient_omega_p1, gradient_omega_p2};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* context,
        const std::vector<at::Tensor>& gradient_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();
        int64_t n = context->saved_data["n"].toInt();
        const bool omega_p1_requires_grad = context->saved_data["omega_p1_requires_grad"].toBool();
        const bool omega_p2_requires_grad = context->saved_data["omega_p2_requires_grad"].toBool();

        const bool gradient_omega_p1_defined = gradient_outputs[0].defined();
        const bool gradient_omega_p2_defined = gradient_outputs[1].defined();

        if (!(gradient_omega_p1_defined && omega_p1_requires_grad) &&
            !(gradient_omega_p2_defined && omega_p2_requires_grad)) {
            return {
                at::Tensor(),  // grad_grad_output
                at::Tensor(),  // grad_n (not differentiable)
                at::Tensor(),  // grad_omega_p1
                at::Tensor(),  // grad_omega_p2
                at::Tensor(),  // grad_omega_p1_requires_grad
                at::Tensor()   // grad_omega_p2_requires_grad
            };
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor gradient_gradient_omega_p1_input;
        at::Tensor gradient_gradient_omega_p2_input;

        if (gradient_omega_p1_defined && omega_p1_requires_grad) {
            gradient_gradient_omega_p1_input = gradient_outputs[0];
        }
        if (gradient_omega_p2_defined && omega_p2_requires_grad) {
            gradient_gradient_omega_p2_input = gradient_outputs[1];
        }

        auto [gradient_gradient_output, gradient_omega_p1, gradient_omega_p2] =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow("torchscience::butterworth_analog_bandpass_filter_backward_backward", "")
                .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                    const at::Tensor&,
                    const at::Tensor&,
                    const at::Tensor&,
                    int64_t,
                    const at::Tensor&,
                    const at::Tensor&
                )>()
                .call(
                    gradient_gradient_omega_p1_input,
                    gradient_gradient_omega_p2_input,
                    saved[0],  // gradient_output
                    n,
                    saved[1],  // omega_p1
                    saved[2]   // omega_p2
                );

        return {
            gradient_gradient_output,
            at::Tensor(),  // grad_n (not differentiable)
            gradient_omega_p1,
            gradient_omega_p2,
            at::Tensor(),  // grad_omega_p1_requires_grad
            at::Tensor()   // grad_omega_p2_requires_grad
        };
    }
};

/**
 * Forward function class with autograd support.
 */
class ButterworthAnalogBandpassFilter
    : public torch::autograd::Function<ButterworthAnalogBandpassFilter> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* context,
        int64_t n,
        const at::Tensor& omega_p1,
        const at::Tensor& omega_p2
    ) {
        context->save_for_backward({omega_p1, omega_p2});
        context->saved_data["n"] = n;

        const bool condition = at::isFloatingType(omega_p1.scalar_type());

        context->saved_data["omega_p1_requires_grad"] = omega_p1.requires_grad() && condition;
        context->saved_data["omega_p2_requires_grad"] = omega_p2.requires_grad() && condition;

        at::AutoDispatchBelowAutograd guard;

        return c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::butterworth_analog_bandpass_filter", "")
            .typed<at::Tensor(int64_t, const at::Tensor&, const at::Tensor&)>()
            .call(n, omega_p1, omega_p2);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        const torch::autograd::variable_list& gradient_outputs
    ) {
        const torch::autograd::variable_list saved = context->get_saved_variables();

        at::Tensor omega_p1 = saved[0];
        at::Tensor omega_p2 = saved[1];
        int64_t n = context->saved_data["n"].toInt();

        at::Tensor gradient_output = gradient_outputs[0];

        bool omega_p1_requires_grad = context->saved_data["omega_p1_requires_grad"].toBool();
        bool omega_p2_requires_grad = context->saved_data["omega_p2_requires_grad"].toBool();

        std::vector<at::Tensor> gradients = ButterworthAnalogBandpassFilterBackward::apply(
            gradient_output,
            n,
            omega_p1,
            omega_p2,
            omega_p1_requires_grad,
            omega_p2_requires_grad
        );

        at::Tensor gradient_omega_p1;
        at::Tensor gradient_omega_p2;

        if (omega_p1_requires_grad) {
            gradient_omega_p1 = gradients[0];
        } else {
            gradient_omega_p1 = at::Tensor();
        }

        if (omega_p2_requires_grad) {
            gradient_omega_p2 = gradients[1];
        } else {
            gradient_omega_p2 = at::Tensor();
        }

        return {
            at::Tensor(),  // grad_n (not differentiable)
            gradient_omega_p1,
            gradient_omega_p2
        };
    }
};

inline at::Tensor butterworth_analog_bandpass_filter(
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    return ButterworthAnalogBandpassFilter::apply(n, omega_p1, omega_p2);
}

}  // namespace torchscience::autograd::filter

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl(
        "butterworth_analog_bandpass_filter",
        &torchscience::autograd::filter::butterworth_analog_bandpass_filter
    );
}
