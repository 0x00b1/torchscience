// src/torchscience/csrc/autograd/statistics/hypothesis_test/f_oneway.h
#pragma once

#include <tuple>
#include <vector>

#include <torch/extension.h>

namespace torchscience::autograd::statistics::hypothesis_test {

/**
 * Autograd Function for f_oneway.
 *
 * Computes F-statistic and p-value with gradient support for the statistic.
 * Note: The p-value gradient is not computed (would require F-distribution derivative).
 */
class FOneway : public torch::autograd::Function<FOneway> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& data,
        const at::Tensor& group_sizes
    ) {
        bool data_requires_grad = data.requires_grad() && at::isFloatingType(data.scalar_type());
        ctx->saved_data["data_requires_grad"] = data_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        auto [statistic, pvalue] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::f_oneway", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(data, group_sizes);

        if (data_requires_grad) {
            ctx->save_for_backward({data, group_sizes});
        }

        return {statistic, pvalue};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        bool data_requires_grad = ctx->saved_data["data_requires_grad"].toBool();

        if (!data_requires_grad) {
            return {at::Tensor(), at::Tensor()};
        }

        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor data = saved[0];
        at::Tensor group_sizes = saved[1];

        at::Tensor grad_statistic = grad_outputs[0];
        // grad_outputs[1] is gradient w.r.t. pvalue, which we ignore
        // (p-value gradient through F-distribution SF is complex)

        if (!grad_statistic.defined()) {
            return {at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_data = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::f_oneway_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&
            )>()
            .call(grad_statistic, data, group_sizes);

        return {grad_data, at::Tensor()};  // No gradient for group_sizes
    }
};

/**
 * Wrapper function for f_oneway with autograd support.
 */
inline std::tuple<at::Tensor, at::Tensor> f_oneway(
    const at::Tensor& data,
    const at::Tensor& group_sizes
) {
    auto results = FOneway::apply(data, group_sizes);
    return std::make_tuple(results[0], results[1]);
}

}  // namespace torchscience::autograd::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("f_oneway", &torchscience::autograd::statistics::hypothesis_test::f_oneway);
}
