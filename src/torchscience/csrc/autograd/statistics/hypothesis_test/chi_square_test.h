#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/autograd.h>
#include <torch/library.h>

namespace torchscience::autograd::statistics::hypothesis_test {

/**
 * Autograd function for chi-square test backward pass.
 */
class ChiSquareTestBackward
    : public torch::autograd::Function<ChiSquareTestBackward> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& grad_statistic,
        const at::Tensor& observed,
        const c10::optional<at::Tensor>& expected,
        bool observed_requires_grad
    ) {
        ctx->save_for_backward({grad_statistic, observed});
        if (expected.has_value()) {
            ctx->save_for_backward({grad_statistic, observed, *expected});
            ctx->saved_data["has_expected"] = true;
        } else {
            ctx->save_for_backward({grad_statistic, observed});
            ctx->saved_data["has_expected"] = false;
        }
        ctx->saved_data["observed_requires_grad"] = observed_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor grad_observed = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::chi_square_test_backward", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                const c10::optional<at::Tensor>&
            )>()
            .call(grad_statistic, observed, expected);

        return grad_observed;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        // Second-order gradients not implemented
        return {
            at::Tensor(),  // grad_grad_statistic
            at::Tensor(),  // grad_observed
            at::Tensor(),  // grad_expected
            at::Tensor()   // grad_observed_requires_grad
        };
    }
};

/**
 * Autograd function for chi-square test.
 */
class ChiSquareTestFunction
    : public torch::autograd::Function<ChiSquareTestFunction> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& observed,
        const c10::optional<at::Tensor>& expected,
        int64_t ddof
    ) {
        bool observed_requires_grad = observed.requires_grad() &&
            at::isFloatingType(observed.scalar_type());
        ctx->saved_data["observed_requires_grad"] = observed_requires_grad;
        ctx->saved_data["ddof"] = ddof;

        if (expected.has_value()) {
            ctx->save_for_backward({observed, *expected});
            ctx->saved_data["has_expected"] = true;
        } else {
            ctx->save_for_backward({observed});
            ctx->saved_data["has_expected"] = false;
        }

        at::AutoDispatchBelowAutograd guard;

        auto [statistic, pvalue] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::chi_square_test", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const c10::optional<at::Tensor>&,
                int64_t
            )>()
            .call(observed, expected, ddof);

        return {statistic, pvalue};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor observed = saved[0];

        c10::optional<at::Tensor> expected;
        if (ctx->saved_data["has_expected"].toBool()) {
            expected = saved[1];
        }

        at::Tensor grad_statistic = grad_outputs[0];
        bool observed_requires_grad = ctx->saved_data["observed_requires_grad"].toBool();

        if (!observed_requires_grad || !grad_statistic.defined()) {
            return {at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::Tensor grad_observed = ChiSquareTestBackward::apply(
            grad_statistic,
            observed,
            expected,
            observed_requires_grad
        );

        return {grad_observed, at::Tensor(), at::Tensor()};
    }
};

inline std::tuple<at::Tensor, at::Tensor> chi_square_test(
    const at::Tensor& observed,
    const c10::optional<at::Tensor>& expected,
    int64_t ddof
) {
    auto results = ChiSquareTestFunction::apply(observed, expected, ddof);
    return std::make_tuple(results[0], results[1]);
}

}  // namespace torchscience::autograd::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("chi_square_test", &torchscience::autograd::statistics::hypothesis_test::chi_square_test);
}
