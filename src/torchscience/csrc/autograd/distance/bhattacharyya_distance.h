#pragma once

#include <string>
#include <tuple>

#include <torch/extension.h>

namespace torchscience::autograd::distance {

class BhattacharyyaDistance
    : public torch::autograd::Function<BhattacharyyaDistance> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& p,
        const at::Tensor& q,
        int64_t dim,
        const std::string& input_type,
        const std::string& reduction,
        bool pairwise
    ) {
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["input_type"] = input_type;
        ctx->saved_data["reduction"] = reduction;
        ctx->saved_data["pairwise"] = pairwise;

        bool p_requires_grad = p.requires_grad() && at::isFloatingType(p.scalar_type());
        bool q_requires_grad = q.requires_grad() && at::isFloatingType(q.scalar_type());

        ctx->saved_data["p_requires_grad"] = p_requires_grad;
        ctx->saved_data["q_requires_grad"] = q_requires_grad;

        at::AutoDispatchBelowAutograd guard;

        at::Tensor output = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::bhattacharyya_distance", "")
            .typed<at::Tensor(
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                bool
            )>()
            .call(p, q, dim, input_type, reduction, pairwise);

        ctx->save_for_backward({p, q, output});

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        const torch::autograd::variable_list& grad_outputs
    ) {
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        at::Tensor p = saved[0];
        at::Tensor q = saved[1];

        at::Tensor grad_output = grad_outputs[0];

        int64_t dim = ctx->saved_data["dim"].toInt();
        std::string input_type = ctx->saved_data["input_type"].toStringRef();
        std::string reduction = ctx->saved_data["reduction"].toStringRef();
        bool pairwise = ctx->saved_data["pairwise"].toBool();
        bool p_requires_grad = ctx->saved_data["p_requires_grad"].toBool();
        bool q_requires_grad = ctx->saved_data["q_requires_grad"].toBool();

        if (!p_requires_grad && !q_requires_grad) {
            return {
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor(),
                at::Tensor()
            };
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_p, grad_q] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::bhattacharyya_distance_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                int64_t,
                const std::string&,
                const std::string&,
                bool
            )>()
            .call(grad_output, p, q, dim, input_type, reduction, pairwise);

        return {
            p_requires_grad ? grad_p : at::Tensor(),
            q_requires_grad ? grad_q : at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor()
        };
    }
};

inline at::Tensor bhattacharyya_distance(
    const at::Tensor& p,
    const at::Tensor& q,
    int64_t dim,
    const std::string& input_type,
    const std::string& reduction,
    bool pairwise
) {
    return BhattacharyyaDistance::apply(p, q, dim, input_type, reduction, pairwise);
}

}  // namespace torchscience::autograd::distance

TORCH_LIBRARY_IMPL(torchscience, Autograd, module) {
    module.impl("bhattacharyya_distance", &torchscience::autograd::distance::bhattacharyya_distance);
}
