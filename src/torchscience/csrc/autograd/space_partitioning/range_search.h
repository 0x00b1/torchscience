#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::space_partitioning {

class RangeSearch
    : public torch::autograd::Function<RangeSearch> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& points,
        const at::Tensor& split_dim,
        const at::Tensor& split_val,
        const at::Tensor& left,
        const at::Tensor& right,
        const at::Tensor& indices_tree,
        const at::Tensor& leaf_starts,
        const at::Tensor& leaf_counts,
        const at::Tensor& queries,
        double radius,
        double p
    ) {
        ctx->saved_data["p"] = p;
        ctx->saved_data["queries_requires_grad"] = queries.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        auto [result_indices, result_distances] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::range_search", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                double, double
            )>()
            .call(points, split_dim, split_val, left, right, indices_tree,
                  leaf_starts, leaf_counts, queries, radius, p);

        // Save regular tensors via save_for_backward
        ctx->save_for_backward({points, queries});

        // Store nested tensors in saved_data (IValue can hold any tensor)
        ctx->saved_data["result_indices"] = result_indices;
        ctx->saved_data["result_distances"] = result_distances;

        return {result_indices, result_distances};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor points = saved[0];
        at::Tensor queries = saved[1];

        // Retrieve nested tensors from saved_data
        at::Tensor result_indices = ctx->saved_data["result_indices"].toTensor();
        at::Tensor result_distances = ctx->saved_data["result_distances"].toTensor();

        bool queries_requires_grad = ctx->saved_data["queries_requires_grad"].toBool();

        if (!queries_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::Tensor grad_distances = grad_outputs[1];  // nested tensor

        int64_t m = queries.size(0);
        int64_t d = queries.size(1);

        // Vectorized gradient using scatter_add
        // First, unbind nested tensors
        std::vector<at::Tensor> idx_list = result_indices.unbind();
        std::vector<at::Tensor> dist_list = result_distances.unbind();
        std::vector<at::Tensor> grad_dist_list = grad_distances.unbind();

        at::Tensor grad_queries = at::zeros_like(queries);

        // Process each query
        for (int64_t q = 0; q < m; ++q) {
            at::Tensor q_indices = idx_list[q];
            at::Tensor q_distances = dist_list[q];
            at::Tensor q_grad_distances = grad_dist_list[q];

            if (q_indices.size(0) == 0) continue;

            // Gather neighbor points
            at::Tensor neighbor_points = at::index_select(points, 0, q_indices);
            at::Tensor diff = queries[q].unsqueeze(0) - neighbor_points;  // (count, d)

            // Safe distance gradient
            at::Tensor safe_dist = q_distances.clamp_min(1e-8).unsqueeze(-1);
            at::Tensor is_zero = q_distances < 1e-8;

            at::Tensor grad_component = at::where(
                is_zero.unsqueeze(-1).expand_as(diff),
                at::zeros_like(diff),
                diff / safe_dist
            );

            grad_component = grad_component * q_grad_distances.unsqueeze(-1);
            grad_queries[q] = grad_component.sum(0);
        }

        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                grad_queries, at::Tensor(), at::Tensor()};
    }
};

inline std::tuple<at::Tensor, at::Tensor> range_search(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices_tree,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    double radius,
    double p
) {
    auto results = RangeSearch::apply(
        points, split_dim, split_val, left, right, indices_tree,
        leaf_starts, leaf_counts, queries, radius, p
    );
    return std::make_tuple(results[0], results[1]);
}

}  // namespace torchscience::autograd::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("range_search", &torchscience::autograd::space_partitioning::range_search);
}
