// src/torchscience/csrc/autograd/space_partitioning/k_nearest_neighbors.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::space_partitioning {

class KNearestNeighbors
    : public torch::autograd::Function<KNearestNeighbors> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& points,
        const at::Tensor& split_dim,
        const at::Tensor& split_val,
        const at::Tensor& left,
        const at::Tensor& right,
        const at::Tensor& indices,
        const at::Tensor& leaf_starts,
        const at::Tensor& leaf_counts,
        const at::Tensor& queries,
        int64_t k,
        double p
    ) {
        ctx->saved_data["p"] = p;
        ctx->saved_data["k"] = k;
        ctx->saved_data["queries_requires_grad"] = queries.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        auto [result_indices, result_distances] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::k_nearest_neighbors", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                int64_t, double
            )>()
            .call(points, split_dim, split_val, left, right, indices,
                  leaf_starts, leaf_counts, queries, k, p);

        ctx->save_for_backward({points, queries, result_indices, result_distances});

        return {result_indices, result_distances};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor points = saved[0];
        at::Tensor queries = saved[1];
        at::Tensor result_indices = saved[2];
        at::Tensor result_distances = saved[3];

        double p = ctx->saved_data["p"].toDouble();
        int64_t k = ctx->saved_data["k"].toInt();
        bool queries_requires_grad = ctx->saved_data["queries_requires_grad"].toBool();

        at::Tensor grad_distances = grad_outputs[1];  // grad for distances

        if (!queries_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor()};
        }

        // Vectorized gradient computation
        int64_t m = queries.size(0);
        int64_t d = queries.size(1);

        // Gather neighbor points: (m, k, d)
        at::Tensor gathered_points = at::index_select(
            points, 0, result_indices.flatten()
        ).view({m, k, d});

        // diff = query - neighbor: (m, k, d)
        at::Tensor diff = queries.unsqueeze(1) - gathered_points;

        // dist: (m, k, 1)
        at::Tensor dist = result_distances.unsqueeze(-1);

        // Safe gradient: avoid division by zero using where
        at::Tensor is_zero = result_distances < 1e-8;  // (m, k)
        at::Tensor safe_dist = dist.clamp_min(1e-8);

        // grad = diff / dist for L2, zero where dist is zero
        at::Tensor grad_component = at::where(
            is_zero.unsqueeze(-1).expand_as(diff),
            at::zeros_like(diff),
            diff / safe_dist
        );

        // Scale by incoming gradient and sum over k
        grad_component = grad_component * grad_distances.unsqueeze(-1);
        at::Tensor grad_queries = grad_component.sum(1);

        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                grad_queries, at::Tensor(), at::Tensor()};
    }
};

inline std::tuple<at::Tensor, at::Tensor> k_nearest_neighbors(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    int64_t k,
    double p
) {
    auto results = KNearestNeighbors::apply(
        points, split_dim, split_val, left, right, indices,
        leaf_starts, leaf_counts, queries, k, p
    );
    return std::make_tuple(results[0], results[1]);
}

}  // namespace torchscience::autograd::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("k_nearest_neighbors", &torchscience::autograd::space_partitioning::k_nearest_neighbors);
}
