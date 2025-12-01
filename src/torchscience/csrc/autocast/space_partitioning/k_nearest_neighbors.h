// src/torchscience/csrc/autocast/space_partitioning/k_nearest_neighbors.h
#pragma once

#include <ATen/autocast_mode.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autocast::space_partitioning {

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
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Upcast to float32 for precision
    at::Tensor points_fp32 = at::autocast::cached_cast(at::kFloat, points);
    at::Tensor split_val_fp32 = at::autocast::cached_cast(at::kFloat, split_val);
    at::Tensor queries_fp32 = at::autocast::cached_cast(at::kFloat, queries);

    auto [result_indices, distances] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::k_nearest_neighbors", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            int64_t, double
        )>()
        .call(points_fp32, split_dim, split_val_fp32, left, right, indices,
              leaf_starts, leaf_counts, queries_fp32, k, p);

    return std::make_tuple(result_indices, distances.to(queries.scalar_type()));
}

}  // namespace torchscience::autocast::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("k_nearest_neighbors", &torchscience::autocast::space_partitioning::k_nearest_neighbors);
}
