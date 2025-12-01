#pragma once

#include <ATen/autocast_mode.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autocast::space_partitioning {

inline std::tuple<at::Tensor, at::Tensor> range_search(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    double radius,
    double p
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    at::Tensor points_fp32 = at::autocast::cached_cast(at::kFloat, points);
    at::Tensor split_val_fp32 = at::autocast::cached_cast(at::kFloat, split_val);
    at::Tensor queries_fp32 = at::autocast::cached_cast(at::kFloat, queries);

    auto [result_indices, distances] = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::range_search", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            const at::Tensor&, const at::Tensor&, const at::Tensor&,
            double, double
        )>()
        .call(points_fp32, split_dim, split_val_fp32, left, right, indices,
              leaf_starts, leaf_counts, queries_fp32, radius, p);

    // Cast nested tensor distances back
    return std::make_tuple(result_indices, distances.to(queries.scalar_type()));
}

}  // namespace torchscience::autocast::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("range_search", &torchscience::autocast::space_partitioning::range_search);
}
