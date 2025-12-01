// src/torchscience/csrc/meta/space_partitioning/k_nearest_neighbors.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::space_partitioning {

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
    TORCH_CHECK(queries.dim() == 2, "k_nearest_neighbors: queries must be 2D");
    int64_t m = queries.size(0);

    return std::make_tuple(
        at::empty({m, k}, queries.options().dtype(at::kLong)),
        at::empty({m, k}, queries.options())
    );
}

}  // namespace torchscience::meta::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("k_nearest_neighbors", &torchscience::meta::space_partitioning::k_nearest_neighbors);
}
