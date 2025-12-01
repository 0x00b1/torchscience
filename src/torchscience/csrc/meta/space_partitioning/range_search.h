#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::space_partitioning {

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
    int64_t n = points.size(0);
    int64_t m = queries.size(0);

    // Return empty nested tensors with correct structure
    // Actual sizes are data-dependent
    std::vector<at::Tensor> idx_tensors(m);
    std::vector<at::Tensor> dist_tensors(m);

    for (int64_t i = 0; i < m; ++i) {
        // Conservative: each query could match all points
        idx_tensors[i] = at::empty({n}, queries.options().dtype(at::kLong));
        dist_tensors[i] = at::empty({n}, queries.options());
    }

    return std::make_tuple(
        at::_nested_tensor_from_tensor_list(idx_tensors),
        at::_nested_tensor_from_tensor_list(dist_tensors)
    );
}

}  // namespace torchscience::meta::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("range_search", &torchscience::meta::space_partitioning::range_search);
}
