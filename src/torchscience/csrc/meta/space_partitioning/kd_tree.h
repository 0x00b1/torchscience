// src/torchscience/csrc/meta/space_partitioning/kd_tree.h
#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/library.h>

namespace torchscience::meta::space_partitioning {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
kd_tree_build_batched(
    const at::Tensor& points,
    int64_t leaf_size
) {
    TORCH_CHECK(points.dim() == 3, "kd_tree_build_batched: points must be 3D (B, n, d)");
    TORCH_CHECK(leaf_size > 0, "kd_tree_build_batched: leaf_size must be > 0");

    auto B = points.sym_size(0);
    auto n = points.sym_size(1);
    auto d = points.sym_size(2);

    // Tree node count and leaf count are data-dependent (depend on point distribution)
    // TODO: Research correct PyTorch 2.x API for unbacked SymInt creation
    // Placeholder: Use conservative upper bounds until correct API is verified
    // Upper bounds: n_nodes <= 2*n - 1, n_leaves <= ceil(n / leaf_size)
    auto n_nodes_upper = 2 * n - 1;
    auto n_leaves_upper = (n + leaf_size - 1) / leaf_size;

    return std::make_tuple(
        at::empty_symint({B, n, d}, points.options()),
        at::empty_symint({B, n_nodes_upper}, points.options().dtype(at::kLong)),
        at::empty_symint({B, n_nodes_upper}, points.options()),  // split_val matches input dtype
        at::empty_symint({B, n_nodes_upper}, points.options().dtype(at::kLong)),
        at::empty_symint({B, n_nodes_upper}, points.options().dtype(at::kLong)),
        at::empty_symint({B, n}, points.options().dtype(at::kLong)),
        at::empty_symint({B, n_leaves_upper}, points.options().dtype(at::kLong)),
        at::empty_symint({B, n_leaves_upper}, points.options().dtype(at::kLong))
    );
}

}  // namespace torchscience::meta::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("kd_tree_build_batched", torchscience::meta::space_partitioning::kd_tree_build_batched);
}
