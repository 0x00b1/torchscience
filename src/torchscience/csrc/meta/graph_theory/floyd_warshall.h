// src/torchscience/csrc/meta/graph_theory/floyd_warshall.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graph_theory {

inline std::tuple<at::Tensor, at::Tensor, bool> floyd_warshall(
    const at::Tensor& input,
    bool directed
) {
    TORCH_CHECK(
        input.dim() >= 2,
        "floyd_warshall: input must be at least 2D, got ", input.dim(), "D"
    );
    TORCH_CHECK(
        input.size(-1) == input.size(-2),
        "floyd_warshall: last two dimensions must be equal, got ",
        input.size(-2), " x ", input.size(-1)
    );

    // Output shapes match input
    at::Tensor distances = at::empty_like(input);
    at::Tensor predecessors = at::empty(
        input.sizes(),
        input.options().dtype(at::kLong)
    );

    return std::make_tuple(distances, predecessors, false);
}

}  // namespace torchscience::meta::graph_theory

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("floyd_warshall", &torchscience::meta::graph_theory::floyd_warshall);
}
