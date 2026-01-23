// src/torchscience/csrc/meta/linear_algebra/ldl_decomposition.h
//
// Shape inference for LDL decomposition:
// Input: a (..., n, n) - symmetric/Hermitian matrix
// Output:
//   L: (..., n, n) - unit lower triangular
//   D: (..., n, n) - diagonal matrix
//   pivots: (..., n) - pivot indices
//   info: (...) - 0 indicates success
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ldl_decomposition(
    const at::Tensor& a
) {
    TORCH_CHECK(a.dim() >= 2, "ldl_decomposition: a must be at least 2D");
    TORCH_CHECK(a.size(-2) == a.size(-1), "ldl_decomposition: a must be square");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "ldl_decomposition: a must be floating-point or complex");

    auto batch_shape = a.sizes().slice(0, a.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t n = a.size(-1);

    auto dtype = a.scalar_type();

    // L: (..., n, n) - unit lower triangular
    std::vector<int64_t> matrix_shape = batch_vec;
    matrix_shape.push_back(n);
    matrix_shape.push_back(n);

    // pivots: (..., n) - pivot indices
    std::vector<int64_t> pivot_shape = batch_vec;
    pivot_shape.push_back(n);

    return std::make_tuple(
        at::empty(matrix_shape, a.options().dtype(dtype)),  // L
        at::empty(matrix_shape, a.options().dtype(dtype)),  // D
        at::empty(pivot_shape, a.options().dtype(at::kLong)),  // pivots
        at::empty(batch_vec.empty() ? at::IntArrayRef({}) : batch_vec, a.options().dtype(at::kInt))  // info
    );
}

}  // namespace torchscience::meta::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("ldl_decomposition", &torchscience::meta::linear_algebra::ldl_decomposition);
}
