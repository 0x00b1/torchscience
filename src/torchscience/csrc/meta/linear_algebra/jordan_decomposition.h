// src/torchscience/csrc/meta/linear_algebra/jordan_decomposition.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> jordan_decomposition(
    const at::Tensor& a
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(a.size(-1) == a.size(-2), "a must be square");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "jordan_decomposition: a must be floating-point or complex");

    auto batch_shape = a.sizes().slice(0, a.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t n = a.size(-1);

    auto dtype = a.scalar_type();

    // Determine the complex output dtype
    at::ScalarType complex_dtype;
    if (at::isComplexType(dtype)) {
        complex_dtype = dtype;
    } else if (dtype == at::kFloat) {
        complex_dtype = at::kComplexFloat;
    } else {
        complex_dtype = at::kComplexDouble;
    }

    std::vector<int64_t> mat_shape = batch_vec;
    mat_shape.push_back(n);
    mat_shape.push_back(n);

    // J: (..., n, n) complex - Jordan normal form
    // P: (..., n, n) complex - Similarity transformation matrix
    // info: (...) int - Convergence info
    return std::make_tuple(
        at::empty(mat_shape, a.options().dtype(complex_dtype)),
        at::empty(mat_shape, a.options().dtype(complex_dtype)),
        at::empty(batch_vec, a.options().dtype(at::kInt))
    );
}

}  // namespace torchscience::meta::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("jordan_decomposition", &torchscience::meta::linear_algebra::jordan_decomposition);
}
