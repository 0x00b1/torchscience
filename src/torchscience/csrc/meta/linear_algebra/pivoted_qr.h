// src/torchscience/csrc/meta/linear_algebra/pivoted_qr.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> pivoted_qr(
    const at::Tensor& a
) {
    TORCH_CHECK(a.dim() >= 2, "pivoted_qr: a must be at least 2D");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "pivoted_qr: a must be floating-point or complex");

    auto batch_shape = a.sizes().slice(0, a.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t m = a.size(-2);
    int64_t n = a.size(-1);
    int64_t k = std::min(m, n);

    auto dtype = a.scalar_type();

    // Q: (..., m, k) - orthogonal/unitary matrix
    std::vector<int64_t> q_shape = batch_vec;
    q_shape.push_back(m);
    q_shape.push_back(k);

    // R: (..., k, n) - upper triangular
    std::vector<int64_t> r_shape = batch_vec;
    r_shape.push_back(k);
    r_shape.push_back(n);

    // pivots: (..., n) - column permutation indices
    std::vector<int64_t> pivot_shape = batch_vec;
    pivot_shape.push_back(n);

    return std::make_tuple(
        at::empty(q_shape, a.options().dtype(dtype)),
        at::empty(r_shape, a.options().dtype(dtype)),
        at::empty(pivot_shape, a.options().dtype(at::kLong)),
        at::empty(batch_vec.empty() ? at::IntArrayRef({}) : batch_vec, a.options().dtype(at::kInt))
    );
}

}  // namespace torchscience::meta::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("pivoted_qr", &torchscience::meta::linear_algebra::pivoted_qr);
}
