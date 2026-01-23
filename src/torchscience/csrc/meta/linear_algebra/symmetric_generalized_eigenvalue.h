// src/torchscience/csrc/meta/linear_algebra/symmetric_generalized_eigenvalue.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> symmetric_generalized_eigenvalue(
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(b.dim() >= 2, "b must be at least 2D");
    TORCH_CHECK(a.size(-1) == a.size(-2), "a must be square");
    TORCH_CHECK(b.size(-1) == b.size(-2), "b must be square");
    TORCH_CHECK(a.size(-1) == b.size(-1), "a and b must have same size");

    // Compute output shapes
    auto batch_shape_a = a.sizes().slice(0, a.dim() - 2);
    auto batch_shape_b = b.sizes().slice(0, b.dim() - 2);

    std::vector<int64_t> batch_vec_a(batch_shape_a.begin(), batch_shape_a.end());
    std::vector<int64_t> batch_vec_b(batch_shape_b.begin(), batch_shape_b.end());

    auto broadcast_shape = at::infer_size(batch_vec_a, batch_vec_b);

    int64_t n = a.size(-1);

    // Determine output dtype
    auto common_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    if (common_dtype != at::kFloat && common_dtype != at::kDouble) {
        common_dtype = at::kDouble;
    }

    // eigenvalues: (..., n)
    std::vector<int64_t> eig_shape = broadcast_shape;
    eig_shape.push_back(n);

    // eigenvectors: (..., n, n)
    std::vector<int64_t> vec_shape = broadcast_shape;
    vec_shape.push_back(n);
    vec_shape.push_back(n);

    at::Tensor eigenvalues = at::empty_symint(eig_shape, a.options().dtype(common_dtype));
    at::Tensor eigenvectors = at::empty_symint(vec_shape, a.options().dtype(common_dtype));
    at::Tensor info = at::empty_symint(broadcast_shape, a.options().dtype(at::kInt));

    return std::make_tuple(eigenvalues, eigenvectors, info);
}

}  // namespace torchscience::meta::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("symmetric_generalized_eigenvalue", &torchscience::meta::linear_algebra::symmetric_generalized_eigenvalue);
}
