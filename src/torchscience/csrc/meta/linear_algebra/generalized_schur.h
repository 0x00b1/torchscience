// src/torchscience/csrc/meta/linear_algebra/generalized_schur.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
generalized_schur(
    const at::Tensor& a,
    const at::Tensor& b,
    c10::string_view output
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(b.dim() >= 2, "b must be at least 2D");
    TORCH_CHECK(output == "real" || output == "complex", "output must be 'real' or 'complex'");

    auto batch_shape_a = a.sizes().slice(0, a.dim() - 2);
    auto batch_shape_b = b.sizes().slice(0, b.dim() - 2);

    std::vector<int64_t> batch_vec_a(batch_shape_a.begin(), batch_shape_a.end());
    std::vector<int64_t> batch_vec_b(batch_shape_b.begin(), batch_shape_b.end());

    auto broadcast_shape = at::infer_size(batch_vec_a, batch_vec_b);

    int64_t n = a.size(-1);

    auto common_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    bool complex_output = (output == "complex") || at::isComplexType(common_dtype);

    auto complex_dtype = at::toComplexType(
        at::isComplexType(common_dtype) ? common_dtype :
        (common_dtype == at::kFloat ? at::kFloat : at::kDouble)
    );
    auto output_dtype = complex_output ? complex_dtype : common_dtype;

    std::vector<int64_t> mat_shape = broadcast_shape;
    mat_shape.push_back(n);
    mat_shape.push_back(n);

    std::vector<int64_t> eig_shape = broadcast_shape;
    eig_shape.push_back(n);

    return std::make_tuple(
        at::empty(mat_shape, a.options().dtype(output_dtype)),   // S
        at::empty(mat_shape, a.options().dtype(output_dtype)),   // T
        at::empty(eig_shape, a.options().dtype(complex_dtype)),  // alpha
        at::empty(eig_shape, a.options().dtype(complex_dtype)),  // beta
        at::empty(mat_shape, a.options().dtype(output_dtype)),   // Q
        at::empty(mat_shape, a.options().dtype(output_dtype)),   // Z
        at::empty(broadcast_shape, a.options().dtype(at::kInt))  // info
    );
}

}  // namespace torchscience::meta::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("generalized_schur", &torchscience::meta::linear_algebra::generalized_schur);
}
