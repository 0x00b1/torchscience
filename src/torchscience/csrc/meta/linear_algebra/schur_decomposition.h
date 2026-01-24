// src/torchscience/csrc/meta/linear_algebra/schur_decomposition.h
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> schur_decomposition(
    const at::Tensor& a,
    c10::string_view output
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(output == "real" || output == "complex", "output must be 'real' or 'complex'");

    bool complex_output = (output == "complex") || at::isComplexType(a.scalar_type());

    auto batch_shape = a.sizes().slice(0, a.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t n = a.size(-1);

    auto dtype = a.scalar_type();
    auto complex_dtype = at::toComplexType(
        at::isComplexType(dtype) ? dtype : (dtype == at::kFloat ? at::kFloat : at::kDouble)
    );
    auto output_dtype = complex_output ? complex_dtype : dtype;

    std::vector<int64_t> mat_shape = batch_vec;
    mat_shape.push_back(n);
    mat_shape.push_back(n);

    std::vector<int64_t> eig_shape = batch_vec;
    eig_shape.push_back(n);

    return std::make_tuple(
        at::empty(mat_shape, a.options().dtype(output_dtype)),
        at::empty(mat_shape, a.options().dtype(output_dtype)),
        at::empty(eig_shape, a.options().dtype(complex_dtype)),
        at::empty(batch_vec, a.options().dtype(at::kInt))
    );
}

}  // namespace torchscience::meta::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("schur_decomposition", &torchscience::meta::linear_algebra::schur_decomposition);
}
