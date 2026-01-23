// src/torchscience/csrc/cpu/linear_algebra/generalized_eigenvalue.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> generalized_eigenvalue(
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(b.dim() >= 2, "b must be at least 2D");
    TORCH_CHECK(a.size(-1) == a.size(-2), "a must be square");
    TORCH_CHECK(b.size(-1) == b.size(-2), "b must be square");
    TORCH_CHECK(a.size(-1) == b.size(-1), "a and b must have same size");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "a must be floating-point or complex");

    auto common_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    if (!at::isFloatingType(common_dtype) && !at::isComplexType(common_dtype)) {
        common_dtype = at::kDouble;
    }

    at::Tensor a_work = a.to(common_dtype).contiguous();
    at::Tensor b_work = b.to(common_dtype).contiguous();

    auto batch_shape_a = a_work.sizes().slice(0, a_work.dim() - 2);
    auto batch_shape_b = b_work.sizes().slice(0, b_work.dim() - 2);

    std::vector<int64_t> batch_vec_a(batch_shape_a.begin(), batch_shape_a.end());
    std::vector<int64_t> batch_vec_b(batch_shape_b.begin(), batch_shape_b.end());

    auto broadcast_shape = at::infer_size(batch_vec_a, batch_vec_b);

    std::vector<int64_t> a_expand_shape = broadcast_shape;
    a_expand_shape.push_back(a_work.size(-2));
    a_expand_shape.push_back(a_work.size(-1));

    a_work = a_work.expand(a_expand_shape).contiguous();
    b_work = b_work.expand(a_expand_shape).contiguous();

    int64_t n = a_work.size(-1);
    int64_t batch_size = a_work.numel() / (n * n);

    auto complex_dtype = at::toComplexType(
        at::isComplexType(common_dtype) ? common_dtype :
        (common_dtype == at::kFloat ? at::kFloat : at::kDouble)
    );

    std::vector<int64_t> eig_shape = broadcast_shape;
    eig_shape.push_back(n);

    std::vector<int64_t> vec_shape = broadcast_shape;
    vec_shape.push_back(n);
    vec_shape.push_back(n);

    at::Tensor eigenvalues = at::empty(eig_shape, a_work.options().dtype(complex_dtype));
    at::Tensor eigenvectors_left = at::empty(vec_shape, a_work.options().dtype(complex_dtype));
    at::Tensor eigenvectors_right = at::empty(vec_shape, a_work.options().dtype(complex_dtype));
    at::Tensor info = at::zeros(broadcast_shape, a_work.options().dtype(at::kInt));

    // Flatten batch dimensions for processing
    at::Tensor a_flat = a_work.reshape({batch_size, n, n});
    at::Tensor b_flat = b_work.reshape({batch_size, n, n});
    at::Tensor eig_flat = eigenvalues.reshape({batch_size, n});
    at::Tensor left_flat = eigenvectors_left.reshape({batch_size, n, n});
    at::Tensor right_flat = eigenvectors_right.reshape({batch_size, n, n});
    at::Tensor info_flat = info.reshape({batch_size});

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto a_slice = a_flat[batch];
        auto b_slice = b_flat[batch];

        try {
            // Solve via B^{-1}A
            at::Tensor b_inv_a = at::linalg_solve(b_slice, a_slice);
            auto [eig_vals, eig_vecs] = at::linalg_eig(b_inv_a);
            auto [_, left_vecs] = at::linalg_eig(b_inv_a.mH());

            eig_flat[batch].copy_(eig_vals);
            right_flat[batch].copy_(eig_vecs);
            left_flat[batch].copy_(left_vecs);
        } catch (const std::exception&) {
            info_flat[batch].fill_(1);
        }
    }

    return std::make_tuple(eigenvalues, eigenvectors_left, eigenvectors_right, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("generalized_eigenvalue", &torchscience::cpu::linear_algebra::generalized_eigenvalue);
}
