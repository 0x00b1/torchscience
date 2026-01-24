// src/torchscience/csrc/cpu/linear_algebra/symmetric_generalized_eigenvalue.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> symmetric_generalized_eigenvalue(
    const at::Tensor& a,
    const at::Tensor& b
) {
    TORCH_CHECK(
        a.dim() >= 2,
        "symmetric_generalized_eigenvalue: a must be at least 2D, got ", a.dim(), "D"
    );
    TORCH_CHECK(
        b.dim() >= 2,
        "symmetric_generalized_eigenvalue: b must be at least 2D, got ", b.dim(), "D"
    );
    TORCH_CHECK(
        a.size(-1) == a.size(-2),
        "symmetric_generalized_eigenvalue: a must be square, got ",
        a.size(-2), " x ", a.size(-1)
    );
    TORCH_CHECK(
        b.size(-1) == b.size(-2),
        "symmetric_generalized_eigenvalue: b must be square, got ",
        b.size(-2), " x ", b.size(-1)
    );
    TORCH_CHECK(
        a.size(-1) == b.size(-1),
        "symmetric_generalized_eigenvalue: a and b must have same size, got ",
        a.size(-1), " and ", b.size(-1)
    );
    TORCH_CHECK(
        at::isFloatingType(a.scalar_type()),
        "symmetric_generalized_eigenvalue: a must be floating-point, got ", a.scalar_type()
    );

    // Promote to common dtype
    auto common_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    if (common_dtype != at::kFloat && common_dtype != at::kDouble) {
        common_dtype = at::kDouble;
    }

    at::Tensor a_work = a.to(common_dtype).contiguous();
    at::Tensor b_work = b.to(common_dtype).contiguous();

    // Broadcast batch dimensions
    auto batch_shape_a = a_work.sizes().slice(0, a_work.dim() - 2);
    auto batch_shape_b = b_work.sizes().slice(0, b_work.dim() - 2);

    std::vector<int64_t> batch_vec_a(batch_shape_a.begin(), batch_shape_a.end());
    std::vector<int64_t> batch_vec_b(batch_shape_b.begin(), batch_shape_b.end());

    auto broadcast_shape = at::infer_size(batch_vec_a, batch_vec_b);

    std::vector<int64_t> a_expand_shape = broadcast_shape;
    a_expand_shape.push_back(a_work.size(-2));
    a_expand_shape.push_back(a_work.size(-1));

    std::vector<int64_t> b_expand_shape = broadcast_shape;
    b_expand_shape.push_back(b_work.size(-2));
    b_expand_shape.push_back(b_work.size(-1));

    a_work = a_work.expand(a_expand_shape).contiguous();
    b_work = b_work.expand(b_expand_shape).contiguous();

    int64_t n = a_work.size(-1);

    // Use Cholesky-based approach: B = LL^T, solve standard eig of L^{-1}AL^{-T}
    at::Tensor eigenvalues;
    at::Tensor eigenvectors;
    at::Tensor info;

    try {
        // Cholesky decomposition of B
        at::Tensor L = at::linalg_cholesky(b_work);

        // Transform: Y = L^{-1} A (solve L Y = A)
        at::Tensor Y = at::linalg_solve_triangular(L, a_work, /*upper=*/false);

        // C = Y L^{-T} = L^{-1} A L^{-T}
        at::Tensor C = at::linalg_solve_triangular(L, Y.mH(), /*upper=*/false).mH();

        // Standard symmetric eigenvalue problem
        std::tie(eigenvalues, eigenvectors) = at::linalg_eigh(C);

        // Back-transform: X = L^{-T} Y_eig
        eigenvectors = at::linalg_solve_triangular(L.mH(), eigenvectors, /*upper=*/true);

        // Success
        info = at::zeros(broadcast_shape, a_work.options().dtype(at::kInt));

    } catch (const std::exception&) {
        // Cholesky failed - B is not positive definite
        std::vector<int64_t> eig_shape = broadcast_shape;
        eig_shape.push_back(n);

        std::vector<int64_t> vec_shape = broadcast_shape;
        vec_shape.push_back(n);
        vec_shape.push_back(n);

        eigenvalues = at::full(eig_shape, std::numeric_limits<double>::quiet_NaN(), a_work.options());
        eigenvectors = at::full(vec_shape, std::numeric_limits<double>::quiet_NaN(), a_work.options());
        info = at::ones(broadcast_shape, a_work.options().dtype(at::kInt));
    }

    return std::make_tuple(eigenvalues, eigenvectors, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("symmetric_generalized_eigenvalue", &torchscience::cpu::linear_algebra::symmetric_generalized_eigenvalue);
}
