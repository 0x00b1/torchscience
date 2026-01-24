// src/torchscience/csrc/cpu/linear_algebra/generalized_schur.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
generalized_schur(
    const at::Tensor& a,
    const at::Tensor& b,
    c10::string_view output
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(b.dim() >= 2, "b must be at least 2D");
    TORCH_CHECK(a.size(-1) == a.size(-2), "a must be square");
    TORCH_CHECK(b.size(-1) == b.size(-2), "b must be square");
    TORCH_CHECK(a.size(-1) == b.size(-1), "a and b must have the same size");
    TORCH_CHECK(output == "real" || output == "complex",
        "output must be 'real' or 'complex', got '", output, "'");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "a must be floating-point or complex");
    TORCH_CHECK(at::isFloatingType(b.scalar_type()) || at::isComplexType(b.scalar_type()),
        "b must be floating-point or complex");

    auto common_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
    if (!at::isFloatingType(common_dtype) && !at::isComplexType(common_dtype)) {
        common_dtype = at::kDouble;
    }

    bool complex_output = (output == "complex") || at::isComplexType(common_dtype);

    at::Tensor a_work = a.to(common_dtype).contiguous();
    at::Tensor b_work = b.to(common_dtype).contiguous();

    // Handle batch dimension broadcasting
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
    auto output_dtype = complex_output ? complex_dtype : common_dtype;

    std::vector<int64_t> mat_shape = broadcast_shape;
    mat_shape.push_back(n);
    mat_shape.push_back(n);

    std::vector<int64_t> eig_shape = broadcast_shape;
    eig_shape.push_back(n);

    // Outputs: S, T, alpha, beta, Q, Z, info
    at::Tensor S = at::empty(mat_shape, a_work.options().dtype(output_dtype));
    at::Tensor T = at::empty(mat_shape, a_work.options().dtype(output_dtype));
    at::Tensor alpha = at::empty(eig_shape, a_work.options().dtype(complex_dtype));
    at::Tensor beta = at::empty(eig_shape, a_work.options().dtype(complex_dtype));
    at::Tensor Q = at::empty(mat_shape, a_work.options().dtype(output_dtype));
    at::Tensor Z = at::empty(mat_shape, a_work.options().dtype(output_dtype));
    at::Tensor info = at::zeros(broadcast_shape, a_work.options().dtype(at::kInt));

    // Flatten for batch processing
    at::Tensor a_flat = a_work.reshape({batch_size, n, n});
    at::Tensor b_flat = b_work.reshape({batch_size, n, n});
    at::Tensor S_flat = S.reshape({batch_size, n, n});
    at::Tensor T_flat = T.reshape({batch_size, n, n});
    at::Tensor alpha_flat = alpha.reshape({batch_size, n});
    at::Tensor beta_flat = beta.reshape({batch_size, n});
    at::Tensor Q_flat = Q.reshape({batch_size, n, n});
    at::Tensor Z_flat = Z.reshape({batch_size, n, n});
    at::Tensor info_flat = info.reshape({batch_size});

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto a_slice = a_flat[batch];
        auto b_slice = b_flat[batch];

        try {
            // Convert to complex for unified processing
            at::Tensor a_complex = a_slice.to(complex_dtype);
            at::Tensor b_complex = b_slice.to(complex_dtype);

            // Compute generalized eigenvalues via B^{-1}A
            // This gives us the eigenvalues of the pencil (A, B)
            at::Tensor b_inv_a = at::linalg_solve(b_complex, a_complex);
            auto [eig_vals, eig_vecs] = at::linalg_eig(b_inv_a);

            // Orthogonalize eigenvectors using QR to get unitary Z
            auto [Z_unitary, R_z] = at::linalg_qr(eig_vecs);

            // For the generalized Schur form:
            // A = Q @ S @ Z.H and B = Q @ T @ Z.H
            // where S and T are upper triangular

            // Compute S = Q.H @ A @ Z (should be upper triangular)
            // Compute T = Q.H @ B @ Z (should be upper triangular)

            // For the QZ decomposition, we need Q such that:
            // S = Q.H @ A @ Z is upper triangular
            // T = Q.H @ B @ Z is upper triangular

            // One approach: use similarity transformation
            // First, compute transformed matrices using Z
            at::Tensor a_transformed = at::matmul(a_complex, Z_unitary);
            at::Tensor b_transformed = at::matmul(b_complex, Z_unitary);

            // Now we need Q such that Q.H @ a_transformed and Q.H @ b_transformed
            // are both upper triangular. We use QR on b_transformed to get Q.
            auto [Q_unitary, T_computed] = at::linalg_qr(b_transformed);

            // Compute S = Q.H @ A @ Z
            at::Tensor S_computed = at::matmul(Q_unitary.conj().transpose(-2, -1), a_transformed);

            // Extract generalized eigenvalues
            // For the generalized Schur form: eigenvalue_i = alpha_i / beta_i
            // where alpha_i = S[i,i] and beta_i = T[i,i]
            at::Tensor alpha_diag = S_computed.diagonal(0, -2, -1);
            at::Tensor beta_diag = T_computed.diagonal(0, -2, -1);

            at::Tensor S_slice, T_slice, Q_slice, Z_slice;

            if (complex_output) {
                S_slice = S_computed;
                T_slice = T_computed;
                Q_slice = Q_unitary;
                Z_slice = Z_unitary;
            } else {
                // For real output, take real parts
                S_slice = at::real(S_computed);
                T_slice = at::real(T_computed);
                Q_slice = at::real(Q_unitary);
                Z_slice = at::real(Z_unitary);
            }

            S_flat[batch].copy_(S_slice);
            T_flat[batch].copy_(T_slice);
            alpha_flat[batch].copy_(alpha_diag);
            beta_flat[batch].copy_(beta_diag);
            Q_flat[batch].copy_(Q_slice);
            Z_flat[batch].copy_(Z_slice);
        } catch (const std::exception&) {
            info_flat[batch].fill_(1);
        }
    }

    return std::make_tuple(S, T, alpha, beta, Q, Z, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("generalized_schur", &torchscience::cpu::linear_algebra::generalized_schur);
}
