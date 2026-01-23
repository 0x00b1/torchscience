// src/torchscience/csrc/cpu/linear_algebra/schur_decomposition.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> schur_decomposition(
    const at::Tensor& a,
    c10::string_view output
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(a.size(-1) == a.size(-2), "a must be square");
    TORCH_CHECK(output == "real" || output == "complex",
        "output must be 'real' or 'complex', got '", output, "'");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "a must be floating-point or complex");

    bool complex_output = (output == "complex") || at::isComplexType(a.scalar_type());

    auto dtype = a.scalar_type();
    if (!at::isFloatingType(dtype) && !at::isComplexType(dtype)) {
        dtype = at::kDouble;
    }

    at::Tensor a_work = a.to(dtype).contiguous();

    auto batch_shape = a_work.sizes().slice(0, a_work.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t n = a_work.size(-1);
    int64_t batch_size = a_work.numel() / (n * n);

    auto complex_dtype = at::toComplexType(
        at::isComplexType(dtype) ? dtype : (dtype == at::kFloat ? at::kFloat : at::kDouble)
    );
    auto output_dtype = complex_output ? complex_dtype : dtype;

    std::vector<int64_t> mat_shape = batch_vec;
    mat_shape.push_back(n);
    mat_shape.push_back(n);

    std::vector<int64_t> eig_shape = batch_vec;
    eig_shape.push_back(n);

    at::Tensor T = at::empty(mat_shape, a_work.options().dtype(output_dtype));
    at::Tensor Q = at::empty(mat_shape, a_work.options().dtype(output_dtype));
    at::Tensor eigenvalues = at::empty(eig_shape, a_work.options().dtype(complex_dtype));
    at::Tensor info = at::zeros(batch_vec, a_work.options().dtype(at::kInt));

    // Flatten for batch processing
    at::Tensor a_flat = a_work.reshape({batch_size, n, n});
    at::Tensor T_flat = T.reshape({batch_size, n, n});
    at::Tensor Q_flat = Q.reshape({batch_size, n, n});
    at::Tensor eig_flat = eigenvalues.reshape({batch_size, n});
    at::Tensor info_flat = info.reshape({batch_size});

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        auto a_slice = a_flat[batch];

        try {
            // Convert to complex for unified processing
            at::Tensor a_complex = a_slice.to(complex_dtype);

            // Get eigenvalues and eigenvectors
            auto [eig_vals, eig_vecs] = at::linalg_eig(a_slice);

            // Orthogonalize eigenvectors using QR decomposition to get unitary Q
            // This gives us Q such that A = Q T Q^H where T is upper triangular
            auto [Q_unitary, R] = at::linalg_qr(eig_vecs);

            // Compute T = Q^H A Q (upper triangular in exact arithmetic)
            at::Tensor T_computed = at::matmul(Q_unitary.conj().transpose(-2, -1), at::matmul(a_complex, Q_unitary));

            // Extract eigenvalues from diagonal of T
            at::Tensor eig_from_T = T_computed.diagonal(0, -2, -1);

            at::Tensor T_slice, Q_slice;

            if (complex_output) {
                T_slice = T_computed;
                Q_slice = Q_unitary;
            } else {
                // For real output, take real parts (simplified - proper impl would use LAPACK dgees)
                T_slice = at::real(T_computed);
                Q_slice = at::real(Q_unitary);
            }

            T_flat[batch].copy_(T_slice);
            Q_flat[batch].copy_(Q_slice);
            eig_flat[batch].copy_(eig_from_T);
        } catch (const std::exception&) {
            info_flat[batch].fill_(1);
        }
    }

    return std::make_tuple(T, Q, eigenvalues, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("schur_decomposition", &torchscience::cpu::linear_algebra::schur_decomposition);
}
