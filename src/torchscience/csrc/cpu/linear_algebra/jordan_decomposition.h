// src/torchscience/csrc/cpu/linear_algebra/jordan_decomposition.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> jordan_decomposition(
    const at::Tensor& a
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(a.size(-1) == a.size(-2), "a must be square");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "a must be floating-point or complex");

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

    at::Tensor a_work = a.to(complex_dtype).contiguous();

    auto batch_shape = a_work.sizes().slice(0, a_work.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t n = a_work.size(-1);
    int64_t batch_size = a_work.numel() / (n * n);

    std::vector<int64_t> mat_shape = batch_vec;
    mat_shape.push_back(n);
    mat_shape.push_back(n);

    at::Tensor J = at::zeros(mat_shape, a_work.options().dtype(complex_dtype));
    at::Tensor P = at::zeros(mat_shape, a_work.options().dtype(complex_dtype));
    at::Tensor info = at::zeros(batch_vec, a_work.options().dtype(at::kInt));

    // Flatten for batch processing
    at::Tensor a_flat = a_work.reshape({batch_size, n, n});
    at::Tensor J_flat = J.reshape({batch_size, n, n});
    at::Tensor P_flat = P.reshape({batch_size, n, n});
    at::Tensor info_flat = info.reshape({batch_size});

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        at::Tensor a_slice = a_flat[batch];

        try {
            // Compute eigenvalues and eigenvectors using torch.linalg.eig
            // This gives us eigenvalues and right eigenvectors
            auto [eigenvalues, eigenvectors] = at::linalg_eig(a_slice);

            // For generic (diagonalizable) matrices, the Jordan form is simply
            // the diagonal matrix of eigenvalues, and P is the eigenvector matrix.
            //
            // For defective matrices (non-diagonalizable), we would need to compute
            // generalized eigenvectors to form Jordan chains. This is a simplified
            // implementation that handles the generic case.
            //
            // The Jordan normal form J = diag(lambda_1, ..., lambda_n) for diagonalizable
            // matrices, with Jordan blocks for repeated eigenvalues in defective matrices.

            // Construct J as diagonal of eigenvalues
            // For the simplified case (diagonalizable), J = diag(eigenvalues)
            J_flat[batch] = at::diag(eigenvalues);

            // P is the matrix of eigenvectors
            P_flat[batch] = eigenvectors;

            // Verify the decomposition quality by checking if P is invertible
            // A singular P indicates the matrix may be defective
            double tol = (dtype == at::kFloat || dtype == at::kComplexFloat) ? 1e-5 : 1e-12;
            at::Tensor cond = at::linalg_cond(eigenvectors);
            if (cond.item<double>() > 1.0 / tol) {
                // Matrix may be nearly defective, set info to indicate this
                info_flat[batch].fill_(1);
            }

        } catch (const std::exception&) {
            info_flat[batch].fill_(-1);  // General failure
        }
    }

    return std::make_tuple(J, P, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("jordan_decomposition", &torchscience::cpu::linear_algebra::jordan_decomposition);
}
