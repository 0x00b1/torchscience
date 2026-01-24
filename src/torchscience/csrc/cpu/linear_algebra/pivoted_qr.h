// src/torchscience/csrc/cpu/linear_algebra/pivoted_qr.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> pivoted_qr(
    const at::Tensor& a
) {
    // Input validation
    TORCH_CHECK(a.dim() >= 2, "pivoted_qr: a must be at least 2D");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "pivoted_qr: a must be floating-point or complex");

    auto dtype = a.scalar_type();
    at::Tensor a_work = a.to(dtype).contiguous();

    auto batch_shape = a_work.sizes().slice(0, a_work.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t m = a_work.size(-2);
    int64_t n = a_work.size(-1);
    int64_t k = std::min(m, n);
    int64_t batch_size = a_work.numel() / (m * n);

    // Output shapes:
    // Q: (..., m, k) - orthogonal/unitary
    // R: (..., k, n) - upper triangular
    // pivots: (..., n) - column permutation indices
    // info: (...) - success indicator
    std::vector<int64_t> q_shape = batch_vec;
    q_shape.push_back(m);
    q_shape.push_back(k);

    std::vector<int64_t> r_shape = batch_vec;
    r_shape.push_back(k);
    r_shape.push_back(n);

    std::vector<int64_t> pivot_shape = batch_vec;
    pivot_shape.push_back(n);

    at::Tensor Q = at::empty(q_shape, a_work.options());
    at::Tensor R = at::empty(r_shape, a_work.options());
    at::Tensor pivots = at::empty(pivot_shape, a_work.options().dtype(at::kLong));
    at::Tensor info = at::zeros(batch_vec.empty() ? at::IntArrayRef({}) : batch_vec,
                                 a_work.options().dtype(at::kInt));

    // Flatten for batch processing
    at::Tensor a_flat = a_work.reshape({batch_size, m, n});
    at::Tensor Q_flat = Q.reshape({batch_size, m, k});
    at::Tensor R_flat = R.reshape({batch_size, k, n});
    at::Tensor pivots_flat = pivots.reshape({batch_size, n});
    at::Tensor info_flat = info.reshape({batch_size});

    // For complex matrices, use simplified approach: standard QR with column norm sorting
    bool is_complex = at::isComplexType(dtype);

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        at::Tensor a_slice = a_flat[batch];

        try {
            if (is_complex) {
                // For complex matrices, use standard QR and sort columns by norm
                // This is a simplified approximation of pivoted QR
                at::Tensor col_norms_complex = at::zeros({n}, a_work.options().dtype(at::kDouble));
                for (int64_t j = 0; j < n; ++j) {
                    col_norms_complex[j] = at::norm(a_slice.slice(1, j, j+1)).item<double>();
                }

                // Sort columns by descending norm
                auto [sorted_norms, sort_idx] = at::sort(col_norms_complex, /*dim=*/-1, /*descending=*/true);

                // Permute columns
                at::Tensor a_permuted = a_slice.index_select(1, sort_idx);

                // Compute QR of permuted matrix
                auto [Q_std, R_std] = at::linalg_qr(a_permuted, "reduced");

                Q_flat[batch].copy_(Q_std);
                R_flat[batch].copy_(R_std);
                pivots_flat[batch].copy_(sort_idx);
                continue;
            }

            // For real matrices: Implement Householder QR with column pivoting
            // At each step, pivot to the column with maximum remaining norm

            at::Tensor A = a_slice.clone();
            at::Tensor Q_acc = at::eye(m, a_work.options());
            at::Tensor perm = at::arange(n, a_work.options().dtype(at::kLong));

            // Compute initial column norms
            at::Tensor col_norms = at::zeros({n}, a_work.options().dtype(at::kDouble));
            for (int64_t j = 0; j < n; ++j) {
                col_norms[j] = at::norm(A.slice(0, 0, m).slice(1, j, j+1)).item<double>();
            }

            for (int64_t i = 0; i < k; ++i) {
                // Find the column with maximum norm among columns i:n
                // in the submatrix A[i:m, i:n]
                int64_t max_col = i;
                double max_norm = 0.0;

                for (int64_t j = i; j < n; ++j) {
                    // Compute norm of A[i:m, j]
                    double norm_val = at::norm(A.slice(0, i, m).slice(1, j, j+1)).item<double>();
                    if (norm_val > max_norm) {
                        max_norm = norm_val;
                        max_col = j;
                    }
                }

                // Swap columns i and max_col in A
                if (max_col != i) {
                    at::Tensor temp_col = A.slice(1, i, i+1).clone();
                    A.slice(1, i, i+1).copy_(A.slice(1, max_col, max_col+1));
                    A.slice(1, max_col, max_col+1).copy_(temp_col);

                    // Update permutation
                    int64_t temp_perm = perm[i].item<int64_t>();
                    perm[i] = perm[max_col];
                    perm[max_col] = temp_perm;
                }

                // Householder reflection for column i
                at::Tensor x = A.slice(0, i, m).slice(1, i, i+1).squeeze(1);
                double x_norm = at::norm(x).item<double>();

                if (x_norm > 1e-15) {
                    // v = x + sign(x[0]) * ||x|| * e_1
                    at::Tensor v = x.clone();
                    double sign_x0 = 1.0;
                    if (at::isComplexType(dtype)) {
                        auto x0 = x[0];
                        if (x0.abs().item<double>() > 1e-15) {
                            // For complex, use phase of x[0]
                            sign_x0 = 1.0; // simplified: always use +1
                        }
                    } else {
                        if (x[0].item<double>() < 0) {
                            sign_x0 = -1.0;
                        }
                    }

                    if (at::isComplexType(dtype)) {
                        v[0] = v[0] + sign_x0 * x_norm;
                    } else {
                        v[0] = v[0].item<double>() + sign_x0 * x_norm;
                    }

                    double v_norm = at::norm(v).item<double>();
                    if (v_norm > 1e-15) {
                        v = v / v_norm;

                        // H = I - 2 * v * v^H
                        // Apply H to A[i:m, i:n]: A[i:m, i:n] = A[i:m, i:n] - 2*v*(v^H * A[i:m, i:n])
                        at::Tensor v_col = v.unsqueeze(1);  // (m-i, 1)
                        at::Tensor A_sub = A.slice(0, i, m).slice(1, i, n);  // (m-i, n-i)

                        at::Tensor vH_A;
                        if (at::isComplexType(dtype)) {
                            vH_A = at::matmul(v_col.conj().transpose(0, 1), A_sub);  // (1, n-i)
                        } else {
                            vH_A = at::matmul(v_col.transpose(0, 1), A_sub);  // (1, n-i)
                        }
                        A.slice(0, i, m).slice(1, i, n).sub_(2.0 * at::matmul(v_col, vH_A));

                        // Apply H to Q_acc[0:m, i:m]: Q_acc[:, i:m] = Q_acc[:, i:m] - 2*Q_acc[:, i:m]*v*v^H
                        at::Tensor Q_sub = Q_acc.slice(1, i, m);  // (m, m-i)
                        at::Tensor Q_v;
                        if (at::isComplexType(dtype)) {
                            Q_v = at::matmul(Q_sub, v_col);  // (m, 1)
                            Q_acc.slice(1, i, m).sub_(2.0 * at::matmul(Q_v, v_col.conj().transpose(0, 1)));
                        } else {
                            Q_v = at::matmul(Q_sub, v_col);  // (m, 1)
                            Q_acc.slice(1, i, m).sub_(2.0 * at::matmul(Q_v, v_col.transpose(0, 1)));
                        }
                    }
                }
            }

            // Extract Q and R
            // Q is the first k columns of Q_acc
            Q_flat[batch].copy_(Q_acc.slice(1, 0, k));

            // R is the first k rows of A
            R_flat[batch].copy_(A.slice(0, 0, k));

            // Store pivots
            pivots_flat[batch].copy_(perm);

        } catch (const std::exception&) {
            info_flat[batch].fill_(1);
            // Fill with fallback: standard QR with identity permutation
            auto [Q_std, R_std] = at::linalg_qr(a_slice, "reduced");
            Q_flat[batch].copy_(Q_std);
            R_flat[batch].copy_(R_std);
            pivots_flat[batch].copy_(at::arange(n, a_work.options().dtype(at::kLong)));
        }
    }

    return std::make_tuple(Q, R, pivots, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("pivoted_qr", &torchscience::cpu::linear_algebra::pivoted_qr);
}
