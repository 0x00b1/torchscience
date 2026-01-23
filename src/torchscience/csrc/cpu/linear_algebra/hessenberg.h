// src/torchscience/csrc/cpu/linear_algebra/hessenberg.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> hessenberg(
    const at::Tensor& a
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(a.size(-1) == a.size(-2), "a must be square");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "a must be floating-point or complex");

    auto dtype = a.scalar_type();
    if (!at::isFloatingType(dtype) && !at::isComplexType(dtype)) {
        dtype = at::kDouble;
    }

    at::Tensor a_work = a.to(dtype).contiguous();

    auto batch_shape = a_work.sizes().slice(0, a_work.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t n = a_work.size(-1);
    int64_t batch_size = a_work.numel() / (n * n);

    std::vector<int64_t> mat_shape = batch_vec;
    mat_shape.push_back(n);
    mat_shape.push_back(n);

    at::Tensor H = at::empty(mat_shape, a_work.options());
    at::Tensor Q = at::empty(mat_shape, a_work.options());
    at::Tensor info = at::zeros(batch_vec, a_work.options().dtype(at::kInt));

    // Flatten for batch processing
    at::Tensor a_flat = a_work.reshape({batch_size, n, n});
    at::Tensor H_flat = H.reshape({batch_size, n, n});
    at::Tensor Q_flat = Q.reshape({batch_size, n, n});
    at::Tensor info_flat = info.reshape({batch_size});

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        at::Tensor a_slice = a_flat[batch].clone();

        try {
            // Compute Hessenberg form using Householder reflections
            // H = Q^H A Q, where Q is the product of Householder reflectors

            // Start with Q = I
            at::Tensor Q_accum = at::eye(n, a_slice.options());

            // Apply Householder reflections to reduce to upper Hessenberg form
            // We zero out columns below the first subdiagonal
            for (int64_t k = 0; k < n - 2; ++k) {
                // Extract the column below the subdiagonal
                at::Tensor x = a_slice.slice(0, k + 1, n).slice(1, k, k + 1).squeeze(-1);

                // Compute Householder vector v such that Hx = ||x|| e_1
                at::Scalar x_norm = x.norm();
                if (x_norm.toDouble() < 1e-15) {
                    continue;  // Skip if column is already zero
                }

                at::Tensor v = x.clone();
                // Add sign(x[0]) * ||x|| to first element to avoid cancellation
                auto x0 = x[0];
                at::Tensor sign_x0;
                if (at::isComplexType(x.scalar_type())) {
                    // For complex, use phase of x0
                    at::Tensor abs_x0 = at::abs(x0);
                    sign_x0 = at::where(abs_x0 > 1e-15, x0 / abs_x0, at::ones({}, x.options()));
                } else {
                    sign_x0 = at::where(x0 >= 0, at::ones({}, x.options()), -at::ones({}, x.options()));
                }
                v[0] = v[0] + sign_x0 * x_norm;

                at::Scalar v_norm = v.norm();
                if (v_norm.toDouble() < 1e-15) {
                    continue;
                }
                v = v / v_norm;

                // Apply Householder reflection: A := (I - 2vv^H) A (I - 2vv^H)
                // First, apply from the left to rows k+1:n
                // A[k+1:n, :] := A[k+1:n, :] - 2 v (v^H A[k+1:n, :])
                at::Tensor A_sub = a_slice.slice(0, k + 1, n);
                at::Tensor vH_A = at::matmul(v.conj().unsqueeze(0), A_sub);
                a_slice.slice(0, k + 1, n) = A_sub - 2.0 * at::matmul(v.unsqueeze(-1), vH_A);

                // Apply from the right to columns k+1:n
                // A[:, k+1:n] := A[:, k+1:n] - 2 (A[:, k+1:n] v) v^H
                at::Tensor A_sub_right = a_slice.slice(1, k + 1, n);
                at::Tensor A_v = at::matmul(A_sub_right, v.unsqueeze(-1));
                a_slice.slice(1, k + 1, n) = A_sub_right - 2.0 * at::matmul(A_v, v.conj().unsqueeze(0));

                // Accumulate Q: Q := Q (I - 2vv^H)
                // Q[:, k+1:n] := Q[:, k+1:n] - 2 (Q[:, k+1:n] v) v^H
                at::Tensor Q_sub = Q_accum.slice(1, k + 1, n);
                at::Tensor Q_v = at::matmul(Q_sub, v.unsqueeze(-1));
                Q_accum.slice(1, k + 1, n) = Q_sub - 2.0 * at::matmul(Q_v, v.conj().unsqueeze(0));
            }

            // Zero out numerical noise below the first subdiagonal
            for (int64_t i = 2; i < n; ++i) {
                for (int64_t j = 0; j < i - 1; ++j) {
                    a_slice[i][j] = 0.0;
                }
            }

            H_flat[batch].copy_(a_slice);
            Q_flat[batch].copy_(Q_accum);
        } catch (const std::exception&) {
            info_flat[batch].fill_(1);
        }
    }

    return std::make_tuple(H, Q, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("hessenberg", &torchscience::cpu::linear_algebra::hessenberg);
}
