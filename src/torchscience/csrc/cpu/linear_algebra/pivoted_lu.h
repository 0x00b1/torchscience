// src/torchscience/csrc/cpu/linear_algebra/pivoted_lu.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> pivoted_lu(
    const at::Tensor& a
) {
    // Input validation
    TORCH_CHECK(a.dim() >= 2, "pivoted_lu: a must be at least 2D");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "pivoted_lu: a must be floating-point or complex");

    auto dtype = a.scalar_type();
    at::Tensor a_work = a.to(dtype).contiguous();

    auto batch_shape = a_work.sizes().slice(0, a_work.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t m = a_work.size(-2);
    int64_t n = a_work.size(-1);
    int64_t k = std::min(m, n);
    int64_t batch_size = a_work.numel() / (m * n);

    // Output shapes:
    // L: (..., m, k) - unit lower triangular
    // U: (..., k, n) - upper triangular
    // pivots: (..., m) - pivot indices (row permutation for all m rows)
    // info: (...) - success indicator
    std::vector<int64_t> l_shape = batch_vec;
    l_shape.push_back(m);
    l_shape.push_back(k);

    std::vector<int64_t> u_shape = batch_vec;
    u_shape.push_back(k);
    u_shape.push_back(n);

    std::vector<int64_t> pivot_shape = batch_vec;
    pivot_shape.push_back(m);

    at::Tensor L = at::empty(l_shape, a_work.options());
    at::Tensor U = at::empty(u_shape, a_work.options());
    at::Tensor pivots = at::empty(pivot_shape, a_work.options().dtype(at::kLong));
    at::Tensor info = at::zeros(batch_vec.empty() ? at::IntArrayRef({}) : batch_vec,
                                 a_work.options().dtype(at::kInt));

    // Flatten for batch processing
    at::Tensor a_flat = a_work.reshape({batch_size, m, n});
    at::Tensor L_flat = L.reshape({batch_size, m, k});
    at::Tensor U_flat = U.reshape({batch_size, k, n});
    at::Tensor pivots_flat = pivots.reshape({batch_size, m});
    at::Tensor info_flat = info.reshape({batch_size});

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        at::Tensor a_slice = a_flat[batch];

        try {
            // Use torch.linalg.lu to get P, L, U
            // at::linalg_lu returns (P, L, U) where P is a permutation matrix
            auto [P, L_full, U_full] = at::linalg_lu(a_slice);

            // Extract L[:, :k] and U[:k, :]
            L_flat[batch].copy_(L_full.slice(/*dim=*/1, /*start=*/0, /*end=*/k));
            U_flat[batch].copy_(U_full.slice(/*dim=*/0, /*start=*/0, /*end=*/k));

            // Convert P matrix to pivot indices
            // P is a permutation matrix where P[i, pivots[i]] = 1
            // We get pivots[i] from argmax of each row of P
            // Note: For complex input, P is complex but values are real (0 or 1)
            // argmax doesn't support complex, so convert to real first
            at::Tensor P_real = at::isComplexType(P.scalar_type()) ? at::real(P) : P;
            at::Tensor pivot_indices = at::argmax(P_real, /*dim=*/-1);
            pivots_flat[batch].copy_(pivot_indices);

        } catch (const std::exception&) {
            info_flat[batch].fill_(1);
        }
    }

    return std::make_tuple(L, U, pivots, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("pivoted_lu", &torchscience::cpu::linear_algebra::pivoted_lu);
}
