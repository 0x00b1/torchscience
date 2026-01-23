// src/torchscience/csrc/cpu/linear_algebra/rank_revealing_qr.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> rank_revealing_qr(
    const at::Tensor& a,
    double tol
) {
    // Input validation
    TORCH_CHECK(a.dim() >= 2, "rank_revealing_qr: a must be at least 2D");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "rank_revealing_qr: a must be floating-point or complex");
    TORCH_CHECK(tol >= 0, "rank_revealing_qr: tolerance must be non-negative");

    // Use pivoted_qr internally (call the dispatcher)
    auto pivoted_result = at::_ops::torchscience_pivoted_qr::call(a);
    at::Tensor Q = std::get<0>(pivoted_result);
    at::Tensor R = std::get<1>(pivoted_result);
    at::Tensor pivots = std::get<2>(pivoted_result);
    at::Tensor info = std::get<3>(pivoted_result);

    // Determine numerical rank from diagonal of R
    // rank = count of |R[i,i]| > tol * |R[0,0]|
    auto batch_shape = a.sizes().slice(0, a.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t m = a.size(-2);
    int64_t n = a.size(-1);
    int64_t k = std::min(m, n);

    // Extract diagonal elements of R: shape (..., k)
    at::Tensor R_diag = at::diagonal(R, 0, -2, -1);  // (..., k)
    at::Tensor R_diag_abs = at::abs(R_diag);  // (..., k)

    // Get the first diagonal element (reference for scaling)
    at::Tensor R00_abs = R_diag_abs.narrow(-1, 0, 1);  // (..., 1)

    // Compute threshold: tol * |R[0,0]|
    at::Tensor threshold = tol * R00_abs;  // (..., 1)

    // Count elements > threshold along the last dimension
    // rank = sum of (|R_diag| > threshold)
    at::Tensor above_threshold = R_diag_abs > threshold;  // (..., k) boolean
    at::Tensor rank = at::sum(above_threshold.to(at::kLong), -1);  // (...)

    // Handle the case where R[0,0] is zero (entire matrix is essentially zero)
    // In this case, threshold would be 0 and all elements would pass
    // We need to check if R[0,0] itself is below machine epsilon
    at::Tensor R00_is_zero = R00_abs.squeeze(-1) < 1e-15;  // (...)
    rank = at::where(R00_is_zero, at::zeros_like(rank), rank);

    return std::make_tuple(Q, R, pivots, rank, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("rank_revealing_qr", &torchscience::cpu::linear_algebra::rank_revealing_qr);
}
