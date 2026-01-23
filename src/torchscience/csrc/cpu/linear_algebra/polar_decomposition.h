// src/torchscience/csrc/cpu/linear_algebra/polar_decomposition.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> polar_decomposition(
    const at::Tensor& a,
    c10::string_view side
) {
    TORCH_CHECK(a.dim() >= 2, "a must be at least 2D");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "a must be floating-point or complex");
    TORCH_CHECK(side == "right" || side == "left",
        "side must be 'right' or 'left', got '", std::string(side), "'");

    auto dtype = a.scalar_type();
    at::Tensor a_work = a.to(dtype).contiguous();

    auto batch_shape = a_work.sizes().slice(0, a_work.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t m = a_work.size(-2);
    int64_t n = a_work.size(-1);
    int64_t batch_size = a_work.numel() / (m * n);

    bool is_right = (side == "right");

    // Output shapes:
    // U: same shape as input (..., m, n)
    // P: (..., n, n) for right polar, (..., m, m) for left polar
    std::vector<int64_t> u_shape = batch_vec;
    u_shape.push_back(m);
    u_shape.push_back(n);

    std::vector<int64_t> p_shape = batch_vec;
    if (is_right) {
        p_shape.push_back(n);
        p_shape.push_back(n);
    } else {
        p_shape.push_back(m);
        p_shape.push_back(m);
    }

    at::Tensor U = at::empty(u_shape, a_work.options());
    at::Tensor P = at::empty(p_shape, a_work.options());
    at::Tensor info = at::zeros(batch_vec, a_work.options().dtype(at::kInt));

    // Flatten for batch processing
    at::Tensor a_flat = a_work.reshape({batch_size, m, n});
    at::Tensor U_flat = U.reshape({batch_size, m, n});
    at::Tensor P_flat = is_right
        ? P.reshape({batch_size, n, n})
        : P.reshape({batch_size, m, m});
    at::Tensor info_flat = info.reshape({batch_size});

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        at::Tensor a_slice = a_flat[batch];

        try {
            // Compute SVD: A = U_svd @ diag(S) @ Vh
            auto [U_svd, S, Vh] = at::linalg_svd(a_slice, /*full_matrices=*/false);

            // V = Vh.mH (conjugate transpose of Vh)
            at::Tensor V = Vh.mH();

            // Unitary factor: U = U_svd @ Vh
            at::Tensor U_result = at::matmul(U_svd, Vh);

            // Positive semidefinite factor P
            at::Tensor P_result;
            if (is_right) {
                // Right polar: A = UP, so P = V @ diag(S) @ V.H
                P_result = at::matmul(V, at::matmul(at::diag_embed(S), Vh));
            } else {
                // Left polar: A = PU, so P = U_svd @ diag(S) @ U_svd.H
                P_result = at::matmul(U_svd, at::matmul(at::diag_embed(S), U_svd.mH()));
            }

            U_flat[batch].copy_(U_result);
            P_flat[batch].copy_(P_result);
        } catch (const std::exception&) {
            info_flat[batch].fill_(1);
        }
    }

    return std::make_tuple(U, P, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("polar_decomposition", &torchscience::cpu::linear_algebra::polar_decomposition);
}
