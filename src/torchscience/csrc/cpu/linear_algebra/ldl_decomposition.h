// src/torchscience/csrc/cpu/linear_algebra/ldl_decomposition.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>

namespace torchscience::cpu::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ldl_decomposition(
    const at::Tensor& a
) {
    // Input validation
    TORCH_CHECK(a.dim() >= 2, "ldl_decomposition: a must be at least 2D");
    TORCH_CHECK(a.size(-2) == a.size(-1), "ldl_decomposition: a must be square");
    TORCH_CHECK(at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type()),
        "ldl_decomposition: a must be floating-point or complex");

    auto dtype = a.scalar_type();
    at::Tensor a_work = a.to(dtype).contiguous();

    auto batch_shape = a_work.sizes().slice(0, a_work.dim() - 2);
    std::vector<int64_t> batch_vec(batch_shape.begin(), batch_shape.end());

    int64_t n = a_work.size(-1);
    int64_t batch_size = a_work.numel() / (n * n);

    // Output shapes:
    // L: (..., n, n) - unit lower triangular
    // D: (..., n, n) - diagonal matrix
    // pivots: (..., n) - pivot indices
    // info: (...) - success indicator
    std::vector<int64_t> matrix_shape = batch_vec;
    matrix_shape.push_back(n);
    matrix_shape.push_back(n);

    std::vector<int64_t> pivot_shape = batch_vec;
    pivot_shape.push_back(n);

    at::Tensor L = at::empty(matrix_shape, a_work.options());
    at::Tensor D = at::empty(matrix_shape, a_work.options());
    at::Tensor pivots = at::empty(pivot_shape, a_work.options().dtype(at::kLong));
    at::Tensor info = at::zeros(batch_vec.empty() ? at::IntArrayRef({}) : batch_vec,
                                 a_work.options().dtype(at::kInt));

    // Flatten for batch processing
    at::Tensor a_flat = a_work.reshape({batch_size, n, n});
    at::Tensor L_flat = L.reshape({batch_size, n, n});
    at::Tensor D_flat = D.reshape({batch_size, n, n});
    at::Tensor pivots_flat = pivots.reshape({batch_size, n});
    at::Tensor info_flat = info.reshape({batch_size});

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        at::Tensor a_slice = a_flat[batch];

        try {
            // Use torch.linalg.ldl_factor_ex to compute LDL decomposition
            // hermitian=true handles both real symmetric and complex Hermitian
            // Returns (LD, pivots, info) where LD is packed format
            auto result = at::linalg_ldl_factor_ex(a_slice, /*hermitian=*/true);
            at::Tensor LD = std::get<0>(result);
            at::Tensor pivots_raw = std::get<1>(result);
            at::Tensor info_raw = std::get<2>(result);

            // Extract L from packed LD format
            // L is unit lower triangular (stored below diagonal of LD)
            // The diagonal of L is implicitly 1
            at::Tensor L_extracted = at::tril(LD, -1) + at::eye(n, LD.options());

            // Extract D from packed LD format
            // D is stored on the diagonal of LD
            at::Tensor D_extracted = at::diag_embed(at::diagonal(LD, 0, -2, -1));

            L_flat[batch].copy_(L_extracted);
            D_flat[batch].copy_(D_extracted);
            pivots_flat[batch].copy_(pivots_raw);

            // Copy info from factor result
            info_flat[batch].fill_(info_raw.item<int>());
        } catch (const std::exception&) {
            info_flat[batch].fill_(1);
            // Initialize outputs to zeros on failure
            L_flat[batch].zero_();
            D_flat[batch].zero_();
            pivots_flat[batch].zero_();
        }
    }

    return std::make_tuple(L, D, pivots, info);
}

}  // namespace torchscience::cpu::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("ldl_decomposition", &torchscience::cpu::linear_algebra::ldl_decomposition);
}
