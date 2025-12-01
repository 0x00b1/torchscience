#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::cpu::optimization::combinatorial {

inline at::Tensor sinkhorn(
    const at::Tensor& C,
    const at::Tensor& a,
    const at::Tensor& b,
    double epsilon,
    int64_t maxiter,
    double tol
) {
    TORCH_CHECK(C.dim() >= 2, "Cost matrix must have at least 2 dimensions");
    TORCH_CHECK(a.dim() >= 1, "Source marginal must have at least 1 dimension");
    TORCH_CHECK(b.dim() >= 1, "Target marginal must have at least 1 dimension");
    TORCH_CHECK(epsilon > 0, "Regularization epsilon must be positive");

    // Compute kernel matrix K = exp(-C / epsilon)
    at::Tensor K = at::exp(-C / epsilon);

    // Initialize scaling vectors
    at::Tensor u = at::ones_like(a);
    at::Tensor v = at::ones_like(b);

    for (int64_t iter = 0; iter < maxiter; ++iter) {
        at::Tensor u_prev = u.clone();

        // v = b / (K^T @ u)
        // K^T @ u: (..., m, n) @ (..., n) -> (..., m)
        at::Tensor Ktu = at::matmul(
            K.transpose(-2, -1),
            u.unsqueeze(-1)
        ).squeeze(-1);
        v = b / at::clamp_min(Ktu, 1e-10);

        // u = a / (K @ v)
        // K @ v: (..., n, m) @ (..., m) -> (..., n)
        at::Tensor Kv = at::matmul(K, v.unsqueeze(-1)).squeeze(-1);
        u = a / at::clamp_min(Kv, 1e-10);

        // Check convergence
        double max_diff = at::max(at::abs(u - u_prev)).item<double>();
        if (max_diff < tol) {
            break;
        }
    }

    // Transport plan: P = diag(u) @ K @ diag(v)
    // Equivalent to: P_ij = u_i * K_ij * v_j
    at::Tensor P = u.unsqueeze(-1) * K * v.unsqueeze(-2);

    return P;
}

inline at::Tensor sinkhorn_backward(
    const at::Tensor& grad_output,
    const at::Tensor& P,
    const at::Tensor& C,
    double epsilon
) {
    // Gradient w.r.t. C:
    // P = diag(u) @ K @ diag(v), where K = exp(-C/epsilon)
    // dP/dC = -P / epsilon (element-wise, since dK/dC = -K/epsilon)
    // dL/dC = dL/dP * dP/dC = grad_output * (-P / epsilon)
    return -grad_output * P / epsilon;
}

}  // namespace torchscience::cpu::optimization::combinatorial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "sinkhorn",
        &torchscience::cpu::optimization::combinatorial::sinkhorn
    );
    module.impl(
        "sinkhorn_backward",
        &torchscience::cpu::optimization::combinatorial::sinkhorn_backward
    );
}
