#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/complex.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/polynomial/polynomial_add.h"
#include "../../../kernel/polynomial/polynomial/polynomial_add_backward.h"
#include "../../../kernel/polynomial/polynomial/polynomial_add_backward_backward.h"

namespace torchscience::cpu::polynomial {

// Forward: p (B, N), q (B, M) -> output (B, max(N, M))
inline at::Tensor polynomial_add(const at::Tensor& p, const at::Tensor& q) {
    TORCH_CHECK(p.dim() >= 1, "p must have at least 1 dimension");
    TORCH_CHECK(q.dim() >= 1, "q must have at least 1 dimension");

    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t K = std::max(N, M);

    const int64_t B_p = p.numel() / N;
    const int64_t B_q = q.numel() / M;
    TORCH_CHECK(B_p == B_q, "Batch dimensions must match. Got ", B_p, " vs ", B_q);
    const int64_t B = B_p;

    auto p_flat = p.reshape({B, N}).contiguous();
    auto q_flat = q.reshape({B, M}).contiguous();
    auto output = at::empty({B, K}, p.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_add",
        [&] {
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_flat.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_add(
                        out_ptr + b * K,
                        p_ptr + b * N,
                        q_ptr + b * M,
                        N, M
                    );
                }
            });
        }
    );
    return output;
}

// Backward: grad_output (B, K), p (B, N), q (B, M) -> (grad_p (B, N), grad_q (B, M))
inline std::tuple<at::Tensor, at::Tensor> polynomial_add_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q
) {
    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t K = std::max(N, M);
    const int64_t B = p.numel() / N;

    auto grad_output_flat = grad_output.reshape({B, K}).contiguous();

    // grad_p is first N elements, grad_q is first M elements
    auto grad_p = grad_output_flat.slice(-1, 0, N).contiguous();
    auto grad_q = grad_output_flat.slice(-1, 0, M).contiguous();

    return {grad_p, grad_q};
}

// Second-order backward: grad_grad_output accumulates gg_p and gg_q contributions
// grad_p and grad_q are zero since backward is linear in grad_output, not in p/q
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> polynomial_add_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& q
) {
    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t K = std::max(N, M);
    const int64_t B = p.numel() / N;

    // grad_grad_output starts as zeros, then we add contributions
    auto grad_grad_output = at::zeros({B, K}, grad_output.options());

    // gg_p contributes to first N elements
    if (gg_p.defined()) {
        grad_grad_output.slice(-1, 0, N).add_(gg_p.reshape({B, N}));
    }

    // gg_q contributes to first M elements
    if (gg_q.defined()) {
        grad_grad_output.slice(-1, 0, M).add_(gg_q.reshape({B, M}));
    }

    return {
        grad_grad_output,
        at::zeros_like(p),
        at::zeros_like(q)
    };
}

} // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("polynomial_add", torchscience::cpu::polynomial::polynomial_add);
    module.impl("polynomial_add_backward", torchscience::cpu::polynomial::polynomial_add_backward);
    module.impl("polynomial_add_backward_backward", torchscience::cpu::polynomial::polynomial_add_backward_backward);
}
