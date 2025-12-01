#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/complex.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/polynomial/polynomial_negate.h"
#include "../../../kernel/polynomial/polynomial/polynomial_negate_backward.h"
#include "../../../kernel/polynomial/polynomial/polynomial_negate_backward_backward.h"

namespace torchscience::cpu::polynomial {

// Forward: p (B, N) -> output (B, N)
inline at::Tensor polynomial_negate(const at::Tensor& p) {
    TORCH_CHECK(p.dim() >= 1, "p must have at least 1 dimension");

    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    auto p_flat = p.reshape({B, N}).contiguous();
    auto output = at::empty({B, N}, p.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_negate",
        [&] {
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_negate(
                        out_ptr + b * N,
                        p_ptr + b * N,
                        N
                    );
                }
            });
        }
    );
    return output;
}

// Backward: grad_output (B, N), p (B, N) -> grad_p (B, N)
inline at::Tensor polynomial_negate_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p
) {
    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    auto grad_output_flat = grad_output.reshape({B, N}).contiguous();
    auto grad_p = at::empty({B, N}, p.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_negate_backward",
        [&] {
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            scalar_t* grad_p_ptr = grad_p.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_negate_backward(
                        grad_p_ptr + b * N,
                        grad_out_ptr + b * N,
                        N
                    );
                }
            });
        }
    );
    return grad_p;
}

// Second-order backward: gg_p (B, N), grad_output (B, N), p (B, N) -> (grad_grad_output, grad_p)
inline std::tuple<at::Tensor, at::Tensor> polynomial_negate_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& grad_output,
    const at::Tensor& p
) {
    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    // grad_grad_output = -gg_p
    auto grad_grad_output = at::empty({B, N}, grad_output.options());
    if (gg_p.defined()) {
        grad_grad_output = -gg_p.reshape({B, N});
    } else {
        grad_grad_output.zero_();
    }

    // grad_p = 0 (backward doesn't depend on p values)
    auto grad_p = at::zeros_like(p);

    return {grad_grad_output, grad_p};
}

} // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("polynomial_negate", torchscience::cpu::polynomial::polynomial_negate);
    module.impl("polynomial_negate_backward", torchscience::cpu::polynomial::polynomial_negate_backward);
    module.impl("polynomial_negate_backward_backward", torchscience::cpu::polynomial::polynomial_negate_backward_backward);
}
