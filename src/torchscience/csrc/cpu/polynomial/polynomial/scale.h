#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/complex.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/polynomial/polynomial_scale.h"
#include "../../../kernel/polynomial/polynomial/polynomial_scale_backward.h"
#include "../../../kernel/polynomial/polynomial/polynomial_scale_backward_backward.h"

namespace torchscience::cpu::polynomial {

// Forward: p (B, N), c (B,) or scalar -> output (B, N)
// output[k] = c * p[k]
inline at::Tensor polynomial_scale(const at::Tensor& p, const at::Tensor& c) {
    TORCH_CHECK(p.dim() >= 1, "p must have at least 1 dimension");

    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    // c should be broadcastable: either scalar (0-d) or match batch dims
    at::Tensor c_expanded;
    if (c.dim() == 0) {
        c_expanded = c.expand({B});
    } else {
        c_expanded = c.reshape({-1});
        TORCH_CHECK(c_expanded.size(0) == B,
            "c batch size must match p batch size. Got c: ", c_expanded.size(0), " vs p: ", B);
    }

    auto p_flat = p.reshape({B, N}).contiguous();
    c_expanded = c_expanded.contiguous();
    auto output = at::empty({B, N}, p.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_scale",
        [&] {
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            const scalar_t* c_ptr = c_expanded.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_scale(
                        out_ptr + b * N,
                        p_ptr + b * N,
                        c_ptr[b],
                        N
                    );
                }
            });
        }
    );
    return output;
}

// Backward: grad_output (B, N), p (B, N), c (B,) -> (grad_p (B, N), grad_c (B,))
inline std::tuple<at::Tensor, at::Tensor> polynomial_scale_backward(
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& c
) {
    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    auto grad_output_flat = grad_output.reshape({B, N}).contiguous();
    auto p_flat = p.reshape({B, N}).contiguous();

    at::Tensor c_expanded;
    if (c.dim() == 0) {
        c_expanded = c.expand({B});
    } else {
        c_expanded = c.reshape({-1});
    }
    c_expanded = c_expanded.contiguous();

    auto grad_p = at::empty({B, N}, p.options());
    auto grad_c = at::empty({B}, c.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_scale_backward",
        [&] {
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            const scalar_t* c_ptr = c_expanded.data_ptr<scalar_t>();
            scalar_t* grad_p_ptr = grad_p.data_ptr<scalar_t>();
            scalar_t* grad_c_ptr = grad_c.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_scale_backward(
                        grad_p_ptr + b * N,
                        grad_c_ptr + b,
                        grad_out_ptr + b * N,
                        p_ptr + b * N,
                        c_ptr[b],
                        N
                    );
                }
            });
        }
    );

    return {grad_p, grad_c};
}

// Second-order backward:
// gg_p (B, N), gg_c (B,), grad_output (B, N), p (B, N), c (B,)
// -> (grad_grad_output (B, N), grad_p (B, N), grad_c (B,))
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> polynomial_scale_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_c,
    const at::Tensor& grad_output,
    const at::Tensor& p,
    const at::Tensor& c
) {
    const int64_t N = p.size(-1);
    const int64_t B = p.numel() / N;

    auto grad_output_flat = grad_output.reshape({B, N}).contiguous();
    auto p_flat = p.reshape({B, N}).contiguous();

    at::Tensor c_expanded;
    if (c.dim() == 0) {
        c_expanded = c.expand({B});
    } else {
        c_expanded = c.reshape({-1});
    }
    c_expanded = c_expanded.contiguous();

    // Handle undefined gradients
    at::Tensor gg_p_flat;
    if (gg_p.defined()) {
        gg_p_flat = gg_p.reshape({B, N}).contiguous();
    } else {
        gg_p_flat = at::zeros({B, N}, p.options());
    }

    at::Tensor gg_c_expanded;
    if (gg_c.defined()) {
        if (gg_c.dim() == 0) {
            gg_c_expanded = gg_c.expand({B});
        } else {
            gg_c_expanded = gg_c.reshape({-1});
        }
        gg_c_expanded = gg_c_expanded.contiguous();
    } else {
        gg_c_expanded = at::zeros({B}, c.options());
    }

    auto grad_grad_output = at::empty({B, N}, grad_output.options());
    auto grad_p = at::empty({B, N}, p.options());
    auto grad_c = at::empty({B}, c.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_scale_backward_backward",
        [&] {
            const scalar_t* gg_p_ptr = gg_p_flat.data_ptr<scalar_t>();
            const scalar_t* gg_c_ptr = gg_c_expanded.data_ptr<scalar_t>();
            const scalar_t* grad_out_ptr = grad_output_flat.data_ptr<scalar_t>();
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            const scalar_t* c_ptr = c_expanded.data_ptr<scalar_t>();
            scalar_t* grad_grad_out_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* grad_p_ptr = grad_p.data_ptr<scalar_t>();
            scalar_t* grad_c_ptr = grad_c.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_scale_backward_backward(
                        grad_grad_out_ptr + b * N,
                        grad_p_ptr + b * N,
                        grad_c_ptr + b,
                        gg_p_ptr + b * N,
                        gg_c_ptr[b],
                        grad_out_ptr + b * N,
                        p_ptr + b * N,
                        c_ptr[b],
                        N
                    );
                }
            });
        }
    );

    return {grad_grad_output, grad_p, grad_c};
}

} // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("polynomial_scale", torchscience::cpu::polynomial::polynomial_scale);
    module.impl("polynomial_scale_backward", torchscience::cpu::polynomial::polynomial_scale_backward);
    module.impl("polynomial_scale_backward_backward", torchscience::cpu::polynomial::polynomial_scale_backward_backward);
}
