#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/complex.h>
#include <torch/library.h>

#include "../../../kernel/polynomial/polynomial/polynomial_divmod.h"
#include "../../../kernel/polynomial/polynomial/polynomial_divmod_backward.h"
#include "../../../kernel/polynomial/polynomial/polynomial_divmod_backward_backward.h"

namespace torchscience::cpu::polynomial {

// Forward: p (B, N), q (B, M) -> (quotient (B, N-M+1), remainder (B, max(M-1, 1)))
// Divides polynomials using long division
// Returns (quotient, remainder) such that p = q * quotient + remainder
// Batch dimensions must match (Python handles broadcasting)
inline std::tuple<at::Tensor, at::Tensor> polynomial_divmod(
    const at::Tensor& p,
    const at::Tensor& q
) {
    TORCH_CHECK(p.dim() >= 1, "p must have at least 1 dimension");
    TORCH_CHECK(q.dim() >= 1, "q must have at least 1 dimension");

    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);

    TORCH_CHECK(N >= M, "Dividend degree must be >= divisor degree. Got N=", N, ", M=", M);
    TORCH_CHECK(M >= 1, "Divisor must have at least 1 coefficient");

    const int64_t B_p = p.numel() / N;
    const int64_t B_q = q.numel() / M;
    TORCH_CHECK(B_p == B_q, "Batch dimensions must match. Got ", B_p, " vs ", B_q);
    const int64_t B = B_p;

    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;

    auto p_flat = p.reshape({B, N}).contiguous();
    auto q_flat = q.reshape({B, M}).contiguous();

    auto quotient = at::empty({B, quot_len}, p.options());
    auto remainder = at::empty({B, rem_len}, p.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_divmod",
        [&] {
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_flat.data_ptr<scalar_t>();
            scalar_t* quot_ptr = quotient.data_ptr<scalar_t>();
            scalar_t* rem_ptr = remainder.data_ptr<scalar_t>();

            // Need work buffer for each batch element
            auto work = at::empty({B, N}, p.options());
            scalar_t* work_ptr = work.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_divmod_with_buffer(
                        quot_ptr + b * quot_len,
                        rem_ptr + b * rem_len,
                        work_ptr + b * N,
                        p_ptr + b * N,
                        q_ptr + b * M,
                        N,
                        M
                    );
                }
            });
        }
    );

    return {quotient, remainder};
}

// Backward: grad_Q (B, N-M+1), grad_R (B, rem_len), Q (B, N-M+1), p (B, N), q (B, M)
//        -> (grad_p (B, N), grad_q (B, M))
inline std::tuple<at::Tensor, at::Tensor> polynomial_divmod_backward(
    const at::Tensor& grad_Q,
    const at::Tensor& grad_R,
    const at::Tensor& Q,
    const at::Tensor& p,
    const at::Tensor& q
) {
    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t B = p.numel() / N;
    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;

    auto p_flat = p.reshape({B, N}).contiguous();
    auto q_flat = q.reshape({B, M}).contiguous();
    auto Q_flat = Q.reshape({B, quot_len}).contiguous();
    auto grad_Q_flat = grad_Q.reshape({B, quot_len}).contiguous();
    auto grad_R_flat = grad_R.reshape({B, rem_len}).contiguous();

    auto grad_p = at::zeros({B, N}, p.options());
    auto grad_q = at::zeros({B, M}, q.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_divmod_backward",
        [&] {
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_flat.data_ptr<scalar_t>();
            const scalar_t* Q_ptr = Q_flat.data_ptr<scalar_t>();
            const scalar_t* grad_Q_ptr = grad_Q_flat.data_ptr<scalar_t>();
            const scalar_t* grad_R_ptr = grad_R_flat.data_ptr<scalar_t>();
            scalar_t* grad_p_ptr = grad_p.data_ptr<scalar_t>();
            scalar_t* grad_q_ptr = grad_q.data_ptr<scalar_t>();

            auto work = at::empty({B, N}, p.options());
            scalar_t* work_ptr = work.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_divmod_backward(
                        grad_p_ptr + b * N,
                        grad_q_ptr + b * M,
                        grad_Q_ptr + b * quot_len,
                        grad_R_ptr + b * rem_len,
                        Q_ptr + b * quot_len,
                        p_ptr + b * N,
                        q_ptr + b * M,
                        N,
                        M,
                        work_ptr + b * N
                    );
                }
            });
        }
    );

    return {grad_p, grad_q};
}

// Second-order backward
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
polynomial_divmod_backward_backward(
    const at::Tensor& gg_p,
    const at::Tensor& gg_q,
    const at::Tensor& grad_Q,
    const at::Tensor& grad_R,
    const at::Tensor& Q,
    const at::Tensor& p,
    const at::Tensor& q
) {
    const int64_t N = p.size(-1);
    const int64_t M = q.size(-1);
    const int64_t B = p.numel() / N;
    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;

    auto p_flat = p.reshape({B, N}).contiguous();
    auto q_flat = q.reshape({B, M}).contiguous();
    auto Q_flat = Q.reshape({B, quot_len}).contiguous();
    auto grad_Q_flat = grad_Q.reshape({B, quot_len}).contiguous();
    auto grad_R_flat = grad_R.reshape({B, rem_len}).contiguous();
    auto gg_p_flat = gg_p.defined() ? gg_p.reshape({B, N}).contiguous() : at::zeros({B, N}, p.options());
    auto gg_q_flat = gg_q.defined() ? gg_q.reshape({B, M}).contiguous() : at::zeros({B, M}, q.options());

    auto grad_grad_Q = at::zeros({B, quot_len}, grad_Q.options());
    auto grad_grad_R = at::zeros({B, rem_len}, grad_R.options());
    auto grad_Q_out = at::zeros({B, quot_len}, Q.options());
    auto grad_p_out = at::zeros({B, N}, p.options());
    auto grad_q_out = at::zeros({B, M}, q.options());

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        p.scalar_type(),
        "polynomial_divmod_backward_backward",
        [&] {
            const scalar_t* p_ptr = p_flat.data_ptr<scalar_t>();
            const scalar_t* q_ptr = q_flat.data_ptr<scalar_t>();
            const scalar_t* Q_ptr = Q_flat.data_ptr<scalar_t>();
            const scalar_t* grad_Q_ptr = grad_Q_flat.data_ptr<scalar_t>();
            const scalar_t* grad_R_ptr = grad_R_flat.data_ptr<scalar_t>();
            const scalar_t* gg_p_ptr = gg_p_flat.data_ptr<scalar_t>();
            const scalar_t* gg_q_ptr = gg_q_flat.data_ptr<scalar_t>();

            scalar_t* ggQ_ptr = grad_grad_Q.data_ptr<scalar_t>();
            scalar_t* ggR_ptr = grad_grad_R.data_ptr<scalar_t>();
            scalar_t* gQ_ptr = grad_Q_out.data_ptr<scalar_t>();
            scalar_t* gp_ptr = grad_p_out.data_ptr<scalar_t>();
            scalar_t* gq_ptr = grad_q_out.data_ptr<scalar_t>();

            auto work1 = at::empty({B, N}, p.options());
            auto work2 = at::empty({B, N}, p.options());
            scalar_t* work1_ptr = work1.data_ptr<scalar_t>();
            scalar_t* work2_ptr = work2.data_ptr<scalar_t>();

            at::parallel_for(0, B, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    kernel::polynomial::polynomial_divmod_backward_backward(
                        ggQ_ptr + b * quot_len,
                        ggR_ptr + b * rem_len,
                        gQ_ptr + b * quot_len,
                        gp_ptr + b * N,
                        gq_ptr + b * M,
                        gg_p_ptr + b * N,
                        gg_q_ptr + b * M,
                        grad_Q_ptr + b * quot_len,
                        grad_R_ptr + b * rem_len,
                        Q_ptr + b * quot_len,
                        p_ptr + b * N,
                        q_ptr + b * M,
                        N,
                        M,
                        work1_ptr + b * N,
                        work2_ptr + b * N
                    );
                }
            });
        }
    );

    return {grad_grad_Q, grad_grad_R, grad_Q_out, grad_p_out, grad_q_out};
}

} // namespace torchscience::cpu::polynomial

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("polynomial_divmod", torchscience::cpu::polynomial::polynomial_divmod);
    module.impl("polynomial_divmod_backward", torchscience::cpu::polynomial::polynomial_divmod_backward);
    module.impl("polynomial_divmod_backward_backward", torchscience::cpu::polynomial::polynomial_divmod_backward_backward);
}
