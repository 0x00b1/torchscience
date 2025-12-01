#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for polynomial addition
//
// backward(grad_output, p, q) returns:
//   grad_p = grad_output[:N]  (slice)
//   grad_q = grad_output[:M]  (slice)
//
// backward_backward needs to compute gradients w.r.t. the backward inputs:
//   grad_grad_output[k] = gg_p[k] if k < N else 0
//                       + gg_q[k] if k < M else 0
//   grad_p = 0 (backward doesn't depend on p values)
//   grad_q = 0 (backward doesn't depend on q values)
//
// Parameters:
//   grad_grad_output: output, size K = max(N, M)
//   grad_p: output, size N
//   grad_q: output, size M
//   gg_p: upstream gradient for grad_p, size N
//   gg_q: upstream gradient for grad_q, size M
//   N, M, K: sizes
template <typename T>
void polynomial_add_backward_backward(
    T* grad_grad_output,
    T* grad_p,
    T* grad_q,
    const T* gg_p,
    const T* gg_q,
    int64_t N,
    int64_t M,
    int64_t K
) {
    // grad_grad_output accumulates contributions from both gg_p and gg_q
    for (int64_t k = 0; k < K; ++k) {
        T val = T(0);
        if (k < N) val += gg_p[k];
        if (k < M) val += gg_q[k];
        grad_grad_output[k] = val;
    }
    // grad_p and grad_q are zero (backward is linear in grad_output, not in p/q)
    for (int64_t k = 0; k < N; ++k) {
        grad_p[k] = T(0);
    }
    for (int64_t k = 0; k < M; ++k) {
        grad_q[k] = T(0);
    }
}

// Complex specialization
template <typename T>
void polynomial_add_backward_backward(
    c10::complex<T>* grad_grad_output,
    c10::complex<T>* grad_p,
    c10::complex<T>* grad_q,
    const c10::complex<T>* gg_p,
    const c10::complex<T>* gg_q,
    int64_t N,
    int64_t M,
    int64_t K
) {
    for (int64_t k = 0; k < K; ++k) {
        c10::complex<T> val(T(0), T(0));
        if (k < N) val += gg_p[k];
        if (k < M) val += gg_q[k];
        grad_grad_output[k] = val;
    }
    for (int64_t k = 0; k < N; ++k) {
        grad_p[k] = c10::complex<T>(T(0), T(0));
    }
    for (int64_t k = 0; k < M; ++k) {
        grad_q[k] = c10::complex<T>(T(0), T(0));
    }
}

} // namespace torchscience::kernel::polynomial
