#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for polynomial subtraction
// For output = p - q:
//   grad_p[k] = grad_output[k] for k < N
//   grad_q[k] = -grad_output[k] for k < M  (note the negation)
//
// Parameters:
//   grad_p: output gradient w.r.t. p, size N
//   grad_q: output gradient w.r.t. q, size M
//   grad_output: upstream gradient, size max(N, M)
//   N, M: coefficient counts
template <typename T>
void polynomial_subtract_backward(
    T* grad_p,
    T* grad_q,
    const T* grad_output,
    int64_t N,
    int64_t M
) {
    for (int64_t k = 0; k < N; ++k) {
        grad_p[k] = grad_output[k];
    }
    for (int64_t k = 0; k < M; ++k) {
        grad_q[k] = -grad_output[k];  // Negation for subtraction
    }
}

// Complex specialization
template <typename T>
void polynomial_subtract_backward(
    c10::complex<T>* grad_p,
    c10::complex<T>* grad_q,
    const c10::complex<T>* grad_output,
    int64_t N,
    int64_t M
) {
    for (int64_t k = 0; k < N; ++k) {
        grad_p[k] = grad_output[k];
    }
    for (int64_t k = 0; k < M; ++k) {
        grad_q[k] = -grad_output[k];  // Negation for subtraction
    }
}

} // namespace torchscience::kernel::polynomial
