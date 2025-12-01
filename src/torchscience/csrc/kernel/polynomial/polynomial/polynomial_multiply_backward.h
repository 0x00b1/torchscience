#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for polynomial multiplication
//
// Forward: y[k] = sum_{i+j=k} p[i] * q[j]
//
// Backward (correlation):
//   grad_p[i] = sum_k grad_output[k] * q[k-i]  for k in [i, i+M-1]
//   grad_q[j] = sum_k grad_output[k] * p[k-j]  for k in [j, j+N-1]
//
// Parameters:
//   grad_p: output gradient w.r.t. p, size N
//   grad_q: output gradient w.r.t. q, size M
//   grad_output: upstream gradient, size N + M - 1
//   p: first polynomial coefficients, size N
//   q: second polynomial coefficients, size M
//   N: number of coefficients in p
//   M: number of coefficients in q
template <typename T>
void polynomial_multiply_backward(
    T* grad_p,
    T* grad_q,
    const T* grad_output,
    const T* p,
    const T* q,
    int64_t N,
    int64_t M
) {
    // Initialize gradients to zero
    for (int64_t i = 0; i < N; ++i) {
        grad_p[i] = T(0);
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_q[j] = T(0);
    }

    // grad_p[i] = sum_{j=0}^{M-1} grad_output[i+j] * q[j]
    // This is correlation of grad_output with q
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            grad_p[i] += grad_output[i + j] * q[j];
        }
    }

    // grad_q[j] = sum_{i=0}^{N-1} grad_output[i+j] * p[i]
    // This is correlation of grad_output with p
    for (int64_t j = 0; j < M; ++j) {
        for (int64_t i = 0; i < N; ++i) {
            grad_q[j] += grad_output[i + j] * p[i];
        }
    }
}

// Complex specialization
template <typename T>
void polynomial_multiply_backward(
    c10::complex<T>* grad_p,
    c10::complex<T>* grad_q,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* p,
    const c10::complex<T>* q,
    int64_t N,
    int64_t M
) {
    for (int64_t i = 0; i < N; ++i) {
        grad_p[i] = c10::complex<T>(T(0), T(0));
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_q[j] = c10::complex<T>(T(0), T(0));
    }

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            grad_p[i] += grad_output[i + j] * q[j];
        }
    }

    for (int64_t j = 0; j < M; ++j) {
        for (int64_t i = 0; i < N; ++i) {
            grad_q[j] += grad_output[i + j] * p[i];
        }
    }
}

} // namespace torchscience::kernel::polynomial
