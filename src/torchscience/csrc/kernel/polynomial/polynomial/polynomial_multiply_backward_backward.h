#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for polynomial multiplication
//
// Forward: y[k] = sum_{i+j=k} p[i] * q[j]
//
// First backward:
//   grad_p[i] = sum_j grad_output[i+j] * q[j]
//   grad_q[j] = sum_i grad_output[i+j] * p[i]
//
// Second backward (given gg_p = dL/d(grad_p), gg_q = dL/d(grad_q)):
//   grad_grad_output[k] = sum_i gg_p[i] * q[k-i] + sum_j gg_q[j] * p[k-j]
//                       = conv(gg_p, q)[k] + conv(gg_q, p)[k]
//   grad_p_out[i] = sum_j gg_q[j] * grad_output[i+j]
//   grad_q_out[j] = sum_i gg_p[i] * grad_output[i+j]
//
// Parameters:
//   grad_grad_output: output, size N + M - 1
//   grad_p_out: output gradient w.r.t. p, size N
//   grad_q_out: output gradient w.r.t. q, size M
//   gg_p: upstream gradient w.r.t. grad_p, size N
//   gg_q: upstream gradient w.r.t. grad_q, size M
//   grad_output: original upstream gradient, size N + M - 1
//   p: first polynomial coefficients, size N
//   q: second polynomial coefficients, size M
//   N: number of coefficients in p
//   M: number of coefficients in q
template <typename T>
void polynomial_multiply_backward_backward(
    T* grad_grad_output,
    T* grad_p_out,
    T* grad_q_out,
    const T* gg_p,
    const T* gg_q,
    const T* grad_output,
    const T* p,
    const T* q,
    int64_t N,
    int64_t M
) {
    const int64_t K = N + M - 1;

    // Initialize outputs to zero
    for (int64_t k = 0; k < K; ++k) {
        grad_grad_output[k] = T(0);
    }
    for (int64_t i = 0; i < N; ++i) {
        grad_p_out[i] = T(0);
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_q_out[j] = T(0);
    }

    // grad_grad_output = conv(gg_p, q) + conv(gg_q, p)
    // conv(gg_p, q)[k] = sum_{i+j=k} gg_p[i] * q[j]
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            grad_grad_output[i + j] += gg_p[i] * q[j];
        }
    }
    // conv(gg_q, p)[k] = sum_{j+i=k} gg_q[j] * p[i]
    for (int64_t j = 0; j < M; ++j) {
        for (int64_t i = 0; i < N; ++i) {
            grad_grad_output[i + j] += gg_q[j] * p[i];
        }
    }

    // grad_p_out[i] = sum_j gg_q[j] * grad_output[i+j] (correlation)
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            grad_p_out[i] += gg_q[j] * grad_output[i + j];
        }
    }

    // grad_q_out[j] = sum_i gg_p[i] * grad_output[i+j] (correlation)
    for (int64_t j = 0; j < M; ++j) {
        for (int64_t i = 0; i < N; ++i) {
            grad_q_out[j] += gg_p[i] * grad_output[i + j];
        }
    }
}

// Complex specialization
template <typename T>
void polynomial_multiply_backward_backward(
    c10::complex<T>* grad_grad_output,
    c10::complex<T>* grad_p_out,
    c10::complex<T>* grad_q_out,
    const c10::complex<T>* gg_p,
    const c10::complex<T>* gg_q,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* p,
    const c10::complex<T>* q,
    int64_t N,
    int64_t M
) {
    const int64_t K = N + M - 1;

    for (int64_t k = 0; k < K; ++k) {
        grad_grad_output[k] = c10::complex<T>(T(0), T(0));
    }
    for (int64_t i = 0; i < N; ++i) {
        grad_p_out[i] = c10::complex<T>(T(0), T(0));
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_q_out[j] = c10::complex<T>(T(0), T(0));
    }

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            grad_grad_output[i + j] += gg_p[i] * q[j];
        }
    }
    for (int64_t j = 0; j < M; ++j) {
        for (int64_t i = 0; i < N; ++i) {
            grad_grad_output[i + j] += gg_q[j] * p[i];
        }
    }

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            grad_p_out[i] += gg_q[j] * grad_output[i + j];
        }
    }

    for (int64_t j = 0; j < M; ++j) {
        for (int64_t i = 0; i < N; ++i) {
            grad_q_out[j] += gg_p[i] * grad_output[i + j];
        }
    }
}

} // namespace torchscience::kernel::polynomial
