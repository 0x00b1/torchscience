#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for polynomial division
//
// Forward: (p, q) -> (Q, R) where p = q * Q + R, deg(R) < deg(q)
//
// Given gradients w.r.t. outputs (grad_Q, grad_R), compute gradients w.r.t.
// inputs (grad_p, grad_q).
//
// The backward is computed by reversing the forward long division algorithm:
// Forward steps (for i from N-M down to 0):
//   Q[i] = work[M-1+i] / q[M-1]
//   work = work - Q[i] * shift(q, i)
//
// Backward steps (for i from 0 to N-M):
//   Propagate gradients back through subtraction and division
//
// Parameters:
//   grad_p: output gradient w.r.t. p, size N
//   grad_q: output gradient w.r.t. q, size M
//   grad_Q: input gradient w.r.t. quotient, size N - M + 1
//   grad_R: input gradient w.r.t. remainder, size max(M-1, 1)
//   Q: quotient from forward pass, size N - M + 1
//   p: dividend polynomial, size N
//   q: divisor polynomial, size M
//   N: size of p
//   M: size of q
//   work_buffer: temporary buffer of size N for intermediate computations
template <typename T>
void polynomial_divmod_backward(
    T* grad_p,
    T* grad_q,
    const T* grad_Q,
    const T* grad_R,
    const T* Q,
    const T* p,
    const T* q,
    int64_t N,
    int64_t M,
    T* work_buffer  // size N
) {
    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;
    const T leading_q = q[M - 1];

    // Initialize grad_q to zero
    for (int64_t j = 0; j < M; ++j) {
        grad_q[j] = T(0);
    }

    // Initialize work_buffer with grad_R padded to size N
    // This represents the gradient w.r.t. the final working remainder
    for (int64_t k = 0; k < rem_len; ++k) {
        work_buffer[k] = grad_R[k];
    }
    for (int64_t k = rem_len; k < N; ++k) {
        work_buffer[k] = T(0);
    }

    // Reverse the division steps
    // Forward was: for i from quot_len-1 down to 0
    // Backward: for i from 0 to quot_len-1
    for (int64_t i = 0; i < quot_len; ++i) {
        // At forward step (quot_len - 1 - i), we computed:
        // Q[i] = work[M-1+i] / leading_q
        // work_new = work_old - Q[i] * shift(q, i)

        // Total gradient on Q[i] includes:
        // 1. Direct gradient from output: grad_Q[i]
        // 2. Gradient from the subtraction: -sum_j work_buffer[i+j] * q[j]
        T total_grad_Q_i = grad_Q[i];
        for (int64_t j = 0; j < M; ++j) {
            total_grad_Q_i -= work_buffer[i + j] * q[j];
        }

        // Gradient from subtraction w.r.t. q:
        // work_new[i+j] = work_old[i+j] - Q[i] * q[j]
        // => grad_q[j] += -Q[i] * grad_work[i+j]
        for (int64_t j = 0; j < M; ++j) {
            grad_q[j] -= Q[i] * work_buffer[i + j];
        }

        // Backward through division: Q[i] = work[M-1+i] / leading_q
        // => grad_work[M-1+i] += total_grad_Q_i / leading_q
        // => grad_q[M-1] -= total_grad_Q_i * Q[i] / leading_q
        work_buffer[M - 1 + i] += total_grad_Q_i / leading_q;
        grad_q[M - 1] -= total_grad_Q_i * Q[i] / leading_q;
    }

    // After all reverse steps, work_buffer contains grad_p
    for (int64_t k = 0; k < N; ++k) {
        grad_p[k] = work_buffer[k];
    }
}

// Complex specialization
template <typename T>
void polynomial_divmod_backward(
    c10::complex<T>* grad_p,
    c10::complex<T>* grad_q,
    const c10::complex<T>* grad_Q,
    const c10::complex<T>* grad_R,
    const c10::complex<T>* Q,
    const c10::complex<T>* p,
    const c10::complex<T>* q,
    int64_t N,
    int64_t M,
    c10::complex<T>* work_buffer
) {
    using C = c10::complex<T>;
    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;
    const C leading_q = q[M - 1];

    for (int64_t j = 0; j < M; ++j) {
        grad_q[j] = C(T(0), T(0));
    }

    for (int64_t k = 0; k < rem_len; ++k) {
        work_buffer[k] = grad_R[k];
    }
    for (int64_t k = rem_len; k < N; ++k) {
        work_buffer[k] = C(T(0), T(0));
    }

    for (int64_t i = 0; i < quot_len; ++i) {
        C total_grad_Q_i = grad_Q[i];
        for (int64_t j = 0; j < M; ++j) {
            total_grad_Q_i -= work_buffer[i + j] * q[j];
        }

        for (int64_t j = 0; j < M; ++j) {
            grad_q[j] -= Q[i] * work_buffer[i + j];
        }

        work_buffer[M - 1 + i] += total_grad_Q_i / leading_q;
        grad_q[M - 1] -= total_grad_Q_i * Q[i] / leading_q;
    }

    for (int64_t k = 0; k < N; ++k) {
        grad_p[k] = work_buffer[k];
    }
}

} // namespace torchscience::kernel::polynomial
