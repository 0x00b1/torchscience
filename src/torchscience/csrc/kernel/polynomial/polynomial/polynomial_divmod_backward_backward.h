#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for polynomial division
//
// This computes gradients of the backward pass w.r.t. its inputs.
//
// Backward pass computed:
//   grad_p, grad_q = divmod_backward(grad_Q, grad_R, Q, p, q)
//
// This function computes:
//   Given gg_p (dL/d(grad_p)) and gg_q (dL/d(grad_q))
//   Compute:
//     grad_grad_Q (dL/d(grad_Q))
//     grad_grad_R (dL/d(grad_R))
//     grad_Q_out (dL/dQ through backward)
//     grad_p_out (dL/dp through backward)
//     grad_q_out (dL/dq through backward)
//
// Parameters:
//   grad_grad_Q: output, size N - M + 1
//   grad_grad_R: output, size max(M-1, 1)
//   grad_Q_out: output, size N - M + 1
//   grad_p_out: output, size N
//   grad_q_out: output, size M
//   gg_p: input, gradient w.r.t. grad_p, size N
//   gg_q: input, gradient w.r.t. grad_q, size M
//   grad_Q: original gradient w.r.t. Q, size N - M + 1
//   grad_R: original gradient w.r.t. R, size max(M-1, 1)
//   Q: quotient from forward, size N - M + 1
//   p: dividend, size N
//   q: divisor, size M
//   work_buffer: temporary, size N
//   work_buffer2: temporary, size N
template <typename T>
void polynomial_divmod_backward_backward(
    T* grad_grad_Q,
    T* grad_grad_R,
    T* grad_Q_out,
    T* grad_p_out,
    T* grad_q_out,
    const T* gg_p,
    const T* gg_q,
    const T* grad_Q,
    const T* grad_R,
    const T* Q,
    const T* p,
    const T* q,
    int64_t N,
    int64_t M,
    T* work_buffer,
    T* work_buffer2
) {
    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;
    const T leading_q = q[M - 1];
    const T leading_q_sq = leading_q * leading_q;

    // Initialize outputs to zero
    for (int64_t i = 0; i < quot_len; ++i) {
        grad_grad_Q[i] = T(0);
        grad_Q_out[i] = T(0);
    }
    for (int64_t k = 0; k < rem_len; ++k) {
        grad_grad_R[k] = T(0);
    }
    for (int64_t k = 0; k < N; ++k) {
        grad_p_out[k] = T(0);
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_q_out[j] = T(0);
    }

    // The backward pass produces grad_p = work_buffer at the end
    // So gg_p contributes to grad_grad_Q and grad_grad_R through the reverse traversal

    // Reconstruct work_buffer states from the backward pass
    // Initialize work_buffer with grad_R (as in backward)
    for (int64_t k = 0; k < rem_len; ++k) {
        work_buffer[k] = grad_R[k];
    }
    for (int64_t k = rem_len; k < N; ++k) {
        work_buffer[k] = T(0);
    }

    // We need to track how gg_p affects the computation
    // gg_p is the gradient of loss w.r.t. the final work_buffer (which becomes grad_p)

    // Store gg_p in work_buffer2 for tracking
    for (int64_t k = 0; k < N; ++k) {
        work_buffer2[k] = gg_p[k];
    }

    // Forward through the backward pass, accumulating contributions
    // The backward pass modifies work_buffer in a specific order
    // We need to differentiate each operation

    for (int64_t i = 0; i < quot_len; ++i) {
        // From backward: total_grad_Q_i = grad_Q[i] - sum_j work_buffer[i+j] * q[j]
        // d(total_grad_Q_i)/d(grad_Q[i]) = 1
        // d(total_grad_Q_i)/d(work_buffer[i+j]) = -q[j]
        // d(total_grad_Q_i)/d(q[j]) = -work_buffer[i+j]

        T total_grad_Q_i = grad_Q[i];
        for (int64_t j = 0; j < M; ++j) {
            total_grad_Q_i -= work_buffer[i + j] * q[j];
        }

        // From backward: grad_q[j] -= Q[i] * work_buffer[i + j]
        // d(grad_q[j])/d(Q[i]) = -work_buffer[i+j]
        // d(grad_q[j])/d(work_buffer[i+j]) = -Q[i]

        // Contribution from gg_q to grad_Q_out (through the Q[i] * work_buffer term)
        for (int64_t j = 0; j < M; ++j) {
            grad_Q_out[i] -= gg_q[j] * work_buffer[i + j];
        }

        // From backward: work_buffer[M-1+i] += total_grad_Q_i / leading_q
        // d(work_buffer[M-1+i])/d(total_grad_Q_i) = 1/leading_q
        // d(work_buffer[M-1+i])/d(leading_q) = -total_grad_Q_i / leading_q^2

        // From backward: grad_q[M-1] -= total_grad_Q_i * Q[i] / leading_q
        // d(grad_q[M-1])/d(total_grad_Q_i) = -Q[i] / leading_q
        // d(grad_q[M-1])/d(Q[i]) = -total_grad_Q_i / leading_q
        // d(grad_q[M-1])/d(leading_q) = total_grad_Q_i * Q[i] / leading_q^2

        // Contribution to grad_Q_out from gg_q[M-1]
        grad_Q_out[i] -= gg_q[M - 1] * total_grad_Q_i / leading_q;

        // Contribution to grad_grad_Q from gg_p and gg_q
        // gg_p[M-1+i] affects total_grad_Q_i through work_buffer update
        T d_total = gg_p[M - 1 + i] / leading_q - gg_q[M - 1] * Q[i] / leading_q;
        grad_grad_Q[i] += d_total;

        // Contribution to grad_q_out from the division operations
        grad_q_out[M - 1] -= gg_p[M - 1 + i] * total_grad_Q_i / leading_q_sq;
        grad_q_out[M - 1] += gg_q[M - 1] * total_grad_Q_i * Q[i] / leading_q_sq;

        // Contributions from the q[j] terms in total_grad_Q_i
        for (int64_t j = 0; j < M; ++j) {
            // total_grad_Q_i contains -work_buffer[i+j] * q[j]
            // This affects grad_q[j] through: grad_q[j] -= Q[i] * work_buffer[i+j]
            // And affects work_buffer[M-1+i] through: += total_grad_Q_i / leading_q

            // Gradient from gg_p[M-1+i] w.r.t. q[j]
            grad_q_out[j] -= gg_p[M - 1 + i] * work_buffer[i + j] / leading_q;

            // Gradient from gg_q[M-1] w.r.t. q[j]
            grad_q_out[j] += gg_q[M - 1] * Q[i] * work_buffer[i + j] / leading_q;

            // grad_grad_R contributions (work_buffer starts from grad_R)
            if (i + j < rem_len) {
                // This work_buffer element came from grad_R
                grad_grad_R[i + j] -= d_total * q[j];
                grad_grad_R[i + j] -= gg_q[j] * Q[i];
            }
        }

        // Update work_buffer as in backward (for next iteration's state)
        work_buffer[M - 1 + i] += total_grad_Q_i / leading_q;
    }

    // grad_p_out: The backward pass doesn't use p directly in the gradient computation
    // (it uses Q which was computed from p, but Q is a separate input to backward)
    // So grad_p_out remains zero
}

// Complex specialization
template <typename T>
void polynomial_divmod_backward_backward(
    c10::complex<T>* grad_grad_Q,
    c10::complex<T>* grad_grad_R,
    c10::complex<T>* grad_Q_out,
    c10::complex<T>* grad_p_out,
    c10::complex<T>* grad_q_out,
    const c10::complex<T>* gg_p,
    const c10::complex<T>* gg_q,
    const c10::complex<T>* grad_Q,
    const c10::complex<T>* grad_R,
    const c10::complex<T>* Q,
    const c10::complex<T>* p,
    const c10::complex<T>* q,
    int64_t N,
    int64_t M,
    c10::complex<T>* work_buffer,
    c10::complex<T>* work_buffer2
) {
    using C = c10::complex<T>;
    const int64_t quot_len = N - M + 1;
    const int64_t rem_len = (M > 1) ? (M - 1) : 1;
    const C leading_q = q[M - 1];
    const C leading_q_sq = leading_q * leading_q;

    for (int64_t i = 0; i < quot_len; ++i) {
        grad_grad_Q[i] = C(T(0), T(0));
        grad_Q_out[i] = C(T(0), T(0));
    }
    for (int64_t k = 0; k < rem_len; ++k) {
        grad_grad_R[k] = C(T(0), T(0));
    }
    for (int64_t k = 0; k < N; ++k) {
        grad_p_out[k] = C(T(0), T(0));
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_q_out[j] = C(T(0), T(0));
    }

    for (int64_t k = 0; k < rem_len; ++k) {
        work_buffer[k] = grad_R[k];
    }
    for (int64_t k = rem_len; k < N; ++k) {
        work_buffer[k] = C(T(0), T(0));
    }

    for (int64_t k = 0; k < N; ++k) {
        work_buffer2[k] = gg_p[k];
    }

    for (int64_t i = 0; i < quot_len; ++i) {
        C total_grad_Q_i = grad_Q[i];
        for (int64_t j = 0; j < M; ++j) {
            total_grad_Q_i -= work_buffer[i + j] * q[j];
        }

        for (int64_t j = 0; j < M; ++j) {
            grad_Q_out[i] -= gg_q[j] * work_buffer[i + j];
        }

        grad_Q_out[i] -= gg_q[M - 1] * total_grad_Q_i / leading_q;

        C d_total = gg_p[M - 1 + i] / leading_q - gg_q[M - 1] * Q[i] / leading_q;
        grad_grad_Q[i] += d_total;

        grad_q_out[M - 1] -= gg_p[M - 1 + i] * total_grad_Q_i / leading_q_sq;
        grad_q_out[M - 1] += gg_q[M - 1] * total_grad_Q_i * Q[i] / leading_q_sq;

        for (int64_t j = 0; j < M; ++j) {
            grad_q_out[j] -= gg_p[M - 1 + i] * work_buffer[i + j] / leading_q;
            grad_q_out[j] += gg_q[M - 1] * Q[i] * work_buffer[i + j] / leading_q;

            if (i + j < rem_len) {
                grad_grad_R[i + j] -= d_total * q[j];
                grad_grad_R[i + j] -= gg_q[j] * Q[i];
            }
        }

        work_buffer[M - 1 + i] += total_grad_Q_i / leading_q;
    }
}

} // namespace torchscience::kernel::polynomial
