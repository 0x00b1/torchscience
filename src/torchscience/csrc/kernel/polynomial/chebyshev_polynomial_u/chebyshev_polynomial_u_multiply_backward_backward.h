#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Second-order backward for Chebyshev U polynomial multiplication.
//
// The multiplication operation is bilinear: c = f(a, b)
// First backward: grad_a, grad_b = f_backward(grad_c, a, b)
//
// For second-order gradients, we compute:
// - grad_grad_output (gg_output): gradient w.r.t. grad_output from backward
// - grad_a_from_gg: gradient w.r.t. a from second-order terms
// - grad_b_from_gg: gradient w.r.t. b from second-order terms
//
// Since the operation is bilinear (linear in each argument):
// - grad_a[i] = sum_{j,k} grad_c[i+j-2k] * b[j]  (linear in b and grad_c)
// - grad_b[j] = sum_{i,k} grad_c[i+j-2k] * a[i]  (linear in a and grad_c)
//
// The second-order terms are:
// - d(grad_a)/d(grad_c) -> gg_a flows to grad_grad_output
// - d(grad_b)/d(grad_c) -> gg_b flows to grad_grad_output
// - d(grad_a)/d(b) -> gg_a flows to grad_b
// - d(grad_b)/d(a) -> gg_b flows to grad_a
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   grad_a_from_gg: additional gradient for a from gg_b, size N
//   grad_b_from_gg: additional gradient for b from gg_a, size M
//   gg_a: incoming second-order gradient for a, size N
//   gg_b: incoming second-order gradient for b, size M
//   grad_output: gradient from forward backward, size output_size
//   a: first polynomial coefficients, size N
//   b: second polynomial coefficients, size M
//   N: number of coefficients in a
//   M: number of coefficients in b
//   output_size: size of grad_output and grad_grad_output
template <typename T>
void chebyshev_polynomial_u_multiply_backward_backward(
    T* grad_grad_output,
    T* grad_a_from_gg,
    T* grad_b_from_gg,
    const T* gg_a,
    const T* gg_b,
    const T* grad_output,
    const T* a,
    const T* b,
    int64_t N,
    int64_t M,
    int64_t output_size
) {
    // Initialize outputs to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = T(0);
    }
    for (int64_t i = 0; i < N; ++i) {
        grad_a_from_gg[i] = T(0);
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b_from_gg[j] = T(0);
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Compute second-order terms
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const int64_t min_ij = (i < j) ? i : j;

            // Sum over k from 0 to min(i, j)
            for (int64_t k = 0; k <= min_ij; ++k) {
                const int64_t idx = i + j - 2 * k;
                if (idx < output_size) {
                    // Forward backward was:
                    // grad_a[i] += grad_c[idx] * b[j]
                    // grad_b[j] += grad_c[idx] * a[i]
                    //
                    // Second-order:
                    // d(grad_a[i])/d(grad_c[idx]) = b[j] -> gg_a[i] * b[j] to grad_grad_output[idx]
                    // d(grad_a[i])/d(b[j]) = grad_c[idx] -> gg_a[i] * grad_c[idx] to grad_b_from_gg[j]
                    // d(grad_b[j])/d(grad_c[idx]) = a[i] -> gg_b[j] * a[i] to grad_grad_output[idx]
                    // d(grad_b[j])/d(a[i]) = grad_c[idx] -> gg_b[j] * grad_c[idx] to grad_a_from_gg[i]
                    grad_grad_output[idx] += gg_a[i] * b[j];
                    grad_grad_output[idx] += gg_b[j] * a[i];
                    grad_b_from_gg[j] += gg_a[i] * grad_output[idx];
                    grad_a_from_gg[i] += gg_b[j] * grad_output[idx];
                }
            }
        }
    }
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_u_multiply_backward_backward(
    c10::complex<T>* grad_grad_output,
    c10::complex<T>* grad_a_from_gg,
    c10::complex<T>* grad_b_from_gg,
    const c10::complex<T>* gg_a,
    const c10::complex<T>* gg_b,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* a,
    const c10::complex<T>* b,
    int64_t N,
    int64_t M,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));

    // Initialize outputs to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = zero;
    }
    for (int64_t i = 0; i < N; ++i) {
        grad_a_from_gg[i] = zero;
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b_from_gg[j] = zero;
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Compute second-order terms
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const int64_t min_ij = (i < j) ? i : j;

            // Sum over k from 0 to min(i, j)
            for (int64_t k = 0; k <= min_ij; ++k) {
                const int64_t idx = i + j - 2 * k;
                if (idx < output_size) {
                    grad_grad_output[idx] += gg_a[i] * b[j];
                    grad_grad_output[idx] += gg_b[j] * a[i];
                    grad_b_from_gg[j] += gg_a[i] * grad_output[idx];
                    grad_a_from_gg[i] += gg_b[j] * grad_output[idx];
                }
            }
        }
    }
}

} // namespace torchscience::kernel::polynomial
