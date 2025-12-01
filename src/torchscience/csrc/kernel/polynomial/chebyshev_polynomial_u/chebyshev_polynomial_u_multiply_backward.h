#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Backward pass for Chebyshev U polynomial multiplication.
//
// Forward used linearization formula:
// U_m(x) * U_n(x) = sum_{k=0}^{min(m,n)} U_{m+n-2k}(x)
//
// For backward, we trace how each output coefficient c[idx] depends on inputs:
// c[i+j-2k] += a[i] * b[j] for k = 0 to min(i,j)
//
// Backward pass reverses this:
// grad_a[i] = sum over j,k: grad_c[i+j-2k] * b[j]
// grad_b[j] = sum over i,k: grad_c[i+j-2k] * a[i]
//
// Parameters:
//   grad_a: output gradient for a, size N
//   grad_b: output gradient for b, size M
//   grad_output: incoming gradient, size output_size
//   a: first polynomial coefficients, size N
//   b: second polynomial coefficients, size M
//   N: number of coefficients in a
//   M: number of coefficients in b
//   output_size: size of grad_output
template <typename T>
void chebyshev_polynomial_u_multiply_backward(
    T* grad_a,
    T* grad_b,
    const T* grad_output,
    const T* a,
    const T* b,
    int64_t N,
    int64_t M,
    int64_t output_size
) {
    // Initialize gradients to zero
    for (int64_t i = 0; i < N; ++i) {
        grad_a[i] = T(0);
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b[j] = T(0);
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Reverse the linearization formula
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const int64_t min_ij = (i < j) ? i : j;

            // Sum over k from 0 to min(i, j)
            for (int64_t k = 0; k <= min_ij; ++k) {
                const int64_t idx = i + j - 2 * k;
                if (idx < output_size) {
                    grad_a[i] += grad_output[idx] * b[j];
                    grad_b[j] += grad_output[idx] * a[i];
                }
            }
        }
    }
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_u_multiply_backward(
    c10::complex<T>* grad_a,
    c10::complex<T>* grad_b,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* a,
    const c10::complex<T>* b,
    int64_t N,
    int64_t M,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));

    // Initialize gradients to zero
    for (int64_t i = 0; i < N; ++i) {
        grad_a[i] = zero;
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b[j] = zero;
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Reverse the linearization formula
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const int64_t min_ij = (i < j) ? i : j;

            // Sum over k from 0 to min(i, j)
            for (int64_t k = 0; k <= min_ij; ++k) {
                const int64_t idx = i + j - 2 * k;
                if (idx < output_size) {
                    grad_a[i] += grad_output[idx] * b[j];
                    grad_b[j] += grad_output[idx] * a[i];
                }
            }
        }
    }
}

} // namespace torchscience::kernel::polynomial
