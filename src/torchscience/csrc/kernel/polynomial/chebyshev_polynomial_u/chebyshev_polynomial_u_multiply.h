#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Chebyshev U polynomial multiplication using linearization formula:
// U_m(x) * U_n(x) = sum_{k=0}^{min(m,n)} U_{m+n-2k}(x)
//
// Given coefficients a[0..N-1] and b[0..M-1], computes output c[0..N+M-2]
//
// Parameters:
//   output: array of size N + M - 1 (initialized to zero)
//   a: first polynomial coefficients, size N
//   b: second polynomial coefficients, size M
//   N: number of coefficients in a
//   M: number of coefficients in b
//
// Returns: size of output (N + M - 1, or 1 if both inputs are size 1)
template <typename T>
int64_t chebyshev_polynomial_u_multiply(
    T* output,
    const T* a,
    const T* b,
    int64_t N,
    int64_t M
) {
    if (N == 0 || M == 0) {
        output[0] = T(0);
        return 1;
    }

    const int64_t output_size = N + M - 1;

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    // Apply linearization formula: U_m * U_n = sum_{k=0}^{min(m,n)} U_{m+n-2k}
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const T product = a[i] * b[j];
            const int64_t min_ij = (i < j) ? i : j;

            // Sum over k from 0 to min(i, j)
            for (int64_t k = 0; k <= min_ij; ++k) {
                const int64_t idx = i + j - 2 * k;
                output[idx] += product;
            }
        }
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_u_multiply(
    c10::complex<T>* output,
    const c10::complex<T>* a,
    const c10::complex<T>* b,
    int64_t N,
    int64_t M
) {
    if (N == 0 || M == 0) {
        output[0] = c10::complex<T>(T(0), T(0));
        return 1;
    }

    const int64_t output_size = N + M - 1;

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = c10::complex<T>(T(0), T(0));
    }

    // Apply linearization formula: U_m * U_n = sum_{k=0}^{min(m,n)} U_{m+n-2k}
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const c10::complex<T> product = a[i] * b[j];
            const int64_t min_ij = (i < j) ? i : j;

            // Sum over k from 0 to min(i, j)
            for (int64_t k = 0; k <= min_ij; ++k) {
                const int64_t idx = i + j - 2 * k;
                output[idx] += product;
            }
        }
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
