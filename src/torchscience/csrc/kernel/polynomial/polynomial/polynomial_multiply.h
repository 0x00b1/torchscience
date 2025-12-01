#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Polynomial multiplication via discrete convolution
// Computes (p * q)(x) where result[k] = sum_{i+j=k} p[i] * q[j]
//
// Parameters:
//   output: output array of size N + M - 1 (must be pre-zeroed or will be zeroed)
//   p: first polynomial coefficients, size N
//   q: second polynomial coefficients, size M
//   N: number of coefficients in p
//   M: number of coefficients in q
//
// Result has degree deg(p) + deg(q), i.e., N + M - 1 coefficients
template <typename T>
void polynomial_multiply(
    T* output,
    const T* p,
    const T* q,
    int64_t N,
    int64_t M
) {
    const int64_t K = N + M - 1;

    // Initialize output to zero
    for (int64_t k = 0; k < K; ++k) {
        output[k] = T(0);
    }

    // Convolution: output[k] = sum_{i+j=k} p[i] * q[j]
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            output[i + j] += p[i] * q[j];
        }
    }
}

// Complex specialization
template <typename T>
void polynomial_multiply(
    c10::complex<T>* output,
    const c10::complex<T>* p,
    const c10::complex<T>* q,
    int64_t N,
    int64_t M
) {
    const int64_t K = N + M - 1;

    for (int64_t k = 0; k < K; ++k) {
        output[k] = c10::complex<T>(T(0), T(0));
    }

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            output[i + j] += p[i] * q[j];
        }
    }
}

} // namespace torchscience::kernel::polynomial
