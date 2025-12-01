#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Chebyshev W polynomial multiplication using linearization formula.
//
// Chebyshev W polynomials (fourth kind) satisfy:
//   W_n(x) = 2x*W_{n-1}(x) - W_{n-2}(x)  with W_0(x) = 1, W_1(x) = 2x + 1
//
// The linearization formula for W polynomials:
//   W_m(x) * W_n(x) = sum_{k=|m-n|}^{m+n} c_k * W_k(x)
//
// For the fourth kind Chebyshev polynomials, the product can be expressed as:
//   W_m(x) * W_n(x) = 0.5 * (W_{m+n}(x) + W_{|m-n|}(x))  for m,n >= 1
//   W_0(x) * W_k(x) = W_k(x)
//
// This is analogous to the T polynomial linearization formula.
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
int64_t chebyshev_polynomial_w_multiply(
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

    // Apply linearization formula
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const T product = a[i] * b[j];
            const int64_t sum_idx = i + j;
            const int64_t diff_idx = std::abs(i - j);

            if (i == 0 || j == 0) {
                // W_0(x) * W_k(x) = W_k(x)
                output[sum_idx] += product;
            } else {
                // W_m(x) * W_n(x) = 0.5 * (W_{m+n}(x) + W_{|m-n|}(x))
                output[sum_idx] += T(0.5) * product;
                output[diff_idx] += T(0.5) * product;
            }
        }
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_w_multiply(
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

    const c10::complex<T> half(T(0.5), T(0));

    // Apply linearization formula
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const c10::complex<T> product = a[i] * b[j];
            const int64_t sum_idx = i + j;
            const int64_t diff_idx = std::abs(i - j);

            if (i == 0 || j == 0) {
                // W_0(x) * W_k(x) = W_k(x)
                output[sum_idx] += product;
            } else {
                // W_m(x) * W_n(x) = 0.5 * (W_{m+n}(x) + W_{|m-n|}(x))
                output[sum_idx] += half * product;
                output[diff_idx] += half * product;
            }
        }
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
