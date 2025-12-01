#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Helper to compute linearization coefficient for Legendre polynomial product
// P_m(x) * P_n(x) = sum_{k=|m-n|}^{m+n} coeff(m,n,k) * P_k(x)
// where k steps by 2 (same parity as m+n)
//
// The coefficient is computed using:
// coeff(m,n,k) = (2k+1) * [s! / ((s-m)! * (s-n)! * (s-k)!)]^2
//                       * [(2s-2m)! * (2s-2n)! * (2s-2k)!] / [(2s+1)!]
// where s = (m+n+k)/2
//
// We use log-gamma to avoid overflow
template <typename T>
T legendre_linearization_coeff(int64_t m, int64_t n, int64_t k) {
    // Check triangle inequality and parity
    const int64_t abs_diff = (m > n) ? (m - n) : (n - m);
    if (k < abs_diff || k > m + n) {
        return T(0);
    }
    if ((m + n + k) % 2 != 0) {
        return T(0);  // Parity condition: m+n+k must be even
    }

    const int64_t s = (m + n + k) / 2;

    // Use lgamma for numerical stability
    // coeff = (2k+1) * exp(2*log_num - 2*log_denom + log_factorials)
    double log_coeff = std::log(static_cast<double>(2 * k + 1));

    // [s! / ((s-m)! * (s-n)! * (s-k)!)]^2
    double log_num = std::lgamma(static_cast<double>(s + 1));
    double log_denom = std::lgamma(static_cast<double>(s - m + 1))
                     + std::lgamma(static_cast<double>(s - n + 1))
                     + std::lgamma(static_cast<double>(s - k + 1));
    log_coeff += 2.0 * (log_num - log_denom);

    // [(2s-2m)! * (2s-2n)! * (2s-2k)!] / [(2s+1)!]
    log_coeff += std::lgamma(static_cast<double>(2 * s - 2 * m + 1));
    log_coeff += std::lgamma(static_cast<double>(2 * s - 2 * n + 1));
    log_coeff += std::lgamma(static_cast<double>(2 * s - 2 * k + 1));
    log_coeff -= std::lgamma(static_cast<double>(2 * s + 2));

    return static_cast<T>(std::exp(log_coeff));
}

// Legendre polynomial multiplication using linearization formula:
// P_m(x) * P_n(x) = sum_{k=|m-n|}^{m+n,step=2} coeff(m,n,k) * P_k(x)
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
int64_t legendre_polynomial_p_multiply(
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
    // For each pair (i, j), compute P_i * P_j = sum_k coeff(i,j,k) * P_k
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const T product = a[i] * b[j];

            // k ranges from |i-j| to i+j with step 2
            const int64_t abs_diff = (i > j) ? (i - j) : (j - i);
            for (int64_t k = abs_diff; k <= i + j; k += 2) {
                const T coeff = legendre_linearization_coeff<T>(i, j, k);
                output[k] = output[k] + product * coeff;
            }
        }
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t legendre_polynomial_p_multiply(
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
    const c10::complex<T> zero(T(0), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    // Apply linearization formula
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const c10::complex<T> product = a[i] * b[j];

            const int64_t abs_diff = (i > j) ? (i - j) : (j - i);
            for (int64_t k = abs_diff; k <= i + j; k += 2) {
                const T coeff = legendre_linearization_coeff<T>(i, j, k);
                output[k] = output[k] + product * c10::complex<T>(coeff, T(0));
            }
        }
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
