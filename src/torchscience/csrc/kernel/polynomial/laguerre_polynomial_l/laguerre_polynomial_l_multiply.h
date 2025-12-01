#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace torchscience::kernel::polynomial {

// Helper: compute binomial coefficient C(n,k)
template <typename T>
T laguerre_binomial(int64_t n, int64_t k) {
    if (k < 0 || k > n) return T(0);
    if (k == 0 || k == n) return T(1);

    // Use symmetry
    if (k > n - k) k = n - k;

    T result = T(1);
    for (int64_t i = 0; i < k; ++i) {
        result = result * T(n - i) / T(i + 1);
    }
    return result;
}

// Helper: compute factorial
template <typename T>
T laguerre_factorial(int64_t n) {
    T result = T(1);
    for (int64_t i = 2; i <= n; ++i) {
        result = result * T(i);
    }
    return result;
}

// Laguerre polynomial multiplication using power series conversion
//
// L_n(x) = sum_{k=0}^n (-1)^k * C(n,k) / k! * x^k
// x^n = n! * sum_{k=0}^n (-1)^{n-k} * C(n,k) * L_k(x)
//
// Algorithm:
// 1. Convert Laguerre to power series
// 2. Convolve power series
// 3. Convert back to Laguerre
//
// Parameters:
//   output: array of size N+M-1 (or 1 if N=0 or M=0)
//   a: first Laguerre coefficients, size N
//   b: second Laguerre coefficients, size M
//   N: number of coefficients in a
//   M: number of coefficients in b
template <typename T>
void laguerre_polynomial_l_multiply(
    T* output,
    const T* a,
    const T* b,
    int64_t N,
    int64_t M
) {
    const int64_t output_N = (N == 0 || M == 0) ? 1 : (N + M - 1);

    // Initialize output to zero
    for (int64_t k = 0; k < output_N; ++k) {
        output[k] = T(0);
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Step 1: Convert a to power series
    // p_k = sum_{j=k}^{N-1} a[j] * (-1)^k * C(j,k) / k!
    std::vector<T> pa(N, T(0));
    for (int64_t k = 0; k < N; ++k) {
        T sign_k = (k % 2 == 0) ? T(1) : T(-1);
        T fact_k = laguerre_factorial<T>(k);
        for (int64_t j = k; j < N; ++j) {
            pa[k] = pa[k] + a[j] * sign_k * laguerre_binomial<T>(j, k) / fact_k;
        }
    }

    // Step 2: Convert b to power series
    std::vector<T> pb(M, T(0));
    for (int64_t k = 0; k < M; ++k) {
        T sign_k = (k % 2 == 0) ? T(1) : T(-1);
        T fact_k = laguerre_factorial<T>(k);
        for (int64_t j = k; j < M; ++j) {
            pb[k] = pb[k] + b[j] * sign_k * laguerre_binomial<T>(j, k) / fact_k;
        }
    }

    // Step 3: Convolve power series
    std::vector<T> pc(output_N, T(0));
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            pc[i + j] = pc[i + j] + pa[i] * pb[j];
        }
    }

    // Step 4: Convert power series back to Laguerre
    // c_k = sum_{j=k}^{output_N-1} p[j] * j! * (-1)^{j-k} * C(j,k)
    for (int64_t k = 0; k < output_N; ++k) {
        for (int64_t j = k; j < output_N; ++j) {
            T sign_jk = ((j - k) % 2 == 0) ? T(1) : T(-1);
            T fact_j = laguerre_factorial<T>(j);
            output[k] = output[k] + pc[j] * fact_j * sign_jk * laguerre_binomial<T>(j, k);
        }
    }
}

// Complex specialization
template <typename T>
void laguerre_polynomial_l_multiply(
    c10::complex<T>* output,
    const c10::complex<T>* a,
    const c10::complex<T>* b,
    int64_t N,
    int64_t M
) {
    const int64_t output_N = (N == 0 || M == 0) ? 1 : (N + M - 1);
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t k = 0; k < output_N; ++k) {
        output[k] = zero;
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Step 1: Convert a to power series
    std::vector<c10::complex<T>> pa(N, zero);
    for (int64_t k = 0; k < N; ++k) {
        T sign_k = (k % 2 == 0) ? T(1) : T(-1);
        T fact_k = laguerre_factorial<T>(k);
        c10::complex<T> coeff(sign_k / fact_k, T(0));
        for (int64_t j = k; j < N; ++j) {
            T binom = laguerre_binomial<T>(j, k);
            pa[k] = pa[k] + a[j] * c10::complex<T>(binom, T(0)) * coeff;
        }
    }

    // Step 2: Convert b to power series
    std::vector<c10::complex<T>> pb(M, zero);
    for (int64_t k = 0; k < M; ++k) {
        T sign_k = (k % 2 == 0) ? T(1) : T(-1);
        T fact_k = laguerre_factorial<T>(k);
        c10::complex<T> coeff(sign_k / fact_k, T(0));
        for (int64_t j = k; j < M; ++j) {
            T binom = laguerre_binomial<T>(j, k);
            pb[k] = pb[k] + b[j] * c10::complex<T>(binom, T(0)) * coeff;
        }
    }

    // Step 3: Convolve
    std::vector<c10::complex<T>> pc(output_N, zero);
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            pc[i + j] = pc[i + j] + pa[i] * pb[j];
        }
    }

    // Step 4: Convert back to Laguerre
    for (int64_t k = 0; k < output_N; ++k) {
        for (int64_t j = k; j < output_N; ++j) {
            T sign_jk = ((j - k) % 2 == 0) ? T(1) : T(-1);
            T fact_j = laguerre_factorial<T>(j);
            T binom = laguerre_binomial<T>(j, k);
            output[k] = output[k] + pc[j] * c10::complex<T>(fact_j * sign_jk * binom, T(0));
        }
    }
}

} // namespace torchscience::kernel::polynomial
