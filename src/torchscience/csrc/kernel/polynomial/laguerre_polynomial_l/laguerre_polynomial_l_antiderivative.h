#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Laguerre polynomial antiderivative using recurrence:
// integral(L_k) dx = L_k - L_{k+1}
//
// For f(x) = sum_{k=0}^{n-1} c_k L_k(x), the antiderivative is:
// F(x) = sum_{k=0}^{n-1} c_k * (L_k - L_{k+1})
//      = sum_{k=0}^{n-1} c_k L_k - sum_{j=1}^{n} c_{j-1} L_j
//
// Coefficient of L_0: c_0
// Coefficient of L_k (1 <= k <= n-1): c_k - c_{k-1}
// Coefficient of L_n: -c_{n-1}
//
// Parameters:
//   output: array of size N+1
//   coeffs: input Laguerre coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (N+1)
template <typename T>
int64_t laguerre_polynomial_l_antiderivative(T* output, const T* coeffs, int64_t N) {
    const int64_t output_size = N + 1;

    if (N == 0) {
        output[0] = T(0);
        return output_size;
    }

    // Coefficient of L_0: c_0
    output[0] = coeffs[0];

    // Coefficient of L_k (1 <= k <= N-1): c_k - c_{k-1}
    for (int64_t k = 1; k < N; ++k) {
        output[k] = coeffs[k] - coeffs[k - 1];
    }

    // Coefficient of L_N: -c_{N-1}
    output[N] = -coeffs[N - 1];

    return output_size;
}

// Complex specialization
template <typename T>
int64_t laguerre_polynomial_l_antiderivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    const int64_t output_size = N + 1;

    if (N == 0) {
        output[0] = c10::complex<T>(T(0), T(0));
        return output_size;
    }

    // Coefficient of L_0: c_0
    output[0] = coeffs[0];

    // Coefficient of L_k (1 <= k <= N-1): c_k - c_{k-1}
    for (int64_t k = 1; k < N; ++k) {
        output[k] = coeffs[k] - coeffs[k - 1];
    }

    // Coefficient of L_N: -c_{N-1}
    output[N] = -coeffs[N - 1];

    return output_size;
}

} // namespace torchscience::kernel::polynomial
