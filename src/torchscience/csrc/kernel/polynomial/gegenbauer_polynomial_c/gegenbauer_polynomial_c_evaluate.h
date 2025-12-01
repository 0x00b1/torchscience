#pragma once

#include <cstdint>

namespace torchscience::kernel::polynomial {

/**
 * Evaluate Gegenbauer series using Clenshaw's algorithm.
 *
 * The Gegenbauer polynomials satisfy:
 *   C_0^{λ}(x) = 1
 *   C_1^{λ}(x) = 2λx
 *   C_{k+1}^{λ}(x) = (2(k+λ)/(k+1)) * x * C_k^{λ}(x) - ((k+2λ-1)/(k+1)) * C_{k-1}^{λ}(x)
 *
 * Clenshaw backward recurrence for f(x) = sum_k c_k * C_k^{λ}(x):
 *   b_{n+1} = b_{n+2} = 0
 *   b_k = c_k + A_k * x * b_{k+1} - C'_{k+1} * b_{k+2}
 *   where A_k = 2(k+λ)/(k+1), C'_{k+1} = (k+2λ)/(k+2)
 *   f(x) = b_0
 *
 * @param coeffs Pointer to N coefficients
 * @param x Evaluation point
 * @param lambda_ The λ parameter (alpha)
 * @param N Number of coefficients
 * @return The evaluated polynomial value
 */
template <typename T>
T gegenbauer_polynomial_c_evaluate(const T* coeffs, T x, T lambda_, int64_t N) {
    if (N == 0) {
        return T(0);
    }

    if (N == 1) {
        return coeffs[0];
    }

    T b_kp2 = T(0);  // b_{k+2}
    T b_kp1 = coeffs[N - 1];  // b_{k+1}, starts as c_{N-1}

    // Clenshaw backward recurrence
    for (int64_t k = N - 2; k >= 0; --k) {
        // A_k = 2(k+λ)/(k+1)
        T a_k = T(2) * (T(k) + lambda_) / T(k + 1);
        // C'_{k+1} = (k+2λ)/(k+2)
        T c_kp1 = (T(k) + T(2) * lambda_) / T(k + 2);

        // b_k = c_k + A_k * x * b_{k+1} - C'_{k+1} * b_{k+2}
        T b_k = coeffs[k] + a_k * x * b_kp1 - c_kp1 * b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }

    return b_kp1;
}

}  // namespace torchscience::kernel::polynomial
