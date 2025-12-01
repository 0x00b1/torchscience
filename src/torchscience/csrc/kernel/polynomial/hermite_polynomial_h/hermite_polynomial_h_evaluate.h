#pragma once

#include <cstdint>

namespace torchscience::kernel::polynomial {

/**
 * Evaluate Physicists' Hermite series using Clenshaw's algorithm.
 *
 * The physicists' Hermite polynomials satisfy:
 *   H_0(x) = 1
 *   H_1(x) = 2x
 *   H_{k+1}(x) = 2x * H_k(x) - 2k * H_{k-1}(x)
 *
 * Clenshaw backward recurrence for f(x) = sum_k c_k * H_k(x):
 *   b_{n+1} = b_{n+2} = 0
 *   b_k = c_k + 2x * b_{k+1} - 2(k+1) * b_{k+2}
 *   f(x) = b_0
 *
 * @param coeffs Pointer to N coefficients
 * @param x Evaluation point
 * @param N Number of coefficients
 * @return The evaluated polynomial value
 */
template <typename T>
T hermite_polynomial_h_evaluate(const T* coeffs, T x, int64_t N) {
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
        // A_k = 2
        // C_{k+1} = 2*(k+1)
        T c_kp1 = T(2) * T(k + 1);

        // b_k = c_k + 2x * b_{k+1} - C_{k+1} * b_{k+2}
        T b_k = coeffs[k] + T(2) * x * b_kp1 - c_kp1 * b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }

    return b_kp1;
}

}  // namespace torchscience::kernel::polynomial
