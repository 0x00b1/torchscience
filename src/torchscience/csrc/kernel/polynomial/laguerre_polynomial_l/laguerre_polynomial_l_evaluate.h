#pragma once

#include <cstdint>

namespace torchscience::kernel::polynomial {

/**
 * Evaluate Laguerre series using Clenshaw's algorithm.
 *
 * The Laguerre polynomials satisfy:
 *   L_0(x) = 1
 *   L_1(x) = 1 - x
 *   L_{k+1}(x) = ((2k+1-x) * L_k(x) - k * L_{k-1}(x)) / (k+1)
 *
 * In standard form: L_{k+1}(x) = (A_k + B_k * x) * L_k(x) - C_k * L_{k-1}(x)
 * where A_k = (2k+1)/(k+1), B_k = -1/(k+1), C_k = k/(k+1)
 *
 * Clenshaw backward recurrence for f(x) = sum_k c_k * L_k(x):
 *   b_{n+1} = b_{n+2} = 0
 *   b_k = c_k + (A_k + B_k * x) * b_{k+1} - C_{k+1} * b_{k+2}
 *   f(x) = b_0
 *
 * @param coeffs Pointer to N coefficients
 * @param x Evaluation point
 * @param N Number of coefficients
 * @return The evaluated polynomial value
 */
template <typename T>
T laguerre_polynomial_l_evaluate(const T* coeffs, T x, int64_t N) {
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
        // A_k = (2k+1)/(k+1)
        T a_k = T(2 * k + 1) / T(k + 1);
        // B_k = -1/(k+1)
        T b_k_coeff = T(-1) / T(k + 1);
        // C_{k+1} = (k+1)/(k+2)
        T c_kp1 = T(k + 1) / T(k + 2);

        // b_k = c_k + (A_k + B_k * x) * b_{k+1} - C_{k+1} * b_{k+2}
        T b_k = coeffs[k] + (a_k + b_k_coeff * x) * b_kp1 - c_kp1 * b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }

    return b_kp1;
}

}  // namespace torchscience::kernel::polynomial
