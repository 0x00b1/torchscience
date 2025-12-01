#pragma once

#include <cstdint>

namespace torchscience::kernel::polynomial {

/**
 * Evaluate Jacobi series using forward recurrence.
 *
 * The Jacobi polynomials satisfy:
 *   P_0^{(α,β)}(x) = 1
 *   P_1^{(α,β)}(x) = (α - β)/2 + (α + β + 2)/2 * x
 *   P_{n+1}^{(α,β)}(x) = ((b_n + c_n*x) * P_n - d_n * P_{n-1}) / a_n
 *
 * where:
 *   a_n = 2(n+1)(n+α+β+1)(2n+α+β)
 *   b_n = (2n+α+β+1)(α²-β²)
 *   c_n = (2n+α+β)(2n+α+β+1)(2n+α+β+2)
 *   d_n = 2(n+α)(n+β)(2n+α+β+2)
 *
 * @param coeffs Pointer to N coefficients
 * @param x Evaluation point
 * @param alpha The α parameter
 * @param beta The β parameter
 * @param N Number of coefficients
 * @return The evaluated polynomial value
 */
template <typename T>
T jacobi_polynomial_p_evaluate(const T* coeffs, T x, T alpha, T beta, int64_t N) {
    if (N == 0) {
        return T(0);
    }

    T ab = alpha + beta;

    // P_0 = 1
    T P_prev_prev = T(1);
    T result = coeffs[0] * P_prev_prev;

    if (N == 1) {
        return result;
    }

    // P_1 = (α - β)/2 + (α + β + 2)/2 * x
    T P_prev = (alpha - beta) / T(2) + (ab + T(2)) / T(2) * x;
    result = result + coeffs[1] * P_prev;

    if (N == 2) {
        return result;
    }

    // Forward recurrence for P_k, k >= 2
    for (int64_t k = 1; k < N - 1; ++k) {
        T k_f = T(k);
        T two_k_ab = T(2) * k_f + ab;

        // Recurrence coefficients
        T a_k = T(2) * (k_f + T(1)) * (k_f + ab + T(1)) * two_k_ab;
        T b_k = (two_k_ab + T(1)) * (alpha * alpha - beta * beta);
        T c_k = two_k_ab * (two_k_ab + T(1)) * (two_k_ab + T(2));
        T d_k = T(2) * (k_f + alpha) * (k_f + beta) * (two_k_ab + T(2));

        // P_{k+1} = ((b_k + c_k*x) * P_k - d_k * P_{k-1}) / a_k
        T P_curr = ((b_k + c_k * x) * P_prev - d_k * P_prev_prev) / a_k;

        result = result + coeffs[k + 1] * P_curr;
        P_prev_prev = P_prev;
        P_prev = P_curr;
    }

    return result;
}

}  // namespace torchscience::kernel::polynomial
