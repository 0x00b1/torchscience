#pragma once

#include <cstdint>

namespace torchscience::kernel::polynomial {

/**
 * Compute gradient of Jacobi series evaluation with respect to x.
 *
 * For f(x) = sum_{k=0}^{n-1} c_k * P_k^{(α,β)}(x), we need:
 *   df/dx = sum_{k=0}^{n-1} c_k * P_k'^{(α,β)}(x)
 *
 * Using the derivative formula:
 *   P_n'^{(α,β)}(x) = (n + α + β + 1)/2 * P_{n-1}^{(α+1,β+1)}(x)
 *
 * @param coeffs Pointer to N coefficients
 * @param x Evaluation point
 * @param alpha The α parameter
 * @param beta The β parameter
 * @param N Number of coefficients
 * @return The gradient df/dx
 */
template <typename T>
T jacobi_polynomial_p_evaluate_backward_x(const T* coeffs, T x, T alpha, T beta, int64_t N) {
    if (N <= 1) {
        return T(0);  // P_0 is constant, derivative is 0
    }

    // Derivative of P_n^{(α,β)}(x) = (n + α + β + 1)/2 * P_{n-1}^{(α+1,β+1)}(x)
    // So df/dx = sum_{k=1}^{n-1} c_k * (k + α + β + 1)/2 * P_{k-1}^{(α+1,β+1)}(x)

    T alpha_p = alpha + T(1);
    T beta_p = beta + T(1);
    T ab_p = alpha_p + beta_p;  // α+1 + β+1 = α + β + 2

    T result = T(0);

    // P_0^{(α+1,β+1)} = 1
    T P_prev_prev = T(1);

    // First term: c_1 * (1 + α + β + 1)/2 * P_0^{(α+1,β+1)}
    T factor_1 = (T(1) + alpha + beta + T(1)) / T(2);  // (1 + α + β + 1)/2
    result = coeffs[1] * factor_1 * P_prev_prev;

    if (N == 2) {
        return result;
    }

    // P_1^{(α+1,β+1)}(x) = (α+1 - β-1)/2 + (α+1 + β+1 + 2)/2 * x
    //                   = (α - β)/2 + (α + β + 4)/2 * x
    T P_prev = (alpha_p - beta_p) / T(2) + (ab_p + T(2)) / T(2) * x;

    T factor_2 = (T(2) + alpha + beta + T(1)) / T(2);  // (2 + α + β + 1)/2
    result = result + coeffs[2] * factor_2 * P_prev;

    if (N == 3) {
        return result;
    }

    // Forward recurrence for P_k^{(α+1,β+1)}, k >= 2
    for (int64_t k = 1; k < N - 2; ++k) {
        T k_f = T(k);
        T two_k_ab = T(2) * k_f + ab_p;

        T a_k = T(2) * (k_f + T(1)) * (k_f + ab_p + T(1)) * two_k_ab;
        T b_k = (two_k_ab + T(1)) * (alpha_p * alpha_p - beta_p * beta_p);
        T c_k = two_k_ab * (two_k_ab + T(1)) * (two_k_ab + T(2));
        T d_k = T(2) * (k_f + alpha_p) * (k_f + beta_p) * (two_k_ab + T(2));

        T P_curr = ((b_k + c_k * x) * P_prev - d_k * P_prev_prev) / a_k;

        // c_{k+2} * (k+2 + α + β + 1)/2 * P_{k+1}^{(α+1,β+1)}
        T factor_k2 = (T(k + 2) + alpha + beta + T(1)) / T(2);
        result = result + coeffs[k + 2] * factor_k2 * P_curr;

        P_prev_prev = P_prev;
        P_prev = P_curr;
    }

    return result;
}

/**
 * Compute all Jacobi polynomials P_k^{(α,β)}(x) for k = 0, ..., N-1.
 *
 * @param P_values Output array of size N
 * @param x Evaluation point
 * @param alpha The α parameter
 * @param beta The β parameter
 * @param N Number of polynomials to compute
 */
template <typename T>
void jacobi_polynomial_p_compute_all(T* P_values, T x, T alpha, T beta, int64_t N) {
    if (N == 0) return;

    T ab = alpha + beta;

    P_values[0] = T(1);
    if (N == 1) return;

    P_values[1] = (alpha - beta) / T(2) + (ab + T(2)) / T(2) * x;
    if (N == 2) return;

    for (int64_t k = 1; k < N - 1; ++k) {
        T k_f = T(k);
        T two_k_ab = T(2) * k_f + ab;

        T a_k = T(2) * (k_f + T(1)) * (k_f + ab + T(1)) * two_k_ab;
        T b_k = (two_k_ab + T(1)) * (alpha * alpha - beta * beta);
        T c_k = two_k_ab * (two_k_ab + T(1)) * (two_k_ab + T(2));
        T d_k = T(2) * (k_f + alpha) * (k_f + beta) * (two_k_ab + T(2));

        P_values[k + 1] = ((b_k + c_k * x) * P_values[k] - d_k * P_values[k - 1]) / a_k;
    }
}

}  // namespace torchscience::kernel::polynomial
