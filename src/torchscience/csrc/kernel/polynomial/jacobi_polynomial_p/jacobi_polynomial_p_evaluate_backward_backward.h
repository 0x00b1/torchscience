#pragma once

#include <cstdint>

namespace torchscience::kernel::polynomial {

/**
 * Compute second derivative of Jacobi series evaluation with respect to x.
 *
 * For f(x) = sum_{k=0}^{n-1} c_k * P_k^{(α,β)}(x), we need:
 *   d²f/dx² = sum_{k=0}^{n-1} c_k * P_k''^{(α,β)}(x)
 *
 * Using the derivative formula twice:
 *   P_n'^{(α,β)}(x) = (n + α + β + 1)/2 * P_{n-1}^{(α+1,β+1)}(x)
 *   P_n''^{(α,β)}(x) = (n + α + β + 1)/2 * (n + α + β + 2)/2 * P_{n-2}^{(α+2,β+2)}(x)
 *
 * @param coeffs Pointer to N coefficients
 * @param x Evaluation point
 * @param alpha The α parameter
 * @param beta The β parameter
 * @param N Number of coefficients
 * @return The second derivative d²f/dx²
 */
template <typename T>
T jacobi_polynomial_p_evaluate_backward_backward_x(const T* coeffs, T x, T alpha, T beta, int64_t N) {
    if (N <= 2) {
        return T(0);  // P_0 and P_1 have zero second derivative
    }

    // P_n''^{(α,β)}(x) = (n+α+β+1)(n+α+β+2)/4 * P_{n-2}^{(α+2,β+2)}(x)
    // So d²f/dx² = sum_{k=2}^{n-1} c_k * (k+α+β+1)(k+α+β+2)/4 * P_{k-2}^{(α+2,β+2)}(x)

    T alpha_pp = alpha + T(2);
    T beta_pp = beta + T(2);
    T ab_pp = alpha_pp + beta_pp;  // α+2 + β+2 = α + β + 4

    T result = T(0);

    // P_0^{(α+2,β+2)} = 1
    T P_prev_prev = T(1);

    // First term: c_2 * (2+α+β+1)(2+α+β+2)/4 * P_0^{(α+2,β+2)}
    T ab = alpha + beta;
    T factor_2 = (T(2) + ab + T(1)) * (T(2) + ab + T(2)) / T(4);
    result = coeffs[2] * factor_2 * P_prev_prev;

    if (N == 3) {
        return result;
    }

    // P_1^{(α+2,β+2)}(x)
    T P_prev = (alpha_pp - beta_pp) / T(2) + (ab_pp + T(2)) / T(2) * x;

    T factor_3 = (T(3) + ab + T(1)) * (T(3) + ab + T(2)) / T(4);
    result = result + coeffs[3] * factor_3 * P_prev;

    if (N == 4) {
        return result;
    }

    // Forward recurrence for P_k^{(α+2,β+2)}, k >= 2
    for (int64_t k = 1; k < N - 3; ++k) {
        T k_f = T(k);
        T two_k_ab = T(2) * k_f + ab_pp;

        T a_k = T(2) * (k_f + T(1)) * (k_f + ab_pp + T(1)) * two_k_ab;
        T b_k = (two_k_ab + T(1)) * (alpha_pp * alpha_pp - beta_pp * beta_pp);
        T c_k = two_k_ab * (two_k_ab + T(1)) * (two_k_ab + T(2));
        T d_k = T(2) * (k_f + alpha_pp) * (k_f + beta_pp) * (two_k_ab + T(2));

        T P_curr = ((b_k + c_k * x) * P_prev - d_k * P_prev_prev) / a_k;

        // c_{k+3} * (k+3+α+β+1)(k+3+α+β+2)/4 * P_{k+1}^{(α+2,β+2)}
        T factor_k3 = (T(k + 3) + ab + T(1)) * (T(k + 3) + ab + T(2)) / T(4);
        result = result + coeffs[k + 3] * factor_k3 * P_curr;

        P_prev_prev = P_prev;
        P_prev = P_curr;
    }

    return result;
}

}  // namespace torchscience::kernel::polynomial
