#pragma once

#include <c10/util/complex.h>
#include <cstdint>

namespace torchscience::kernel::polynomial {

/**
 * Multiply a Jacobi polynomial series by x.
 *
 * Uses the recurrence relation:
 *   P_{k+1}^{(α,β)} = (A_k + B_k * x) * P_k^{(α,β)} - C_k * P_{k-1}^{(α,β)}
 *
 * Solving for x * P_k:
 *   x * P_k = (P_{k+1} - A_k * P_k + C_k * P_{k-1}) / B_k
 *
 * For k = 0:
 *   x * P_0 = (2*P_1 - (α - β)*P_0) / (α + β + 2)
 *
 * @param output Output coefficients [N + 1]
 * @param coeffs Input coefficients [N]
 * @param alpha Alpha parameter
 * @param beta Beta parameter
 * @param N Number of input coefficients
 */
template <typename T>
void jacobi_polynomial_p_mulx(
    T* output,
    const T* coeffs,
    T alpha,
    T beta,
    int64_t N
) {
    const int64_t output_size = N + 1;

    // Initialize output to zero
    for (int64_t i = 0; i < output_size; ++i) {
        output[i] = T(0);
    }

    T ab = alpha + beta;

    for (int64_t k = 0; k < N; ++k) {
        T c_k = coeffs[k];
        T k_f = static_cast<T>(k);

        if (k == 0) {
            // x * P_0 = (2*P_1 - (α - β)*P_0) / (α + β + 2)
            T denom = ab + T(2);
            T inv_denom = T(1) / denom;
            // Contribution to P_0: -c_0 * (α - β) / (α + β + 2)
            output[0] += -c_k * (alpha - beta) * inv_denom;
            // Contribution to P_1: c_0 * 2 / (α + β + 2)
            output[1] += c_k * T(2) * inv_denom;
        } else {
            // General case k >= 1
            T two_k_ab = T(2) * k_f + ab;
            T two_k_ab_p2 = two_k_ab + T(2);

            // A_k = (α² - β²) / ((2k + α + β)(2k + α + β + 2))
            T A_k = (alpha * alpha - beta * beta) / (two_k_ab * two_k_ab_p2);

            // B_k = (2k + α + β + 1)(2k + α + β + 2) / (2(k + 1)(k + α + β + 1))
            T B_k = (two_k_ab + T(1)) * two_k_ab_p2 /
                    (T(2) * (k_f + T(1)) * (k_f + ab + T(1)));

            // C_k = (k + α)(k + β)(2k + α + β + 2) / ((k + 1)(k + α + β + 1)(2k + α + β))
            T C_k = (k_f + alpha) * (k_f + beta) * two_k_ab_p2 /
                    ((k_f + T(1)) * (k_f + ab + T(1)) * two_k_ab);

            T inv_B_k = T(1) / B_k;

            // x * P_k = (P_{k+1} - A_k * P_k + C_k * P_{k-1}) / B_k
            // Contribution to P_{k-1}
            output[k - 1] += c_k * C_k * inv_B_k;
            // Contribution to P_k
            output[k] += -c_k * A_k * inv_B_k;
            // Contribution to P_{k+1}
            output[k + 1] += c_k * inv_B_k;
        }
    }
}

// Complex specialization
template <typename T>
void jacobi_polynomial_p_mulx(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    c10::complex<T> alpha,
    c10::complex<T> beta,
    int64_t N
) {
    const int64_t output_size = N + 1;

    // Initialize output to zero
    for (int64_t i = 0; i < output_size; ++i) {
        output[i] = c10::complex<T>(0, 0);
    }

    c10::complex<T> ab = alpha + beta;
    c10::complex<T> one(1, 0);
    c10::complex<T> two(2, 0);

    for (int64_t k = 0; k < N; ++k) {
        c10::complex<T> c_k = coeffs[k];
        c10::complex<T> k_f(static_cast<T>(k), 0);

        if (k == 0) {
            c10::complex<T> denom = ab + two;
            c10::complex<T> inv_denom = one / denom;
            output[0] += -c_k * (alpha - beta) * inv_denom;
            output[1] += c_k * two * inv_denom;
        } else {
            c10::complex<T> two_k_ab = two * k_f + ab;
            c10::complex<T> two_k_ab_p2 = two_k_ab + two;

            c10::complex<T> A_k = (alpha * alpha - beta * beta) / (two_k_ab * two_k_ab_p2);
            c10::complex<T> B_k = (two_k_ab + one) * two_k_ab_p2 /
                                  (two * (k_f + one) * (k_f + ab + one));
            c10::complex<T> C_k = (k_f + alpha) * (k_f + beta) * two_k_ab_p2 /
                                  ((k_f + one) * (k_f + ab + one) * two_k_ab);

            c10::complex<T> inv_B_k = one / B_k;

            output[k - 1] += c_k * C_k * inv_B_k;
            output[k] += -c_k * A_k * inv_B_k;
            output[k + 1] += c_k * inv_B_k;
        }
    }
}

} // namespace torchscience::kernel::polynomial
