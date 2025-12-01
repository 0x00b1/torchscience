#pragma once

#include <c10/util/complex.h>
#include <cstdint>

namespace torchscience::kernel::polynomial {

/**
 * Second-order backward pass for Jacobi polynomial mulx.
 *
 * Computes gradient of grad_output with respect to gg_coeffs.
 * This is the same operation as the forward pass since the backward
 * w.r.t. grad_output is the transpose of the Jacobian applied to gg_coeffs.
 *
 * @param grad_grad_output Output [output_size]
 * @param gg_coeffs Second-order gradient input [N]
 * @param alpha Alpha parameter
 * @param beta Beta parameter
 * @param N Number of coefficients
 * @param output_size Size of output (N + 1)
 */
template <typename T>
void jacobi_polynomial_p_mulx_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    T alpha,
    T beta,
    int64_t N,
    int64_t output_size
) {
    // Initialize output to zero
    for (int64_t i = 0; i < output_size; ++i) {
        grad_grad_output[i] = T(0);
    }

    T ab = alpha + beta;
    T ab_p2 = ab + T(2);
    T inv_ab_p2 = T(1) / ab_p2;
    T alpha_minus_beta = alpha - beta;

    // k = 0: apply the same forward mapping
    T gg_0 = gg_coeffs[0];
    grad_grad_output[0] += -gg_0 * alpha_minus_beta * inv_ab_p2;
    grad_grad_output[1] += gg_0 * T(2) * inv_ab_p2;

    // k >= 1
    for (int64_t k = 1; k < N; ++k) {
        T k_f = static_cast<T>(k);
        T two_k_ab = T(2) * k_f + ab;
        T two_k_ab_p2 = two_k_ab + T(2);

        // A_k, B_k, C_k as in forward
        T alpha_sq = alpha * alpha;
        T beta_sq = beta * beta;
        T A_k = (alpha_sq - beta_sq) / (two_k_ab * two_k_ab_p2);

        T k_p1 = k_f + T(1);
        T k_ab_p1 = k_f + ab + T(1);
        T B_k = (two_k_ab + T(1)) * two_k_ab_p2 / (T(2) * k_p1 * k_ab_p1);
        T inv_B_k = T(1) / B_k;

        T k_alpha = k_f + alpha;
        T k_beta = k_f + beta;
        T C_k = k_alpha * k_beta * two_k_ab_p2 / (k_p1 * k_ab_p1 * two_k_ab);

        T gg_k = gg_coeffs[k];

        // Apply forward mapping: gg_k contributes to output[k-1], output[k], output[k+1]
        grad_grad_output[k - 1] += gg_k * C_k * inv_B_k;
        grad_grad_output[k] += -gg_k * A_k * inv_B_k;
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] += gg_k * inv_B_k;
        }
    }
}

// Complex specialization
template <typename T>
void jacobi_polynomial_p_mulx_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    c10::complex<T> alpha,
    c10::complex<T> beta,
    int64_t N,
    int64_t output_size
) {
    using C = c10::complex<T>;
    C one(1, 0);
    C two(2, 0);

    for (int64_t i = 0; i < output_size; ++i) {
        grad_grad_output[i] = C(0, 0);
    }

    C ab = alpha + beta;
    C ab_p2 = ab + two;
    C inv_ab_p2 = one / ab_p2;
    C alpha_minus_beta = alpha - beta;

    C gg_0 = gg_coeffs[0];
    grad_grad_output[0] += -gg_0 * alpha_minus_beta * inv_ab_p2;
    grad_grad_output[1] += gg_0 * two * inv_ab_p2;

    for (int64_t k = 1; k < N; ++k) {
        C k_f(static_cast<T>(k), 0);
        C two_k_ab = two * k_f + ab;
        C two_k_ab_p2 = two_k_ab + two;

        C alpha_sq = alpha * alpha;
        C beta_sq = beta * beta;
        C A_k = (alpha_sq - beta_sq) / (two_k_ab * two_k_ab_p2);

        C k_p1 = k_f + one;
        C k_ab_p1 = k_f + ab + one;
        C B_k = (two_k_ab + one) * two_k_ab_p2 / (two * k_p1 * k_ab_p1);
        C inv_B_k = one / B_k;

        C k_alpha = k_f + alpha;
        C k_beta = k_f + beta;
        C C_k = k_alpha * k_beta * two_k_ab_p2 / (k_p1 * k_ab_p1 * two_k_ab);

        C gg_k = gg_coeffs[k];

        grad_grad_output[k - 1] += gg_k * C_k * inv_B_k;
        grad_grad_output[k] += -gg_k * A_k * inv_B_k;
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] += gg_k * inv_B_k;
        }
    }
}

} // namespace torchscience::kernel::polynomial
