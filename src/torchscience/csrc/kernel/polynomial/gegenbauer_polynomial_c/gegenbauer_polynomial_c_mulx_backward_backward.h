#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Gegenbauer polynomial mulx
//
// This computes grad_grad_output given gg_coeffs.
// The grad_alpha from gg_alpha is handled separately.
//
// Backward: grad_coeffs[0] = grad_output[1] / (2*alpha)
//           grad_coeffs[k] = coeff_km1 * grad_output[k-1] + coeff_kp1 * grad_output[k+1]
//
// grad_grad_output[j] = sum_k gg_coeffs[k] * d(grad_coeffs[k])/d(grad_output[j])
//
// d(grad_coeffs[0])/d(grad_output[1]) = 1/(2*alpha)
// d(grad_coeffs[k])/d(grad_output[k-1]) = coeff_km1 for k >= 1
// d(grad_coeffs[k])/d(grad_output[k+1]) = coeff_kp1 for k >= 1
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: second-order gradient w.r.t. coeffs, size N
//   alpha: the Gegenbauer parameter
//   N: size of coeffs
//   output_size: size of grad_output and grad_grad_output
template <typename T>
void gegenbauer_polynomial_c_mulx_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    T alpha,
    int64_t N,
    int64_t output_size
) {
    // Initialize to zero
    for (int64_t j = 0; j < output_size; ++j) {
        grad_grad_output[j] = T(0);
    }

    if (N == 0) {
        return;
    }

    // From grad_coeffs[0] = grad_output[1] / (2*alpha):
    // grad_grad_output[1] += gg_coeffs[0] / (2*alpha)
    grad_grad_output[1] = grad_grad_output[1] + gg_coeffs[0] / (T(2) * alpha);

    // From grad_coeffs[k] = coeff_km1 * grad_output[k-1] + coeff_kp1 * grad_output[k+1]:
    for (int64_t k = 1; k < N; ++k) {
        T denom = T(2) * (T(k) + alpha);
        T coeff_km1 = (T(k) + T(2) * alpha - T(1)) / denom;
        T coeff_kp1 = T(k + 1) / denom;

        grad_grad_output[k - 1] = grad_grad_output[k - 1] + coeff_km1 * gg_coeffs[k];
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] = grad_grad_output[k + 1] + coeff_kp1 * gg_coeffs[k];
        }
    }
}

// Complex specialization
template <typename T>
void gegenbauer_polynomial_c_mulx_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    c10::complex<T> alpha,
    int64_t N,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> one(T(1), T(0));

    for (int64_t j = 0; j < output_size; ++j) {
        grad_grad_output[j] = zero;
    }

    if (N == 0) {
        return;
    }

    grad_grad_output[1] = grad_grad_output[1] + gg_coeffs[0] / (two * alpha);

    for (int64_t k = 1; k < N; ++k) {
        c10::complex<T> ck(T(k), T(0));
        c10::complex<T> denom = two * (ck + alpha);
        c10::complex<T> coeff_km1 = (ck + two * alpha - one) / denom;
        c10::complex<T> coeff_kp1 = c10::complex<T>(T(k + 1), T(0)) / denom;

        grad_grad_output[k - 1] = grad_grad_output[k - 1] + coeff_km1 * gg_coeffs[k];
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] = grad_grad_output[k + 1] + coeff_kp1 * gg_coeffs[k];
        }
    }
}

} // namespace torchscience::kernel::polynomial
