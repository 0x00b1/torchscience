#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward for Gegenbauer polynomial mulx
//
// Forward: x * C_k^{alpha} = ((k+1)/(2*(k+alpha))) * C_{k+1}
//                          + ((k+2*alpha-1)/(2*(k+alpha))) * C_{k-1}
//
// Gradients w.r.t. coeffs:
//   grad_coeffs[0] = grad_output[1] / (2*alpha)
//   grad_coeffs[k] = ((k+2*alpha-1)/(2*(k+alpha))) * grad_output[k-1]
//                  + ((k+1)/(2*(k+alpha))) * grad_output[k+1]  for k >= 1
//
// Gradient w.r.t. alpha (accumulated over all k):
//   d(output[1])/d(alpha) = -coeffs[0] / (2*alpha^2)
//   For k >= 1:
//     d(coeff_km1)/d(alpha) = (k+1) / (2*(k+alpha))^2
//     d(coeff_kp1)/d(alpha) = -(k+1) / (2*(k+alpha))^2
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coeffs, size N
//   grad_alpha: output gradient w.r.t. alpha (scalar, accumulated)
//   grad_output: incoming gradient, size output_size
//   coeffs: original input coefficients
//   alpha: the Gegenbauer parameter
//   N: size of coeffs
//   output_size: size of grad_output
template <typename T>
void gegenbauer_polynomial_c_mulx_backward(
    T* grad_coeffs,
    T* grad_alpha,
    const T* grad_output,
    const T* coeffs,
    T alpha,
    int64_t N,
    int64_t output_size
) {
    (void)output_size;

    *grad_alpha = T(0);

    if (N == 0) {
        return;
    }

    // k = 0: grad_coeffs[0] = grad_output[1] / (2*alpha)
    grad_coeffs[0] = grad_output[1] / (T(2) * alpha);

    // grad_alpha contribution from k = 0: -coeffs[0] / (2*alpha^2) * grad_output[1]
    *grad_alpha = *grad_alpha - coeffs[0] / (T(2) * alpha * alpha) * grad_output[1];

    // k >= 1
    for (int64_t k = 1; k < N; ++k) {
        T denom = T(2) * (T(k) + alpha);
        T coeff_km1 = (T(k) + T(2) * alpha - T(1)) / denom;
        T coeff_kp1 = T(k + 1) / denom;

        grad_coeffs[k] = coeff_km1 * grad_output[k - 1] + coeff_kp1 * grad_output[k + 1];

        // grad_alpha contributions
        // d(coeff_km1)/d(alpha) = (k+1) / (2*(k+alpha))^2
        // d(coeff_kp1)/d(alpha) = -(k+1) / (2*(k+alpha))^2
        T denom_sq = denom * denom;
        T d_coeff_km1_d_alpha = T(k + 1) / denom_sq;
        T d_coeff_kp1_d_alpha = -T(k + 1) / denom_sq;

        *grad_alpha = *grad_alpha + (d_coeff_km1_d_alpha * coeffs[k] * grad_output[k - 1]
                                   + d_coeff_kp1_d_alpha * coeffs[k] * grad_output[k + 1]);
    }
}

// Complex specialization
template <typename T>
void gegenbauer_polynomial_c_mulx_backward(
    c10::complex<T>* grad_coeffs,
    c10::complex<T>* grad_alpha,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    c10::complex<T> alpha,
    int64_t N,
    int64_t output_size
) {
    (void)output_size;
    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> one(T(1), T(0));

    *grad_alpha = zero;

    if (N == 0) {
        return;
    }

    grad_coeffs[0] = grad_output[1] / (two * alpha);
    *grad_alpha = *grad_alpha - coeffs[0] / (two * alpha * alpha) * grad_output[1];

    for (int64_t k = 1; k < N; ++k) {
        c10::complex<T> ck(T(k), T(0));
        c10::complex<T> denom = two * (ck + alpha);
        c10::complex<T> coeff_km1 = (ck + two * alpha - one) / denom;
        c10::complex<T> coeff_kp1 = c10::complex<T>(T(k + 1), T(0)) / denom;

        grad_coeffs[k] = coeff_km1 * grad_output[k - 1] + coeff_kp1 * grad_output[k + 1];

        c10::complex<T> denom_sq = denom * denom;
        c10::complex<T> d_coeff_km1_d_alpha = c10::complex<T>(T(k + 1), T(0)) / denom_sq;
        c10::complex<T> d_coeff_kp1_d_alpha = c10::complex<T>(T(-k - 1), T(0)) / denom_sq;

        *grad_alpha = *grad_alpha + (d_coeff_km1_d_alpha * coeffs[k] * grad_output[k - 1]
                                   + d_coeff_kp1_d_alpha * coeffs[k] * grad_output[k + 1]);
    }
}

} // namespace torchscience::kernel::polynomial
