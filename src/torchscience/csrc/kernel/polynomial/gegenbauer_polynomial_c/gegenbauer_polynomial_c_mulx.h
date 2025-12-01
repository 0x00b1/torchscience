#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Multiply Gegenbauer polynomial by x in coefficient space
//
// Uses x * C_k^{alpha} = ((k+1)/(2*(k+alpha))) * C_{k+1}^{alpha}
//                      + ((k+2*alpha-1)/(2*(k+alpha))) * C_{k-1}^{alpha}
//
// For k=0: x * C_0 = (1/(2*alpha)) * C_1
//
// Parameters:
//   output: array of size output_size to store result coefficients
//   coeffs: array of size N with input coefficients
//   alpha: the Gegenbauer parameter (lambda)
//   N: number of input coefficients
// Returns:
//   output_size: number of output coefficients (N + 1)
template <typename T>
int64_t gegenbauer_polynomial_c_mulx(
    T* output,
    const T* coeffs,
    T alpha,
    int64_t N
) {
    const int64_t output_size = N + 1;

    // Initialize to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    if (N == 0) {
        return output_size;
    }

    // x * C_0 = (1/(2*alpha)) * C_1
    output[1] = output[1] + coeffs[0] / (T(2) * alpha);

    // For k >= 1: x * C_k = ((k+1)/(2*(k+alpha))) * C_{k+1}
    //                     + ((k+2*alpha-1)/(2*(k+alpha))) * C_{k-1}
    for (int64_t k = 1; k < N; ++k) {
        T denom = T(2) * (T(k) + alpha);
        T coeff_kp1 = T(k + 1) / denom;
        T coeff_km1 = (T(k) + T(2) * alpha - T(1)) / denom;

        output[k - 1] = output[k - 1] + coeff_km1 * coeffs[k];
        output[k + 1] = output[k + 1] + coeff_kp1 * coeffs[k];
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t gegenbauer_polynomial_c_mulx(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    c10::complex<T> alpha,
    int64_t N
) {
    const int64_t output_size = N + 1;
    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> one(T(1), T(0));

    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    if (N == 0) {
        return output_size;
    }

    output[1] = output[1] + coeffs[0] / (two * alpha);

    for (int64_t k = 1; k < N; ++k) {
        c10::complex<T> ck(T(k), T(0));
        c10::complex<T> denom = two * (ck + alpha);
        c10::complex<T> coeff_kp1 = c10::complex<T>(T(k + 1), T(0)) / denom;
        c10::complex<T> coeff_km1 = (ck + two * alpha - one) / denom;

        output[k - 1] = output[k - 1] + coeff_km1 * coeffs[k];
        output[k + 1] = output[k + 1] + coeff_kp1 * coeffs[k];
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
