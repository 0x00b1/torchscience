#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Antiderivative of Hermite He (probabilists') polynomial in coefficient space
//
// Uses integral(He_k) = He_{k+1} / (k+1)
// For series f(x) = sum_k c_k He_k(x):
//   integral(f(x)) dx = sum_k c_k He_{k+1}(x) / (k+1)
//
// Parameters:
//   output: array of size output_size to store result coefficients
//   coeffs: array of size N with input coefficients
//   N: number of input coefficients
// Returns:
//   output_size: number of output coefficients (N + 1)
template <typename T>
int64_t hermite_polynomial_he_antiderivative(
    T* output,
    const T* coeffs,
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

    // output[k+1] = coeffs[k] / (k+1)
    for (int64_t k = 0; k < N; ++k) {
        output[k + 1] = coeffs[k] / T(k + 1);
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t hermite_polynomial_he_antiderivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    const int64_t output_size = N + 1;
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    if (N == 0) {
        return output_size;
    }

    for (int64_t k = 0; k < N; ++k) {
        output[k + 1] = coeffs[k] / c10::complex<T>(T(k + 1), T(0));
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
