#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Multiply Hermite He polynomial by x in coefficient space
//
// Uses x * He_k = He_{k+1} + k * He_{k-1}
//
// For k=0: x * He_0 = He_1 (contributes to index 1)
// For k>=1: x * He_k = He_{k+1} + k * He_{k-1}
//
// Parameters:
//   output: array of size output_size to store result coefficients
//   coeffs: array of size N with input coefficients
//   N: number of input coefficients
// Returns:
//   output_size: number of output coefficients (N + 1)
template <typename T>
int64_t hermite_polynomial_he_mulx(
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

    // x * He_0 = He_1
    output[1] = output[1] + coeffs[0];

    // For k >= 1: x * He_k = He_{k+1} + k * He_{k-1}
    for (int64_t k = 1; k < N; ++k) {
        output[k - 1] = output[k - 1] + T(k) * coeffs[k];
        output[k + 1] = output[k + 1] + coeffs[k];
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t hermite_polynomial_he_mulx(
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

    output[1] = output[1] + coeffs[0];

    for (int64_t k = 1; k < N; ++k) {
        output[k - 1] = output[k - 1] + c10::complex<T>(T(k), T(0)) * coeffs[k];
        output[k + 1] = output[k + 1] + coeffs[k];
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
