#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Derivative of Hermite He (probabilists') polynomial in coefficient space
//
// Uses d/dx He_n = n * He_{n-1}
// For series f(x) = sum_k c_k He_k(x):
//   f'(x) = sum_j ((j+1) * c_{j+1}) He_j(x)
//
// Parameters:
//   output: array of size output_size to store result coefficients
//   coeffs: array of size N with input coefficients
//   N: number of input coefficients
// Returns:
//   output_size: number of output coefficients
template <typename T>
int64_t hermite_polynomial_he_derivative(
    T* output,
    const T* coeffs,
    int64_t N
) {
    if (N <= 1) {
        output[0] = T(0);
        return 1;
    }

    const int64_t output_size = N - 1;

    // output[j] = (j+1) * coeffs[j+1] for j = 0, ..., N-2
    for (int64_t j = 0; j < output_size; ++j) {
        output[j] = T(j + 1) * coeffs[j + 1];
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t hermite_polynomial_he_derivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    if (N <= 1) {
        output[0] = c10::complex<T>(T(0), T(0));
        return 1;
    }

    const int64_t output_size = N - 1;

    for (int64_t j = 0; j < output_size; ++j) {
        output[j] = c10::complex<T>(T(j + 1), T(0)) * coeffs[j + 1];
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
