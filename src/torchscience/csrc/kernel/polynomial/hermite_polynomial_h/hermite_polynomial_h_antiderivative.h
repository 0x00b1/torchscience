#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Hermite H polynomial antiderivative:
// integral(H_k) dx = H_{k+1} / (2*(k+1))
//
// Parameters:
//   output: array of size N+1
//   coeffs: input Hermite H coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (N+1)
template <typename T>
int64_t hermite_polynomial_h_antiderivative(T* output, const T* coeffs, int64_t N) {
    const int64_t output_size = N + 1;

    // Initialize to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    if (N == 0) {
        return output_size;
    }

    // output[k+1] = coeffs[k] / (2*(k+1))
    for (int64_t k = 0; k < N; ++k) {
        output[k + 1] = coeffs[k] / T(2 * (k + 1));
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t hermite_polynomial_h_antiderivative(
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
        output[k + 1] = coeffs[k] / c10::complex<T>(T(2 * (k + 1)), T(0));
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
