#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Multiply Legendre series by x using recurrence relation:
// x * P_k(x) = [(k+1)*P_{k+1}(x) + k*P_{k-1}(x)] / (2k+1)
//
// Parameters:
//   output: array of size N+1
//   coeffs: input Legendre coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (N+1)
template <typename T>
int64_t legendre_polynomial_p_mulx(T* output, const T* coeffs, int64_t N) {
    const int64_t output_size = N + 1;

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    if (N == 0) {
        return output_size;
    }

    // x * P_0 = P_1: contributes coeffs[0] to output[1]
    output[1] = output[1] + coeffs[0];

    // For k >= 1: x * P_k = [(k+1)*P_{k+1} + k*P_{k-1}] / (2k+1)
    for (int64_t k = 1; k < N; ++k) {
        const T c_k = coeffs[k];
        const T denom = T(2 * k + 1);

        // Contribution to P_{k-1}
        output[k - 1] = output[k - 1] + (T(k) / denom) * c_k;
        // Contribution to P_{k+1}
        output[k + 1] = output[k + 1] + (T(k + 1) / denom) * c_k;
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t legendre_polynomial_p_mulx(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    const int64_t output_size = N + 1;
    const c10::complex<T> zero(T(0), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    if (N == 0) {
        return output_size;
    }

    // x * P_0 = P_1
    output[1] = output[1] + coeffs[0];

    // For k >= 1
    for (int64_t k = 1; k < N; ++k) {
        const c10::complex<T> c_k = coeffs[k];
        const c10::complex<T> denom(T(2 * k + 1), T(0));

        output[k - 1] = output[k - 1] + (c10::complex<T>(T(k), T(0)) / denom) * c_k;
        output[k + 1] = output[k + 1] + (c10::complex<T>(T(k + 1), T(0)) / denom) * c_k;
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
