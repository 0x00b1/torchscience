#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Multiply Hermite H series by x using recurrence:
// x * H_k(x) = H_{k+1}(x)/2 + k * H_{k-1}(x)
//
// Parameters:
//   output: array of size N+1
//   coeffs: input Hermite H coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (N+1)
template <typename T>
int64_t hermite_polynomial_h_mulx(T* output, const T* coeffs, int64_t N) {
    const int64_t output_size = N + 1;

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    if (N == 0) {
        return output_size;
    }

    // x * H_0 = H_1/2 (k=0 case)
    output[1] = output[1] + coeffs[0] / T(2);

    // For k >= 1: x * H_k = H_{k+1}/2 + k * H_{k-1}
    for (int64_t k = 1; k < N; ++k) {
        const T c_k = coeffs[k];
        output[k - 1] = output[k - 1] + T(k) * c_k;       // H_{k-1} contribution
        output[k + 1] = output[k + 1] + c_k / T(2);        // H_{k+1} contribution
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t hermite_polynomial_h_mulx(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    const int64_t output_size = N + 1;
    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> half(T(0.5), T(0));

    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    if (N == 0) {
        return output_size;
    }

    output[1] = output[1] + coeffs[0] * half;

    for (int64_t k = 1; k < N; ++k) {
        const c10::complex<T> c_k = coeffs[k];
        output[k - 1] = output[k - 1] + c10::complex<T>(T(k), T(0)) * c_k;
        output[k + 1] = output[k + 1] + c_k * half;
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
