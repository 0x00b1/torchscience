#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Legendre polynomial derivative using recurrence relation:
// P'_n = (2n-1)*P_{n-1} + P'_{n-2}
//
// Algorithm (matching numpy's legder):
//   der[j-1] = (2j-1)*c[j], c[j-2] += c[j]  for j = n, n-1, ..., 3
//   der[1] = 3*c[2]
//   der[0] = c[1]
//
// Parameters:
//   output: array of size max(N-1, 1)
//   coeffs: input Legendre coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (max(N-1, 1))
template <typename T>
int64_t legendre_polynomial_p_derivative(T* output, const T* coeffs, int64_t N) {
    if (N <= 1) {
        output[0] = T(0);
        return 1;
    }

    const int64_t output_size = N - 1;

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    // Copy coeffs so we can modify during accumulation
    T* tmp = new T[N];
    for (int64_t k = 0; k < N; ++k) {
        tmp[k] = coeffs[k];
    }

    // Backward recurrence: for j from N-1 down to 3
    // der[j-1] = (2j-1) * c[j]
    // c[j-2] += c[j]
    for (int64_t j = N - 1; j >= 3; --j) {
        output[j - 1] = T(2 * j - 1) * tmp[j];
        tmp[j - 2] = tmp[j - 2] + tmp[j];
    }

    // Handle j=2: der[1] = 3 * c[2]
    if (output_size > 1) {
        output[1] = T(3) * tmp[2];
    }

    // Handle j=1: der[0] = c[1]
    output[0] = tmp[1];

    delete[] tmp;
    return output_size;
}

// Complex specialization
template <typename T>
int64_t legendre_polynomial_p_derivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    if (N <= 1) {
        output[0] = c10::complex<T>(T(0), T(0));
        return 1;
    }

    const int64_t output_size = N - 1;
    const c10::complex<T> zero(T(0), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    // Copy coeffs so we can modify during accumulation
    c10::complex<T>* tmp = new c10::complex<T>[N];
    for (int64_t k = 0; k < N; ++k) {
        tmp[k] = coeffs[k];
    }

    // Backward recurrence
    for (int64_t j = N - 1; j >= 3; --j) {
        output[j - 1] = c10::complex<T>(T(2 * j - 1), T(0)) * tmp[j];
        tmp[j - 2] = tmp[j - 2] + tmp[j];
    }

    if (output_size > 1) {
        output[1] = c10::complex<T>(T(3), T(0)) * tmp[2];
    }

    output[0] = tmp[1];

    delete[] tmp;
    return output_size;
}

} // namespace torchscience::kernel::polynomial
