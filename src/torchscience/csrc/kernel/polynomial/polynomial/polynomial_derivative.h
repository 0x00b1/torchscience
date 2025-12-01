#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Polynomial derivative: d/dx (c_0 + c_1*x + ... + c_{N-1}*x^{N-1})
//   = c_1 + 2*c_2*x + ... + (N-1)*c_{N-1}*x^{N-2}
// Output has N-1 coefficients (or 1 for constant input)
//
// Parameters:
//   output: array of size max(N-1, 1)
//   coeffs: input coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (max(N-1, 1))
template <typename T>
int64_t polynomial_derivative(T* output, const T* coeffs, int64_t N) {
    if (N <= 1) {
        output[0] = T(0);
        return 1;
    }

    for (int64_t k = 1; k < N; ++k) {
        output[k - 1] = T(k) * coeffs[k];
    }
    return N - 1;
}

// Complex specialization
template <typename T>
int64_t polynomial_derivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    if (N <= 1) {
        output[0] = c10::complex<T>(T(0), T(0));
        return 1;
    }

    for (int64_t k = 1; k < N; ++k) {
        output[k - 1] = c10::complex<T>(T(k), T(0)) * coeffs[k];
    }
    return N - 1;
}

} // namespace torchscience::kernel::polynomial
