#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Polynomial antiderivative (indefinite integral):
// Given p(x) = c_0 + c_1*x + ... + c_{N-1}*x^{N-1}
// The antiderivative is:
// P(x) = C + c_0*x + c_1*x^2/2 + c_2*x^3/3 + ... + c_{N-1}*x^N/N
//
// Output has N+1 coefficients: [C, c_0, c_1/2, c_2/3, ..., c_{N-1}/N]
//
// Parameters:
//   output: array of size N+1
//   coeffs: input coefficients, size N
//   constant: the integration constant C
//   N: number of input coefficients
template <typename T>
void polynomial_antiderivative(
    T* output,
    const T* coeffs,
    T constant,
    int64_t N
) {
    output[0] = constant;
    for (int64_t k = 0; k < N; ++k) {
        output[k + 1] = coeffs[k] / T(k + 1);
    }
}

// Complex specialization
template <typename T>
void polynomial_antiderivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    c10::complex<T> constant,
    int64_t N
) {
    output[0] = constant;
    for (int64_t k = 0; k < N; ++k) {
        output[k + 1] = coeffs[k] / c10::complex<T>(T(k + 1), T(0));
    }
}

} // namespace torchscience::kernel::polynomial
