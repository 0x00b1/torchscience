#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Horner's method for polynomial evaluation
// Evaluates p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_{N-1}*x^{N-1}
// Using: b_{N-1} = c_{N-1}, b_k = c_k + x * b_{k+1}
//
// Parameters:
//   coeffs: pointer to N coefficients [c_0, c_1, ..., c_{N-1}]
//   x: evaluation point
//   N: number of coefficients (degree + 1)
//
// Returns: p(x)
template <typename T>
T polynomial_evaluate(const T* coeffs, T x, int64_t N) {
    if (N == 0) {
        return T(0);
    }

    T result = coeffs[N - 1];
    for (int64_t k = N - 2; k >= 0; --k) {
        result = result * x + coeffs[k];
    }
    return result;
}

// Complex specialization - same algorithm works for complex types
template <typename T>
c10::complex<T> polynomial_evaluate(
    const c10::complex<T>* coeffs,
    c10::complex<T> x,
    int64_t N
) {
    if (N == 0) {
        return c10::complex<T>(T(0), T(0));
    }

    c10::complex<T> result = coeffs[N - 1];
    for (int64_t k = N - 2; k >= 0; --k) {
        result = result * x + coeffs[k];
    }
    return result;
}

} // namespace torchscience::kernel::polynomial
