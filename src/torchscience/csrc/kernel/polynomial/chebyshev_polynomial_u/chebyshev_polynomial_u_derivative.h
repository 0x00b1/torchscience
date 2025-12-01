#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Chebyshev U polynomial derivative.
//
// The derivative of U_n(x) is: d/dx U_n(x) = (n+1) * U_{n-1}(x) / (2*x) + ...
// Actually, the cleaner relation is:
//   d/dx U_n(x) = ((n+1)*T_{n+1}(x) - x*U_n(x)) / (x^2 - 1)
//
// For coefficient-based differentiation, we use the relation:
//   If p(x) = sum_{k=0}^{N-1} c_k * U_k(x), then
//   p'(x) = sum_{k=0}^{N-2} d_k * U_k(x)
//
// The recurrence for Chebyshev U derivative coefficients:
//   d_{n-1} = 2*n*c_n
//   d_k = d_{k+2} + 2*(k+1)*c_{k+1}  for k = n-2, ..., 0
//
// This is similar to Chebyshev T but without the special case for d_0.
//
// Parameters:
//   output: array of size max(N-1, 1)
//   coeffs: input Chebyshev U coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (max(N-1, 1))
template <typename T>
int64_t chebyshev_polynomial_u_derivative(T* output, const T* coeffs, int64_t N) {
    if (N <= 1) {
        output[0] = T(0);
        return 1;
    }

    const int64_t deg = N - 1;  // degree of input polynomial
    const int64_t output_size = deg;  // derivative has degree deg-1, so deg coefficients

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    // Backward recurrence for Chebyshev U derivative
    // d_{deg-1} = 2 * deg * c_{deg}
    if (deg >= 1) {
        output[deg - 1] = T(2 * deg) * coeffs[deg];
    }

    // d_k = d_{k+2} + 2*(k+1)*c_{k+1} for k = deg-2 down to 0
    for (int64_t k = deg - 2; k >= 0; --k) {
        T d_k = T(2 * (k + 1)) * coeffs[k + 1];
        if (k + 2 < output_size) {
            d_k = d_k + output[k + 2];
        }
        output[k] = d_k;
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_u_derivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    if (N <= 1) {
        output[0] = c10::complex<T>(T(0), T(0));
        return 1;
    }

    const int64_t deg = N - 1;
    const int64_t output_size = deg;

    const c10::complex<T> zero(T(0), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    // Backward recurrence for Chebyshev U derivative
    if (deg >= 1) {
        output[deg - 1] = c10::complex<T>(T(2 * deg), T(0)) * coeffs[deg];
    }

    for (int64_t k = deg - 2; k >= 0; --k) {
        c10::complex<T> d_k = c10::complex<T>(T(2 * (k + 1)), T(0)) * coeffs[k + 1];
        if (k + 2 < output_size) {
            d_k = d_k + output[k + 2];
        }
        output[k] = d_k;
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
