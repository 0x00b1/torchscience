#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Chebyshev V polynomial derivative using recurrence relation:
//
// V_n(x) = 2x*V_{n-1}(x) - V_{n-2}(x) with V_0(x) = 1, V_1(x) = 2x - 1
//
// The derivative relationship for Chebyshev V polynomials:
// d/dx[V_n(x)] = 2*n*U_{n-1}(x)  where U is the Chebyshev polynomial of second kind
//
// Since V_n(x) = T_n(x) + T_{n-1}(x) for n >= 1, and derivative relationships
// between Chebyshev polynomials, we can express the derivative in terms of V basis.
//
// For coefficient representation, using the backward recurrence:
//   d_{n-1} = 2*n*c_n
//   d_k = d_{k+2} + 2*(k+1)*c_{k+1}  for k = n-2, ..., 1
//   d_0 = d_2 + 2*c_1
//
// This computes the Chebyshev V coefficients of the derivative polynomial.
//
// Parameters:
//   output: array of size max(N-1, 1)
//   coeffs: input Chebyshev V coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (max(N-1, 1))
template <typename T>
int64_t chebyshev_polynomial_v_derivative(T* output, const T* coeffs, int64_t N) {
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

    // Backward recurrence for Chebyshev V derivative
    // d_{deg-1} = 2 * deg * c_{deg}
    if (deg >= 1) {
        output[deg - 1] = T(2 * deg) * coeffs[deg];
    }

    // d_k = d_{k+2} + 2*(k+1)*c_{k+1} for k = deg-2 down to 1
    for (int64_t k = deg - 2; k >= 1; --k) {
        T d_k = T(2 * (k + 1)) * coeffs[k + 1];
        if (k + 2 < output_size) {
            d_k = d_k + output[k + 2];
        }
        output[k] = d_k;
    }

    // d_0 = 2*c_1 + d_2 (special case for k=0)
    if (deg >= 1) {
        output[0] = T(2) * coeffs[1];
        if (output_size >= 3) {
            output[0] = output[0] + output[2];
        }
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_v_derivative(
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

    // Backward recurrence for Chebyshev V derivative
    if (deg >= 1) {
        output[deg - 1] = c10::complex<T>(T(2 * deg), T(0)) * coeffs[deg];
    }

    for (int64_t k = deg - 2; k >= 1; --k) {
        c10::complex<T> d_k = c10::complex<T>(T(2 * (k + 1)), T(0)) * coeffs[k + 1];
        if (k + 2 < output_size) {
            d_k = d_k + output[k + 2];
        }
        output[k] = d_k;
    }

    if (deg >= 1) {
        output[0] = c10::complex<T>(T(2), T(0)) * coeffs[1];
        if (output_size >= 3) {
            output[0] = output[0] + output[2];
        }
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
