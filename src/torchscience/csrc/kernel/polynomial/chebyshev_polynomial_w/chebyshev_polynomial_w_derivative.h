#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Chebyshev W polynomial derivative using recurrence relation.
//
// For Chebyshev W polynomial (fourth kind) with coefficients c[0..N-1],
// the derivative has coefficients d[0..N-2] computed via backward recurrence.
//
// The Chebyshev W polynomials satisfy: W_n(x) = 2x*W_{n-1}(x) - W_{n-2}(x)
// with W_0(x) = 1 and W_1(x) = 2x + 1.
//
// The derivative formula for Chebyshev W polynomials:
//   d/dx[W_n(x)] = 2*n*U_{n-1}(x) + 2*sum_{k=0}^{n-1}(n-k)*U_k(x) for even n-k
//
// However, to express the derivative in the W basis, we use the recurrence:
//   d_{n-1} = 2*n*c_n
//   d_k = d_{k+2} + 2*(k+1)*c_{k+1}  for k = n-2, ..., 1
//   d_0 = 0.5*d_2 + c_1
//
// This computes the Chebyshev W coefficients of the derivative polynomial.
//
// Parameters:
//   output: array of size max(N-1, 1)
//   coeffs: input Chebyshev W coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (max(N-1, 1))
template <typename T>
int64_t chebyshev_polynomial_w_derivative(T* output, const T* coeffs, int64_t N) {
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

    // Backward recurrence for Chebyshev W derivative
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

    // d_0 = c_1 + 0.5*d_2 (special case for k=0)
    if (deg >= 1) {
        output[0] = coeffs[1];
        if (output_size >= 3) {
            output[0] = output[0] + T(0.5) * output[2];
        }
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_w_derivative(
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
    const c10::complex<T> half(T(0.5), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    // Backward recurrence for Chebyshev W derivative
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
        output[0] = coeffs[1];
        if (output_size >= 3) {
            output[0] = output[0] + half * output[2];
        }
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
