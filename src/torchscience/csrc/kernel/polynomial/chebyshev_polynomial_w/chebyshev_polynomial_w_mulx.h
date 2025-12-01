#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Multiply a Chebyshev W polynomial by x (shift operation).
//
// Chebyshev W polynomials satisfy: W_n(x) = 2x*W_{n-1}(x) - W_{n-2}(x)
// with W_0(x) = 1 and W_1(x) = 2x + 1.
//
// From the recurrence relation, we can derive:
//   x * W_n(x) = 0.5 * (W_{n+1}(x) + W_{n-1}(x))  for n >= 1
//   x * W_0(x) = 0.5 * (W_1(x) - 1) = 0.5 * W_1(x) - 0.5
//
// However, for consistency with the coefficient representation:
//   x * W_0(x) = x * 1 = 0.5 * (W_1(x) + W_{-1}(x)) where W_{-1}(x) = -1
//              = 0.5 * (W_1(x) - 1)
//
// Actually, let's derive more carefully:
// From W_1(x) = 2x + 1, we get x = (W_1(x) - 1) / 2
// So x * W_0(x) = (W_1(x) - 1) / 2 = 0.5 * W_1(x) - 0.5 * W_0(x)
//
// For n >= 1: From W_{n+1}(x) = 2x*W_n(x) - W_{n-1}(x)
//            x * W_n(x) = 0.5 * (W_{n+1}(x) + W_{n-1}(x))
//
// Given coefficients c[0..N-1], multiply by x gives coefficients m[0..N].
//
// Parameters:
//   output: array of size N + 1 (initialized to zero by this function)
//   coeffs: polynomial coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (N + 1, or 1 if N == 0)
template <typename T>
int64_t chebyshev_polynomial_w_mulx(
    T* output,
    const T* coeffs,
    int64_t N
) {
    if (N == 0) {
        output[0] = T(0);
        return 1;
    }

    const int64_t output_size = N + 1;

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    // c_0 * W_0: x * c_0 * W_0 = c_0 * (0.5 * W_1 - 0.5 * W_0)
    //                         = -0.5 * c_0 * W_0 + 0.5 * c_0 * W_1
    output[0] -= T(0.5) * coeffs[0];
    output[1] += T(0.5) * coeffs[0];

    // c_k * W_k for k >= 1: x * c_k * W_k = 0.5 * c_k * (W_{k+1} + W_{k-1})
    for (int64_t k = 1; k < N; ++k) {
        output[k + 1] += T(0.5) * coeffs[k];
        output[k - 1] += T(0.5) * coeffs[k];
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_w_mulx(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    if (N == 0) {
        output[0] = c10::complex<T>(T(0), T(0));
        return 1;
    }

    const int64_t output_size = N + 1;
    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> half(T(0.5), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    // c_0 * W_0: x * c_0 * W_0 = c_0 * (0.5 * W_1 - 0.5 * W_0)
    output[0] -= half * coeffs[0];
    output[1] += half * coeffs[0];

    // c_k * W_k for k >= 1: x * c_k * W_k = 0.5 * c_k * (W_{k+1} + W_{k-1})
    for (int64_t k = 1; k < N; ++k) {
        output[k + 1] += half * coeffs[k];
        output[k - 1] += half * coeffs[k];
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
