#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Multiply a Chebyshev V polynomial by x (shift operation).
//
// For Chebyshev V polynomials:
// V_n(x) = 2x*V_{n-1}(x) - V_{n-2}(x) with V_0(x) = 1, V_1(x) = 2x - 1
//
// Rearranging: x*V_{n-1}(x) = (V_n(x) + V_{n-2}(x))/2
// Or: x*V_n(x) = (V_{n+1}(x) + V_{n-1}(x))/2  for n >= 1
//
// For V_0: x*V_0(x) = x = (V_1(x) + 1)/2 = (V_1(x) + V_0(x))/2 + (1 - V_0(x))/2
// Actually: x = (V_1(x) + 1)/2, but V_0(x) = 1, so x*V_0(x) = (V_1(x) + 1)/2
// In coefficient form: x*V_0 contributes 0.5 to c_0 and 0.5 to c_1
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
int64_t chebyshev_polynomial_v_mulx(
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

    // c_0 * V_0: x * c_0 * V_0 = c_0 * (V_1 + 1)/2
    // In V basis: contributes 0.5*c_0 to V_0 and 0.5*c_0 to V_1
    output[0] += T(0.5) * coeffs[0];
    output[1] += T(0.5) * coeffs[0];

    // c_k * V_k for k >= 1: x * c_k * V_k = 0.5 * c_k * (V_{k+1} + V_{k-1})
    for (int64_t k = 1; k < N; ++k) {
        output[k + 1] += T(0.5) * coeffs[k];
        output[k - 1] += T(0.5) * coeffs[k];
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_v_mulx(
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

    // c_0 * V_0: x * c_0 * V_0 = c_0 * (V_1 + 1)/2
    output[0] += half * coeffs[0];
    output[1] += half * coeffs[0];

    // c_k * V_k for k >= 1: x * c_k * V_k = 0.5 * c_k * (V_{k+1} + V_{k-1})
    for (int64_t k = 1; k < N; ++k) {
        output[k + 1] += half * coeffs[k];
        output[k - 1] += half * coeffs[k];
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
