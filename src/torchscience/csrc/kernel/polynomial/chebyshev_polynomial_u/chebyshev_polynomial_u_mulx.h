#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Multiply a Chebyshev U polynomial by x (shift operation).
//
// Using the identity: x * U_n(x) = 0.5 * (U_{n+1}(x) + U_{n-1}(x))  for n >= 1
// And: x * U_0(x) = 0.5 * U_1(x) + 0.5 * T_1(x) = 0.5 * U_1(x) + 0.5 * x
//
// However, to stay within the Chebyshev U basis, we use:
// x * U_n(x) = 0.5 * (U_{n+1}(x) + U_{n-1}(x))  for n >= 1
// x * U_0(x) = 0.5 * U_1(x) + 0.5 (since U_0 = 1 and x*1 = x, and x = 0.5*U_1 + 0.5*U_{-1})
//
// For the Chebyshev U basis, we have U_{-1}(x) = 0, so:
// x * U_0(x) = 0.5 * U_1(x)
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
int64_t chebyshev_polynomial_u_mulx(
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

    // c_0 * U_0: x * c_0 * U_0 = 0.5 * c_0 * U_1
    output[1] += T(0.5) * coeffs[0];

    // c_k * U_k for k >= 1: x * c_k * U_k = 0.5 * c_k * (U_{k+1} + U_{k-1})
    for (int64_t k = 1; k < N; ++k) {
        output[k + 1] += T(0.5) * coeffs[k];
        output[k - 1] += T(0.5) * coeffs[k];
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_u_mulx(
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

    // c_0 * U_0: x * c_0 * U_0 = 0.5 * c_0 * U_1
    output[1] += half * coeffs[0];

    // c_k * U_k for k >= 1: x * c_k * U_k = 0.5 * c_k * (U_{k+1} + U_{k-1})
    for (int64_t k = 1; k < N; ++k) {
        output[k + 1] += half * coeffs[k];
        output[k - 1] += half * coeffs[k];
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
