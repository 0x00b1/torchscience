#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Legendre polynomial antiderivative using the formula:
// integral(P_k(x)) = (P_{k+1}(x) - P_{k-1}(x)) / (2k+1)  for k >= 1
// integral(P_0(x)) = x = P_1(x)
//
// Note: This computes the antiderivative with constant=0 at x=0.
// The caller should adjust the constant term if needed.
//
// Parameters:
//   output: array of size N+1
//   coeffs: input Legendre coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (N+1)
template <typename T>
int64_t legendre_polynomial_p_antiderivative(T* output, const T* coeffs, int64_t N) {
    const int64_t output_size = N + 1;

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    if (N == 0) {
        return output_size;
    }

    // k=0 term: integral(P_0) = P_1
    output[1] = output[1] + coeffs[0];

    // k>=1 terms: integral(P_k) = (P_{k+1} - P_{k-1}) / (2k+1)
    for (int64_t k = 1; k < N; ++k) {
        const T factor = T(1) / T(2 * k + 1);
        // Contribution to P_{k+1}
        output[k + 1] = output[k + 1] + coeffs[k] * factor;
        // Contribution to P_{k-1}
        output[k - 1] = output[k - 1] - coeffs[k] * factor;
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t legendre_polynomial_p_antiderivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    const int64_t output_size = N + 1;
    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> one(T(1), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    if (N == 0) {
        return output_size;
    }

    // k=0 term: integral(P_0) = P_1
    output[1] = output[1] + coeffs[0];

    // k>=1 terms
    for (int64_t k = 1; k < N; ++k) {
        const c10::complex<T> factor = one / c10::complex<T>(T(2 * k + 1), T(0));
        output[k + 1] = output[k + 1] + coeffs[k] * factor;
        output[k - 1] = output[k - 1] - coeffs[k] * factor;
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
