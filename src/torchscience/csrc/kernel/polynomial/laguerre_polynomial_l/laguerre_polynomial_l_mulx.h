#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Multiply Laguerre series by x using recurrence relation:
// x * L_k(x) = (2k+1)*L_k(x) - (k+1)*L_{k+1}(x) - k*L_{k-1}(x)
//
// Parameters:
//   output: array of size N+1
//   coeffs: input Laguerre coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (N+1)
template <typename T>
int64_t laguerre_polynomial_l_mulx(T* output, const T* coeffs, int64_t N) {
    const int64_t output_size = N + 1;

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    if (N == 0) {
        return output_size;
    }

    // x * L_0 = L_0 - L_1 (since k=0: (2*0+1)*L_0 - 1*L_1 - 0*L_{-1})
    output[0] = output[0] + coeffs[0];           // L_0 contribution
    output[1] = output[1] - coeffs[0];           // L_1 contribution

    // For k >= 1: x * L_k = (2k+1)*L_k - (k+1)*L_{k+1} - k*L_{k-1}
    for (int64_t k = 1; k < N; ++k) {
        const T c_k = coeffs[k];
        const T factor_k = T(2 * k + 1);
        const T factor_k1 = T(k + 1);
        const T factor_km1 = T(k);

        output[k - 1] = output[k - 1] - factor_km1 * c_k;   // L_{k-1} contribution
        output[k] = output[k] + factor_k * c_k;              // L_k contribution
        output[k + 1] = output[k + 1] - factor_k1 * c_k;     // L_{k+1} contribution
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t laguerre_polynomial_l_mulx(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    const int64_t output_size = N + 1;
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    if (N == 0) {
        return output_size;
    }

    // x * L_0 = L_0 - L_1
    output[0] = output[0] + coeffs[0];
    output[1] = output[1] - coeffs[0];

    // For k >= 1
    for (int64_t k = 1; k < N; ++k) {
        const c10::complex<T> c_k = coeffs[k];
        const c10::complex<T> factor_k(T(2 * k + 1), T(0));
        const c10::complex<T> factor_k1(T(k + 1), T(0));
        const c10::complex<T> factor_km1(T(k), T(0));

        output[k - 1] = output[k - 1] - factor_km1 * c_k;
        output[k] = output[k] + factor_k * c_k;
        output[k + 1] = output[k + 1] - factor_k1 * c_k;
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
