#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Hermite H polynomial derivative using:
// d/dx H_n = 2n * H_{n-1}
//
// For f(x) = sum c_k H_k, the derivative is:
// f'(x) = sum c_k * 2k * H_{k-1}
//       = sum_{j=0}^{n-2} (2*(j+1) * c_{j+1}) * H_j
//
// Parameters:
//   output: array of size max(N-1, 1)
//   coeffs: input Hermite H coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (max(N-1, 1))
template <typename T>
int64_t hermite_polynomial_h_derivative(T* output, const T* coeffs, int64_t N) {
    if (N <= 1) {
        output[0] = T(0);
        return 1;
    }

    const int64_t output_size = N - 1;

    // output[j] = 2*(j+1) * coeffs[j+1]
    for (int64_t j = 0; j < output_size; ++j) {
        output[j] = T(2 * (j + 1)) * coeffs[j + 1];
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t hermite_polynomial_h_derivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    if (N <= 1) {
        output[0] = c10::complex<T>(T(0), T(0));
        return 1;
    }

    const int64_t output_size = N - 1;

    for (int64_t j = 0; j < output_size; ++j) {
        output[j] = c10::complex<T>(T(2 * (j + 1)), T(0)) * coeffs[j + 1];
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
