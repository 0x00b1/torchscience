#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Laguerre polynomial derivative using recurrence relation:
// d/dx L_n = -sum_{k=0}^{n-1} L_k
//
// For a series f(x) = sum c_k L_k, the derivative is:
// f'(x) = sum c_k * (-sum_{j=0}^{k-1} L_j)
//       = -sum_j (sum_{k=j+1}^{n-1} c_k) L_j
//
// So output[j] = -sum_{k=j+1}^{n-1} c_k
//
// Parameters:
//   output: array of size max(N-1, 1)
//   coeffs: input Laguerre coefficients, size N
//   N: number of input coefficients
//
// Returns: size of output (max(N-1, 1))
template <typename T>
int64_t laguerre_polynomial_l_derivative(T* output, const T* coeffs, int64_t N) {
    if (N <= 1) {
        output[0] = T(0);
        return 1;
    }

    const int64_t output_size = N - 1;

    // Compute cumulative sum from right to left
    // output[j] = -sum_{k=j+1}^{N-1} coeffs[k]
    T cumsum = T(0);
    for (int64_t k = N - 1; k >= 1; --k) {
        cumsum = cumsum + coeffs[k];
        output[k - 1] = -cumsum;
    }

    return output_size;
}

// Complex specialization
template <typename T>
int64_t laguerre_polynomial_l_derivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    if (N <= 1) {
        output[0] = c10::complex<T>(T(0), T(0));
        return 1;
    }

    const int64_t output_size = N - 1;
    c10::complex<T> cumsum(T(0), T(0));

    for (int64_t k = N - 1; k >= 1; --k) {
        cumsum = cumsum + coeffs[k];
        output[k - 1] = -cumsum;
    }

    return output_size;
}

} // namespace torchscience::kernel::polynomial
