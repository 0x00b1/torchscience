#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward for Laguerre polynomial derivative
//
// Forward: output[j] = -sum_{k=j+1}^{N-1} coeffs[k]
// So: d(output[j])/d(coeffs[k]) = -1 for k > j, 0 otherwise
//
// grad_coeffs[k] = sum_j grad_output[j] * d(output[j])/d(coeffs[k])
//                = sum_{j=0}^{k-1} grad_output[j] * (-1)
//                = -sum_{j=0}^{k-1} grad_output[j]
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coeffs, size N
//   grad_output: incoming gradient, size output_size
//   N: size of coeffs
//   output_size: size of grad_output
template <typename T>
void laguerre_polynomial_l_derivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    int64_t N,
    int64_t output_size
) {
    (void)output_size;

    // Initialize grad_coeffs to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = T(0);
    }

    if (N <= 1) {
        return;
    }

    // grad_coeffs[k] = -sum_{j=0}^{k-1} grad_output[j]
    // grad_coeffs[0] = 0 (no j < 0)
    // grad_coeffs[1] = -grad_output[0]
    // grad_coeffs[2] = -(grad_output[0] + grad_output[1])
    // etc.
    T cumsum = T(0);
    for (int64_t k = 1; k < N; ++k) {
        cumsum = cumsum + grad_output[k - 1];
        grad_coeffs[k] = -cumsum;
    }
}

// Complex specialization
template <typename T>
void laguerre_polynomial_l_derivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    int64_t N,
    int64_t output_size
) {
    (void)output_size;
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = zero;
    }

    if (N <= 1) {
        return;
    }

    c10::complex<T> cumsum = zero;
    for (int64_t k = 1; k < N; ++k) {
        cumsum = cumsum + grad_output[k - 1];
        grad_coeffs[k] = -cumsum;
    }
}

} // namespace torchscience::kernel::polynomial
