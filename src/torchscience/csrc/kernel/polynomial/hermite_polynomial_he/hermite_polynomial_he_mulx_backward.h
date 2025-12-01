#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward for Hermite He polynomial mulx
//
// Forward: x * He_k = He_{k+1} + k * He_{k-1}
// output[1] += coeffs[0]
// output[k-1] += k * coeffs[k] for k >= 1
// output[k+1] += coeffs[k] for k >= 1
//
// Gradients:
// grad_coeffs[0] = grad_output[1]
// grad_coeffs[k] = k * grad_output[k-1] + grad_output[k+1] for k >= 1
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coeffs, size N
//   grad_output: incoming gradient, size output_size
//   coeffs: original input coefficients (unused)
//   N: size of coeffs
//   output_size: size of grad_output
template <typename T>
void hermite_polynomial_he_mulx_backward(
    T* grad_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;
    (void)output_size;

    if (N == 0) {
        return;
    }

    // k = 0: grad_coeffs[0] = grad_output[1]
    grad_coeffs[0] = grad_output[1];

    // k >= 1: grad_coeffs[k] = k * grad_output[k-1] + grad_output[k+1]
    for (int64_t k = 1; k < N; ++k) {
        grad_coeffs[k] = T(k) * grad_output[k - 1] + grad_output[k + 1];
    }
}

// Complex specialization
template <typename T>
void hermite_polynomial_he_mulx_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;
    (void)output_size;

    if (N == 0) {
        return;
    }

    grad_coeffs[0] = grad_output[1];

    for (int64_t k = 1; k < N; ++k) {
        grad_coeffs[k] = c10::complex<T>(T(k), T(0)) * grad_output[k - 1] + grad_output[k + 1];
    }
}

} // namespace torchscience::kernel::polynomial
