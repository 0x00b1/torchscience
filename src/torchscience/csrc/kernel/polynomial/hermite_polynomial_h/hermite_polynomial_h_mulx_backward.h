#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward for Hermite H polynomial mulx
//
// Forward: x * H_k = H_{k+1}/2 + k * H_{k-1}
// output[1] += coeffs[0] / 2
// output[k-1] += k * coeffs[k] for k >= 1
// output[k+1] += coeffs[k] / 2 for k >= 1
//
// Gradients:
// grad_coeffs[0] = grad_output[1] / 2
// grad_coeffs[k] = k * grad_output[k-1] + grad_output[k+1] / 2 for k >= 1
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coeffs, size N
//   grad_output: incoming gradient, size output_size
//   coeffs: original input coefficients (unused)
//   N: size of coeffs
//   output_size: size of grad_output
template <typename T>
void hermite_polynomial_h_mulx_backward(
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

    // k = 0: grad_coeffs[0] = grad_output[1] / 2
    grad_coeffs[0] = grad_output[1] / T(2);

    // k >= 1: grad_coeffs[k] = k * grad_output[k-1] + grad_output[k+1] / 2
    for (int64_t k = 1; k < N; ++k) {
        grad_coeffs[k] = T(k) * grad_output[k - 1] + grad_output[k + 1] / T(2);
    }
}

// Complex specialization
template <typename T>
void hermite_polynomial_h_mulx_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;
    (void)output_size;
    const c10::complex<T> half(T(0.5), T(0));

    if (N == 0) {
        return;
    }

    grad_coeffs[0] = grad_output[1] * half;

    for (int64_t k = 1; k < N; ++k) {
        grad_coeffs[k] = c10::complex<T>(T(k), T(0)) * grad_output[k - 1] + grad_output[k + 1] * half;
    }
}

} // namespace torchscience::kernel::polynomial
