#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Hermite H polynomial derivative
//
// Forward is linear, backward is also linear.
// grad_coeffs[k] = 2*k * grad_output[k-1] for k >= 1
// grad_grad_output[j] = sum_k gg_coeffs[k] * d(grad_coeffs[k])/d(grad_output[j])
//                     = gg_coeffs[j+1] * 2*(j+1)
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: second-order gradient w.r.t. coeffs, size N
//   N: size of coeffs
//   output_size: size of grad_output and grad_grad_output
template <typename T>
void hermite_polynomial_h_derivative_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    // Initialize grad_grad_output to zero
    for (int64_t j = 0; j < output_size; ++j) {
        grad_grad_output[j] = T(0);
    }

    if (N <= 1) {
        return;
    }

    // grad_grad_output[j] = 2*(j+1) * gg_coeffs[j+1]
    for (int64_t j = 0; j < output_size; ++j) {
        if (j + 1 < N) {
            grad_grad_output[j] = T(2 * (j + 1)) * gg_coeffs[j + 1];
        }
    }
}

// Complex specialization
template <typename T>
void hermite_polynomial_h_derivative_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t j = 0; j < output_size; ++j) {
        grad_grad_output[j] = zero;
    }

    if (N <= 1) {
        return;
    }

    for (int64_t j = 0; j < output_size; ++j) {
        if (j + 1 < N) {
            grad_grad_output[j] = c10::complex<T>(T(2 * (j + 1)), T(0)) * gg_coeffs[j + 1];
        }
    }
}

} // namespace torchscience::kernel::polynomial
