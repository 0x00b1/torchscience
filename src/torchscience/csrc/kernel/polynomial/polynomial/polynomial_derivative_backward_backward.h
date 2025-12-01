#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for polynomial derivative
// The derivative operator is linear, so second-order gradients are straightforward:
// gg_coeffs -> grad_grad_output: gg_output[k-1] = k * gg_coeffs[k]
// (same as forward, just applied to gradient inputs)
template <typename T>
void polynomial_derivative_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    if (output_size == 0) {
        return;
    }
    for (int64_t k = 0; k < output_size; ++k) {
        if (k + 1 < N) {
            grad_grad_output[k] = T(k + 1) * gg_coeffs[k + 1];
        } else {
            grad_grad_output[k] = T(0);
        }
    }
}

template <typename T>
void polynomial_derivative_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    if (output_size == 0) {
        return;
    }
    for (int64_t k = 0; k < output_size; ++k) {
        if (k + 1 < N) {
            grad_grad_output[k] = c10::complex<T>(T(k + 1), T(0)) * gg_coeffs[k + 1];
        } else {
            grad_grad_output[k] = c10::complex<T>(T(0), T(0));
        }
    }
}

} // namespace torchscience::kernel::polynomial
