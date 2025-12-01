#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for polynomial derivative
// Forward: output[k-1] = k * coeffs[k] for k = 1..N-1
// Backward: grad_coeffs[k] = k * grad_output[k-1] for k = 1..N-1
//           grad_coeffs[0] = 0 (constant term doesn't affect derivative)
template <typename T>
void polynomial_derivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    int64_t N,
    int64_t output_size
) {
    if (N == 0) {
        return;
    }
    grad_coeffs[0] = T(0);
    for (int64_t k = 1; k < N; ++k) {
        if (k - 1 < output_size) {
            grad_coeffs[k] = T(k) * grad_output[k - 1];
        } else {
            grad_coeffs[k] = T(0);
        }
    }
}

template <typename T>
void polynomial_derivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    int64_t N,
    int64_t output_size
) {
    if (N == 0) {
        return;
    }
    grad_coeffs[0] = c10::complex<T>(T(0), T(0));
    for (int64_t k = 1; k < N; ++k) {
        if (k - 1 < output_size) {
            grad_coeffs[k] = c10::complex<T>(T(k), T(0)) * grad_output[k - 1];
        } else {
            grad_coeffs[k] = c10::complex<T>(T(0), T(0));
        }
    }
}

} // namespace torchscience::kernel::polynomial
