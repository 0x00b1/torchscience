#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Hermite He polynomial derivative
//
// Backward: grad_coeffs[j+1] = (j+1) * grad_output[j]
//
// grad_grad_output[j] = sum_k gg_coeffs[k] * d(grad_coeffs[k])/d(grad_output[j])
//
// d(grad_coeffs[j+1])/d(grad_output[j]) = (j+1)
// So grad_grad_output[j] = (j+1) * gg_coeffs[j+1]
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: second-order gradient w.r.t. coeffs, size N
//   N: size of coeffs
//   output_size: size of grad_output and grad_grad_output
template <typename T>
void hermite_polynomial_he_derivative_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    for (int64_t j = 0; j < output_size; ++j) {
        if (j + 1 < N) {
            grad_grad_output[j] = T(j + 1) * gg_coeffs[j + 1];
        } else {
            grad_grad_output[j] = T(0);
        }
    }
}

// Complex specialization
template <typename T>
void hermite_polynomial_he_derivative_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t j = 0; j < output_size; ++j) {
        if (j + 1 < N) {
            grad_grad_output[j] = c10::complex<T>(T(j + 1), T(0)) * gg_coeffs[j + 1];
        } else {
            grad_grad_output[j] = zero;
        }
    }
}

} // namespace torchscience::kernel::polynomial
