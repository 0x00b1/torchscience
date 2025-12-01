#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Hermite He polynomial antiderivative
//
// Backward: grad_coeffs[k] = grad_output[k+1] / (k+1)
//
// grad_grad_output[j] = sum_k gg_coeffs[k] * d(grad_coeffs[k])/d(grad_output[j])
//
// d(grad_coeffs[k])/d(grad_output[k+1]) = 1 / (k+1)
// So grad_grad_output[k+1] = gg_coeffs[k] / (k+1)
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: second-order gradient w.r.t. coeffs, size N
//   grad_output: original gradient (unused)
//   coeffs: original input coefficients (unused)
//   N: size of coeffs
//   output_size: size of grad_output and grad_grad_output
template <typename T>
void hermite_polynomial_he_antiderivative_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)grad_output;
    (void)coeffs;

    // Initialize to zero
    for (int64_t j = 0; j < output_size; ++j) {
        grad_grad_output[j] = T(0);
    }

    // grad_grad_output[k+1] = gg_coeffs[k] / (k+1)
    for (int64_t k = 0; k < N && k + 1 < output_size; ++k) {
        grad_grad_output[k + 1] = gg_coeffs[k] / T(k + 1);
    }
}

// Complex specialization
template <typename T>
void hermite_polynomial_he_antiderivative_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)grad_output;
    (void)coeffs;
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t j = 0; j < output_size; ++j) {
        grad_grad_output[j] = zero;
    }

    for (int64_t k = 0; k < N && k + 1 < output_size; ++k) {
        grad_grad_output[k + 1] = gg_coeffs[k] / c10::complex<T>(T(k + 1), T(0));
    }
}

} // namespace torchscience::kernel::polynomial
