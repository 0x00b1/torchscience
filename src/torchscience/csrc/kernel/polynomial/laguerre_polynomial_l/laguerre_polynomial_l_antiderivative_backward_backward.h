#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Laguerre polynomial antiderivative
//
// The antiderivative is linear, and the backward is also linear.
// Forward: output[k] = coeffs[k] - coeffs[k-1] (with boundary adjustments)
// Backward: grad_coeffs[k] = grad_output[k] - grad_output[k+1]
//
// So: grad_grad_output[j] = sum_k gg_coeffs[k] * d(grad_coeffs[k])/d(grad_output[j])
//
// d(grad_coeffs[k])/d(grad_output[j]) = 1 if j == k, -1 if j == k+1, 0 otherwise
//
// grad_grad_output[j] = gg_coeffs[j] - gg_coeffs[j-1] (for valid indices)
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: second-order gradient w.r.t. coeffs, size N
//   grad_output: original gradient (unused)
//   coeffs: original input coefficients (unused)
//   N: size of coeffs
//   output_size: size of grad_output and grad_grad_output (N+1)
template <typename T>
void laguerre_polynomial_l_antiderivative_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)grad_output;
    (void)coeffs;

    // Initialize
    for (int64_t j = 0; j < output_size; ++j) {
        grad_grad_output[j] = T(0);
    }

    if (N == 0) {
        return;
    }

    // grad_grad_output[j] = gg_coeffs[j] - gg_coeffs[j-1]
    // where gg_coeffs[j] is 0 for j >= N and gg_coeffs[-1] = 0

    // j = 0: gg_coeffs[0] - 0 = gg_coeffs[0]
    grad_grad_output[0] = gg_coeffs[0];

    // j = 1 to N-1: gg_coeffs[j] - gg_coeffs[j-1]
    for (int64_t j = 1; j < N; ++j) {
        grad_grad_output[j] = gg_coeffs[j] - gg_coeffs[j - 1];
    }

    // j = N: 0 - gg_coeffs[N-1] = -gg_coeffs[N-1]
    grad_grad_output[N] = -gg_coeffs[N - 1];
}

// Complex specialization
template <typename T>
void laguerre_polynomial_l_antiderivative_backward_backward(
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

    if (N == 0) {
        return;
    }

    grad_grad_output[0] = gg_coeffs[0];

    for (int64_t j = 1; j < N; ++j) {
        grad_grad_output[j] = gg_coeffs[j] - gg_coeffs[j - 1];
    }

    grad_grad_output[N] = -gg_coeffs[N - 1];
}

} // namespace torchscience::kernel::polynomial
