#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Laguerre polynomial derivative
//
// The derivative operation is linear in coeffs, so the Hessian is zero.
// grad_grad_output[j] = 0 for all j (grad_output -> grad_coeffs is linear in grad_output)
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: second-order gradient w.r.t. coeffs
//   N: size of coeffs
//   output_size: size of grad_output and grad_grad_output
template <typename T>
void laguerre_polynomial_l_derivative_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    // Initialize grad_grad_output to zero
    for (int64_t j = 0; j < output_size; ++j) {
        grad_grad_output[j] = T(0);
    }

    // Since the forward is linear: output[j] = -sum_{k=j+1}^{N-1} coeffs[k]
    // And backward: grad_coeffs[k] = -sum_{j=0}^{k-1} grad_output[j]
    // The backward is also linear in grad_output, so:
    // grad_grad_output[j] = sum_k gg_coeffs[k] * d(grad_coeffs[k])/d(grad_output[j])
    //                     = sum_{k=j+1}^{N-1} gg_coeffs[k] * (-1)
    //                     = -sum_{k=j+1}^{N-1} gg_coeffs[k]

    if (N <= 1) {
        return;
    }

    // Compute from right to left
    T cumsum = T(0);
    for (int64_t k = N - 1; k >= 1; --k) {
        cumsum = cumsum + gg_coeffs[k];
        if (k - 1 < output_size) {
            grad_grad_output[k - 1] = -cumsum;
        }
    }
}

// Complex specialization
template <typename T>
void laguerre_polynomial_l_derivative_backward_backward(
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

    c10::complex<T> cumsum = zero;
    for (int64_t k = N - 1; k >= 1; --k) {
        cumsum = cumsum + gg_coeffs[k];
        if (k - 1 < output_size) {
            grad_grad_output[k - 1] = -cumsum;
        }
    }
}

} // namespace torchscience::kernel::polynomial
