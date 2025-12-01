#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Hermite H polynomial mulx
//
// Backward: grad_coeffs[0] = grad_output[1] / 2
//           grad_coeffs[k] = k * grad_output[k-1] + grad_output[k+1] / 2 for k >= 1
//
// grad_grad_output[j] = sum_k gg_coeffs[k] * d(grad_coeffs[k])/d(grad_output[j])
//
// d(grad_coeffs[0])/d(grad_output[1]) = 1/2
// d(grad_coeffs[k])/d(grad_output[k-1]) = k for k >= 1
// d(grad_coeffs[k])/d(grad_output[k+1]) = 1/2 for k >= 1
//
// grad_grad_output[j] = gg_coeffs[j-1] / 2 (from k=j-1, if j >= 1 and k < N)
//                     + (j+1) * gg_coeffs[j+1] (from k=j+1, if j+1 < N)
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: second-order gradient w.r.t. coeffs, size N
//   grad_output: original gradient (unused)
//   coeffs: original input coefficients (unused)
//   N: size of coeffs
//   output_size: size of grad_output and grad_grad_output
template <typename T>
void hermite_polynomial_h_mulx_backward_backward(
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

    if (N == 0) {
        return;
    }

    // From grad_coeffs[0] = grad_output[1] / 2:
    // grad_grad_output[1] += gg_coeffs[0] / 2
    grad_grad_output[1] = grad_grad_output[1] + gg_coeffs[0] / T(2);

    // From grad_coeffs[k] = k * grad_output[k-1] + grad_output[k+1] / 2 for k >= 1:
    for (int64_t k = 1; k < N; ++k) {
        // grad_grad_output[k-1] += k * gg_coeffs[k]
        grad_grad_output[k - 1] = grad_grad_output[k - 1] + T(k) * gg_coeffs[k];
        // grad_grad_output[k+1] += gg_coeffs[k] / 2
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] = grad_grad_output[k + 1] + gg_coeffs[k] / T(2);
        }
    }
}

// Complex specialization
template <typename T>
void hermite_polynomial_h_mulx_backward_backward(
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
    const c10::complex<T> half(T(0.5), T(0));

    for (int64_t j = 0; j < output_size; ++j) {
        grad_grad_output[j] = zero;
    }

    if (N == 0) {
        return;
    }

    grad_grad_output[1] = grad_grad_output[1] + gg_coeffs[0] * half;

    for (int64_t k = 1; k < N; ++k) {
        grad_grad_output[k - 1] = grad_grad_output[k - 1] + c10::complex<T>(T(k), T(0)) * gg_coeffs[k];
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] = grad_grad_output[k + 1] + gg_coeffs[k] * half;
        }
    }
}

} // namespace torchscience::kernel::polynomial
