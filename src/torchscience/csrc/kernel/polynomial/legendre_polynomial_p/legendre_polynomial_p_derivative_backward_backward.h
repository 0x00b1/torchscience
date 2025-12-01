#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Legendre polynomial derivative
// Since derivative is linear, the structure of second-order gradients
// mirrors the first-order backward.
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: second-order gradient w.r.t. coefficients, size N
//   N: number of original coefficients
//   output_size: size of output (should be N-1)
template <typename T>
void legendre_polynomial_p_derivative_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    if (output_size == 0) {
        return;
    }

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = T(0);
    }

    if (N <= 1) {
        return;
    }

    // The second-order gradient follows the same linear map as forward.
    // We need to apply the derivative operation to gg_coeffs.

    // Copy gg_coeffs so we can modify during accumulation
    T* tmp = new T[N];
    for (int64_t k = 0; k < N; ++k) {
        tmp[k] = gg_coeffs[k];
    }

    // Apply the same algorithm as forward derivative
    for (int64_t j = N - 1; j >= 3; --j) {
        if (j - 1 < output_size) {
            grad_grad_output[j - 1] = T(2 * j - 1) * tmp[j];
        }
        tmp[j - 2] = tmp[j - 2] + tmp[j];
    }

    if (output_size > 1 && N > 2) {
        grad_grad_output[1] = T(3) * tmp[2];
    }

    if (output_size > 0 && N > 1) {
        grad_grad_output[0] = tmp[1];
    }

    delete[] tmp;
}

// Complex specialization
template <typename T>
void legendre_polynomial_p_derivative_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    if (output_size == 0) {
        return;
    }

    const c10::complex<T> zero(T(0), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = zero;
    }

    if (N <= 1) {
        return;
    }

    c10::complex<T>* tmp = new c10::complex<T>[N];
    for (int64_t k = 0; k < N; ++k) {
        tmp[k] = gg_coeffs[k];
    }

    for (int64_t j = N - 1; j >= 3; --j) {
        if (j - 1 < output_size) {
            grad_grad_output[j - 1] = c10::complex<T>(T(2 * j - 1), T(0)) * tmp[j];
        }
        tmp[j - 2] = tmp[j - 2] + tmp[j];
    }

    if (output_size > 1 && N > 2) {
        grad_grad_output[1] = c10::complex<T>(T(3), T(0)) * tmp[2];
    }

    if (output_size > 0 && N > 1) {
        grad_grad_output[0] = tmp[1];
    }

    delete[] tmp;
}

} // namespace torchscience::kernel::polynomial
