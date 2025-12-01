#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Chebyshev U polynomial derivative.
//
// The derivative operation is linear: output = L * coeffs, where L is a linear operator.
// Therefore:
//   grad_coeffs = L^T * grad_output  (first backward)
//   grad_grad_output = L * gg_coeffs  (second backward, same as forward applied to gg_coeffs)
//
// Since the operation is linear, the second-order backward is essentially
// applying the same transformation to gg_coeffs.
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: incoming second-order gradient w.r.t. coeffs, size N
//   N: number of input coefficients
//   output_size: size of output (should be max(N-1, 1))
template <typename T>
void chebyshev_polynomial_u_derivative_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = T(0);
    }

    if (N <= 1) {
        // Derivative of constant is zero
        return;
    }

    const int64_t deg = N - 1;

    // Since the derivative is a linear operation, applying the second-order backward
    // is the same as applying the forward transformation to gg_coeffs.
    //
    // Forward recurrence:
    //   d_{deg-1} = 2 * deg * c_{deg}
    //   d_k = d_{k+2} + 2*(k+1)*c_{k+1}  for k = deg-2, ..., 0

    // d_{deg-1} = 2 * deg * gg_coeffs[deg]
    if (deg - 1 < output_size && deg < N) {
        grad_grad_output[deg - 1] = T(2 * deg) * gg_coeffs[deg];
    }

    // d_k = d_{k+2} + 2*(k+1)*gg_coeffs[k+1] for k = deg-2 down to 0
    for (int64_t k = deg - 2; k >= 0; --k) {
        if (k < output_size) {
            T d_k = T(0);
            if (k + 1 < N) {
                d_k = T(2 * (k + 1)) * gg_coeffs[k + 1];
            }
            if (k + 2 < output_size) {
                d_k = d_k + grad_grad_output[k + 2];
            }
            grad_grad_output[k] = d_k;
        }
    }
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_u_derivative_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = zero;
    }

    if (N <= 1) {
        return;
    }

    const int64_t deg = N - 1;

    if (deg - 1 < output_size && deg < N) {
        grad_grad_output[deg - 1] = c10::complex<T>(T(2 * deg), T(0)) * gg_coeffs[deg];
    }

    for (int64_t k = deg - 2; k >= 0; --k) {
        if (k < output_size) {
            c10::complex<T> d_k = zero;
            if (k + 1 < N) {
                d_k = c10::complex<T>(T(2 * (k + 1)), T(0)) * gg_coeffs[k + 1];
            }
            if (k + 2 < output_size) {
                d_k = d_k + grad_grad_output[k + 2];
            }
            grad_grad_output[k] = d_k;
        }
    }
}

} // namespace torchscience::kernel::polynomial
