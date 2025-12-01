#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Legendre polynomial antiderivative
// Since antiderivative is linear, the structure mirrors forward.
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: second-order gradient w.r.t. coefficients, size N
//   N: number of original coefficients
//   output_size: size of output (N+1)
template <typename T>
void legendre_polynomial_p_antiderivative_backward_backward(
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

    if (N == 0) {
        return;
    }

    // Apply the same linear transformation as forward
    // output[1] += gg_coeffs[0]
    if (output_size > 1) {
        grad_grad_output[1] = grad_grad_output[1] + gg_coeffs[0];
    }

    // k >= 1 terms
    for (int64_t k = 1; k < N; ++k) {
        const T factor = T(1) / T(2 * k + 1);
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] = grad_grad_output[k + 1] + gg_coeffs[k] * factor;
        }
        if (k - 1 < output_size) {
            grad_grad_output[k - 1] = grad_grad_output[k - 1] - gg_coeffs[k] * factor;
        }
    }
}

// Complex specialization
template <typename T>
void legendre_polynomial_p_antiderivative_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    int64_t N,
    int64_t output_size
) {
    if (output_size == 0) {
        return;
    }

    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> one(T(1), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = zero;
    }

    if (N == 0) {
        return;
    }

    // output[1] += gg_coeffs[0]
    if (output_size > 1) {
        grad_grad_output[1] = grad_grad_output[1] + gg_coeffs[0];
    }

    // k >= 1 terms
    for (int64_t k = 1; k < N; ++k) {
        const c10::complex<T> factor = one / c10::complex<T>(T(2 * k + 1), T(0));
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] = grad_grad_output[k + 1] + gg_coeffs[k] * factor;
        }
        if (k - 1 < output_size) {
            grad_grad_output[k - 1] = grad_grad_output[k - 1] - gg_coeffs[k] * factor;
        }
    }
}

} // namespace torchscience::kernel::polynomial
