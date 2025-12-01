#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Laguerre polynomial mulx
//
// Since mulx is linear in coeffs, the Hessian is zero.
// The backward is: grad_coeffs[k] = (2k+1)*grad_output[k] - k*grad_output[k-1] - (k+1)*grad_output[k+1]
// which is linear in grad_output.
//
// So: grad_grad_output[j] = sum_k gg_coeffs[k] * d(grad_coeffs[k])/d(grad_output[j])
//
// d(grad_coeffs[k])/d(grad_output[j]):
//   = (2k+1) if j == k
//   = -k if j == k-1
//   = -(k+1) if j == k+1
//
// grad_grad_output[j] = (2j+1)*gg_coeffs[j] - (j+1)*gg_coeffs[j+1] - j*gg_coeffs[j-1]
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   gg_coeffs: second-order gradient w.r.t. coeffs, size N
//   grad_output: original gradient (unused)
//   coeffs: original input coefficients (unused)
//   N: size of coeffs
//   output_size: size of grad_output and grad_grad_output
template <typename T>
void laguerre_polynomial_l_mulx_backward_backward(
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

    // j = 0: (2*0+1)*gg_coeffs[0] - 1*gg_coeffs[1] - 0*gg_coeffs[-1]
    //      = gg_coeffs[0] - gg_coeffs[1] (if N > 1)
    grad_grad_output[0] = gg_coeffs[0];
    if (N > 1) {
        grad_grad_output[0] = grad_grad_output[0] - gg_coeffs[1];
    }

    // j = 1 to N-1: (2j+1)*gg_coeffs[j] - (j+1)*gg_coeffs[j+1] - j*gg_coeffs[j-1]
    for (int64_t j = 1; j < N; ++j) {
        const T factor_j = T(2 * j + 1);
        const T factor_jm1 = T(j);
        const T factor_j1 = T(j + 1);

        grad_grad_output[j] = factor_j * gg_coeffs[j] - factor_jm1 * gg_coeffs[j - 1];
        if (j + 1 < N) {
            grad_grad_output[j] = grad_grad_output[j] - factor_j1 * gg_coeffs[j + 1];
        }
    }

    // j = N: only contributions from k = N-1 (if j == k+1)
    // grad_grad_output[N] = -N * gg_coeffs[N-1]
    grad_grad_output[N] = -T(N) * gg_coeffs[N - 1];
}

// Complex specialization
template <typename T>
void laguerre_polynomial_l_mulx_backward_backward(
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
    if (N > 1) {
        grad_grad_output[0] = grad_grad_output[0] - gg_coeffs[1];
    }

    for (int64_t j = 1; j < N; ++j) {
        const c10::complex<T> factor_j(T(2 * j + 1), T(0));
        const c10::complex<T> factor_jm1(T(j), T(0));
        const c10::complex<T> factor_j1(T(j + 1), T(0));

        grad_grad_output[j] = factor_j * gg_coeffs[j] - factor_jm1 * gg_coeffs[j - 1];
        if (j + 1 < N) {
            grad_grad_output[j] = grad_grad_output[j] - factor_j1 * gg_coeffs[j + 1];
        }
    }

    grad_grad_output[N] = c10::complex<T>(T(-N), T(0)) * gg_coeffs[N - 1];
}

} // namespace torchscience::kernel::polynomial
