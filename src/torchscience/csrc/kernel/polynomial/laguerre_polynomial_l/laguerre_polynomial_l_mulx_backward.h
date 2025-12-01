#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward for Laguerre polynomial mulx
//
// Forward: x * L_k = (2k+1)*L_k - (k+1)*L_{k+1} - k*L_{k-1}
//
// output[0] += coeffs[0]           (from k=0, L_0 term)
// output[1] -= coeffs[0]           (from k=0, L_1 term)
// output[k-1] -= k * coeffs[k]     (from k>=1)
// output[k] += (2k+1) * coeffs[k]  (from k>=1)
// output[k+1] -= (k+1) * coeffs[k] (from k>=1)
//
// Gradients:
// grad_coeffs[0] = grad_output[0] - grad_output[1]
// grad_coeffs[k] = (2k+1)*grad_output[k] - k*grad_output[k-1] - (k+1)*grad_output[k+1]  for k >= 1
//
// Parameters:
//   grad_coeffs: output gradient w.r.t. coeffs, size N
//   grad_output: incoming gradient, size output_size
//   coeffs: original input coefficients (unused)
//   N: size of coeffs
//   output_size: size of grad_output
template <typename T>
void laguerre_polynomial_l_mulx_backward(
    T* grad_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;
    (void)output_size;

    if (N == 0) {
        return;
    }

    // k = 0: grad_coeffs[0] = grad_output[0] - grad_output[1]
    grad_coeffs[0] = grad_output[0] - grad_output[1];

    // k >= 1: grad_coeffs[k] = (2k+1)*grad_output[k] - k*grad_output[k-1] - (k+1)*grad_output[k+1]
    for (int64_t k = 1; k < N; ++k) {
        const T factor_k = T(2 * k + 1);
        const T factor_km1 = T(k);
        const T factor_k1 = T(k + 1);

        grad_coeffs[k] = factor_k * grad_output[k] - factor_km1 * grad_output[k - 1] - factor_k1 * grad_output[k + 1];
    }
}

// Complex specialization
template <typename T>
void laguerre_polynomial_l_mulx_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N,
    int64_t output_size
) {
    (void)coeffs;
    (void)output_size;

    if (N == 0) {
        return;
    }

    grad_coeffs[0] = grad_output[0] - grad_output[1];

    for (int64_t k = 1; k < N; ++k) {
        const c10::complex<T> factor_k(T(2 * k + 1), T(0));
        const c10::complex<T> factor_km1(T(k), T(0));
        const c10::complex<T> factor_k1(T(k + 1), T(0));

        grad_coeffs[k] = factor_k * grad_output[k] - factor_km1 * grad_output[k - 1] - factor_k1 * grad_output[k + 1];
    }
}

} // namespace torchscience::kernel::polynomial
