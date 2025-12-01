#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Second-order backward for Chebyshev T polynomial multiply-by-x.
//
// The mulx operation is linear in coeffs:
//   output = A * coeffs   (where A is the mulx transformation matrix)
//
// First backward:
//   grad_coeffs = A^T * grad_output
//
// Since the operation is linear (not depending on coeffs values):
// - d(grad_coeffs)/d(coeffs) = 0  (the transformation doesn't depend on coeffs values)
// - d(grad_coeffs)/d(grad_output) = A^T
//
// Second-order terms:
// - grad_grad_output: gg_coeffs flows through A^T^T = A to grad_grad_output
//   So grad_grad_output = A * gg_coeffs = mulx(gg_coeffs)
// - grad_coeffs_from_gg: Since backward is independent of coeffs, this is 0
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   grad_coeffs_from_gg: additional gradient for coeffs from gg_coeffs, size N (will be zero)
//   gg_coeffs: incoming second-order gradient for coeffs, size N
//   grad_output: gradient from forward backward, size output_size (unused)
//   coeffs: original coefficients, size N (unused)
//   N: number of input coefficients
//   output_size: size of grad_output and grad_grad_output
template <typename T>
void chebyshev_polynomial_t_mulx_backward_backward(
    T* grad_grad_output,
    T* grad_coeffs_from_gg,
    const T* gg_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N,
    int64_t output_size
) {
    // Initialize outputs to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = T(0);
    }
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs_from_gg[k] = T(0);
    }

    if (N == 0) {
        return;
    }

    // grad_grad_output = mulx(gg_coeffs)
    // Same formula as forward:
    // output[1] += coeffs[0]
    // output[k+1] += 0.5 * coeffs[k]  for k >= 1
    // output[k-1] += 0.5 * coeffs[k]  for k >= 1

    // From gg_coeffs[0]: contributes to grad_grad_output[1]
    if (1 < output_size) {
        grad_grad_output[1] += gg_coeffs[0];
    }

    // From gg_coeffs[k] for k >= 1: contributes to grad_grad_output[k+1] and grad_grad_output[k-1]
    for (int64_t k = 1; k < N; ++k) {
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] += T(0.5) * gg_coeffs[k];
        }
        if (k - 1 >= 0) {
            grad_grad_output[k - 1] += T(0.5) * gg_coeffs[k];
        }
    }

    // grad_coeffs_from_gg is zero since the backward is independent of coeffs
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_t_mulx_backward_backward(
    c10::complex<T>* grad_grad_output,
    c10::complex<T>* grad_coeffs_from_gg,
    const c10::complex<T>* gg_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> half(T(0.5), T(0));

    // Initialize outputs to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = zero;
    }
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs_from_gg[k] = zero;
    }

    if (N == 0) {
        return;
    }

    // grad_grad_output = mulx(gg_coeffs)
    // From gg_coeffs[0]: contributes to grad_grad_output[1]
    if (1 < output_size) {
        grad_grad_output[1] += gg_coeffs[0];
    }

    // From gg_coeffs[k] for k >= 1: contributes to grad_grad_output[k+1] and grad_grad_output[k-1]
    for (int64_t k = 1; k < N; ++k) {
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] += half * gg_coeffs[k];
        }
        if (k - 1 >= 0) {
            grad_grad_output[k - 1] += half * gg_coeffs[k];
        }
    }

    // grad_coeffs_from_gg is zero since the backward is independent of coeffs
}

} // namespace torchscience::kernel::polynomial
