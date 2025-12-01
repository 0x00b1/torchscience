#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Chebyshev U polynomial antiderivative.
//
// The antiderivative is a linear operation: a = A * c (matrix-vector multiply)
// where A is the transformation matrix.
//
// First backward: grad_c = A^T * grad_a
//
// For second-order, since the operation is purely linear (no dependence on c
// in the coefficients of A), we have:
//   - grad_grad_output (w.r.t. grad_a) = A * gg_c
//   - grad_coeffs (w.r.t. c) = 0 (no second-order dependence)
//
// That is, the second-order gradient just applies the forward transformation
// to the incoming gradient gg_coeffs.
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size N+1
//   gg_coeffs: incoming second-order gradient for coefficients, size N
//   grad_output: original gradient from forward backward, size N+1
//   coeffs: original input coefficients, size N
//   N: number of input coefficients
template <typename T>
void chebyshev_polynomial_u_antiderivative_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N
) {
    const int64_t output_size = N + 1;

    // Initialize grad_grad_output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = T(0);
    }

    if (N == 0) {
        return;
    }

    // Apply the forward transformation A to gg_coeffs
    // This is the same as the forward pass but with gg_coeffs as input

    // gg_c_0 contributes gg_c_0/2 to output[1]
    grad_grad_output[1] += gg_coeffs[0] / T(2);

    // For k >= 1
    for (int64_t k = 1; k < N; ++k) {
        // Contribution to U_{k+1}
        grad_grad_output[k + 1] += gg_coeffs[k] / (T(2) * T(k + 2));
        // Contribution to U_{k-1}
        grad_grad_output[k - 1] -= gg_coeffs[k] / (T(2) * T(k));
    }

    // Compute a_0 so that P(0) = 0
    T p_at_zero = T(0);
    for (int64_t k = 0; k <= N; k += 2) {
        int64_t sign = ((k / 2) % 2 == 0) ? 1 : -1;
        p_at_zero += T(sign) * grad_grad_output[k];
    }
    grad_grad_output[0] = -p_at_zero;
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_u_antiderivative_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    using C = c10::complex<T>;
    const int64_t output_size = N + 1;
    const C zero(T(0), T(0));

    // Initialize grad_grad_output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = zero;
    }

    if (N == 0) {
        return;
    }

    // Apply the forward transformation A to gg_coeffs
    grad_grad_output[1] += gg_coeffs[0] / C(T(2), T(0));

    for (int64_t k = 1; k < N; ++k) {
        grad_grad_output[k + 1] += gg_coeffs[k] / C(T(2) * T(k + 2), T(0));
        grad_grad_output[k - 1] -= gg_coeffs[k] / C(T(2) * T(k), T(0));
    }

    C p_at_zero = zero;
    for (int64_t k = 0; k <= N; k += 2) {
        int64_t sign = ((k / 2) % 2 == 0) ? 1 : -1;
        p_at_zero += C(T(sign), T(0)) * grad_grad_output[k];
    }
    grad_grad_output[0] = -p_at_zero;
}

} // namespace torchscience::kernel::polynomial
