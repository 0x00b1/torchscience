#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for Chebyshev V polynomial antiderivative.
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
void chebyshev_polynomial_v_antiderivative_backward_backward(
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

    // Helper lambda for V_k(0) sign: pattern {1, -1, -1, 1} repeating
    auto v_sign = [](int64_t k) -> int64_t {
        int64_t mod4 = k % 4;
        return (mod4 == 0 || mod4 == 3) ? 1 : -1;
    };

    // Apply the forward transformation A to gg_coeffs
    // This is the same as the forward pass but with gg_coeffs as input

    // a_N = c_{N-1} / (2*N)
    grad_grad_output[N] = gg_coeffs[N - 1] / (T(2) * T(N));

    // a_k = (c_{k-1} + c_{k+1}) / (2*k) for k = 2..N-1
    for (int64_t k = N - 1; k >= 2; --k) {
        T c_km1 = gg_coeffs[k - 1];
        T c_kp1 = (k + 1 < N) ? gg_coeffs[k + 1] : T(0);
        grad_grad_output[k] = (c_km1 + c_kp1) / (T(2) * T(k));
    }

    // a_1 = c_0/2 + c_2/4
    if (N >= 1) {
        T c_2 = (N > 2) ? gg_coeffs[2] : T(0);
        grad_grad_output[1] = gg_coeffs[0] / T(2) + c_2 / T(4);
    }

    // a_0 = -sum_{k} sign_k * a_k
    T p_at_zero = T(0);
    for (int64_t k = 0; k <= N; ++k) {
        int64_t sign = v_sign(k);
        p_at_zero += T(sign) * grad_grad_output[k];
    }
    grad_grad_output[0] = -p_at_zero;
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_v_antiderivative_backward_backward(
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

    auto v_sign = [](int64_t k) -> int64_t {
        int64_t mod4 = k % 4;
        return (mod4 == 0 || mod4 == 3) ? 1 : -1;
    };

    // Apply the forward transformation A to gg_coeffs
    grad_grad_output[N] = gg_coeffs[N - 1] / C(T(2) * T(N), T(0));

    for (int64_t k = N - 1; k >= 2; --k) {
        C c_km1 = gg_coeffs[k - 1];
        C c_kp1 = (k + 1 < N) ? gg_coeffs[k + 1] : zero;
        grad_grad_output[k] = (c_km1 + c_kp1) / C(T(2) * T(k), T(0));
    }

    if (N >= 1) {
        C c_2 = (N > 2) ? gg_coeffs[2] : zero;
        grad_grad_output[1] = gg_coeffs[0] / C(T(2), T(0)) + c_2 / C(T(4), T(0));
    }

    C p_at_zero = zero;
    for (int64_t k = 0; k <= N; ++k) {
        int64_t sign = v_sign(k);
        p_at_zero += C(T(sign), T(0)) * grad_grad_output[k];
    }
    grad_grad_output[0] = -p_at_zero;
}

} // namespace torchscience::kernel::polynomial
