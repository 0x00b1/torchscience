#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for Chebyshev U polynomial antiderivative.
//
// Forward computes a[0..N] from c[0..N-1]:
//   a_1 += c_0/2
//   a_{k+1} += c_k / (2*(k+2))  for k >= 1
//   a_{k-1} -= c_k / (2*k)      for k >= 1
//   a_0 = -sum_{k even, k>0} sign_k * a_k  (to make P(0) = 0)
//
// For backward, we compute grad_c[i] = sum_k (da_k/dc_i) * grad_a[k]
//
// Direct contributions:
//   da_1/dc_0 = 1/2
//   da_{k+1}/dc_k = 1/(2*(k+2)) for k >= 1
//   da_{k-1}/dc_k = -1/(2*k) for k >= 1
//
// a_0 depends on all even-indexed a_k (k >= 2), creating indirect contributions.
//
// Parameters:
//   grad_coeffs: output gradient for input coefficients, size N
//   grad_output: incoming gradient for antiderivative coefficients, size N+1
//   coeffs: original input coefficients
//   N: number of input coefficients
template <typename T>
void chebyshev_polynomial_u_antiderivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    const T* coeffs,
    int64_t N
) {
    // Initialize gradients to zero
    for (int64_t i = 0; i < N; ++i) {
        grad_coeffs[i] = T(0);
    }

    if (N == 0) {
        return;
    }

    // Compute contribution from a_0's dependence on even-indexed a_k
    T grad_a0 = grad_output[0];

    // Direct contribution from a_1 += c_0/2
    // da_1/dc_0 = 1/2
    grad_coeffs[0] += grad_output[1] / T(2);

    // Also, a_0 may depend on a_1 indirectly? No, only even indices.
    // a_0 = -sum_{k even, k>=0} sign_k * a_k (but a_0 is what we're computing)
    // Actually a_0 = -(sum_{k even, k>=2} sign_k * a_k + sign_0 * 0)
    // The contribution of a_0 to itself is handled in the forward

    // For k >= 1: contributions to a_{k+1} and a_{k-1}
    for (int64_t k = 1; k < N; ++k) {
        T scale_plus = T(1) / (T(2) * T(k + 2));
        T scale_minus = T(1) / (T(2) * T(k));

        // Direct: da_{k+1}/dc_k = scale_plus
        if (k + 1 <= N) {
            grad_coeffs[k] += grad_output[k + 1] * scale_plus;
        }

        // Direct: da_{k-1}/dc_k = -scale_minus
        grad_coeffs[k] -= grad_output[k - 1] * scale_minus;

        // Indirect through a_0 for even indices
        // If k+1 is even, da_0/da_{k+1} = -sign_{k+1}
        if ((k + 1) % 2 == 0 && k + 1 <= N) {
            int64_t sign_kp1 = (((k + 1) / 2) % 2 == 0) ? 1 : -1;
            grad_coeffs[k] += grad_a0 * T(-sign_kp1) * scale_plus;
        }

        // If k-1 is even, da_0/da_{k-1} = -sign_{k-1}
        if ((k - 1) % 2 == 0) {
            int64_t sign_km1 = (((k - 1) / 2) % 2 == 0) ? 1 : -1;
            // da_0/dc_k = -sign_{k-1} * (-scale_minus) = sign_{k-1} * scale_minus
            grad_coeffs[k] += grad_a0 * T(sign_km1) * scale_minus;
        }
    }
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_u_antiderivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    using C = c10::complex<T>;
    const C zero(T(0), T(0));

    // Initialize gradients to zero
    for (int64_t i = 0; i < N; ++i) {
        grad_coeffs[i] = zero;
    }

    if (N == 0) {
        return;
    }

    C grad_a0 = grad_output[0];

    // Direct contribution from a_1 += c_0/2
    grad_coeffs[0] += grad_output[1] / C(T(2), T(0));

    // For k >= 1
    for (int64_t k = 1; k < N; ++k) {
        C scale_plus = C(T(1), T(0)) / C(T(2) * T(k + 2), T(0));
        C scale_minus = C(T(1), T(0)) / C(T(2) * T(k), T(0));

        if (k + 1 <= N) {
            grad_coeffs[k] += grad_output[k + 1] * scale_plus;
        }

        grad_coeffs[k] -= grad_output[k - 1] * scale_minus;

        if ((k + 1) % 2 == 0 && k + 1 <= N) {
            int64_t sign_kp1 = (((k + 1) / 2) % 2 == 0) ? 1 : -1;
            grad_coeffs[k] += grad_a0 * C(T(-sign_kp1), T(0)) * scale_plus;
        }

        if ((k - 1) % 2 == 0) {
            int64_t sign_km1 = (((k - 1) / 2) % 2 == 0) ? 1 : -1;
            grad_coeffs[k] += grad_a0 * C(T(sign_km1), T(0)) * scale_minus;
        }
    }
}

} // namespace torchscience::kernel::polynomial
