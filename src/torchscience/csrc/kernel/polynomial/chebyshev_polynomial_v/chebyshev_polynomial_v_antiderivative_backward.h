#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for Chebyshev V polynomial antiderivative.
//
// Forward computes a[0..N] from c[0..N-1]:
//   a_N = c_{N-1} / (2*N)
//   a_k = (c_{k-1} + c_{k+1}) / (2*k)  for k = 2..N-1
//   a_1 = c_0/2 + c_2/4
//   a_0 = -sum_{k} sign_k * a_k  (to make P(0) = 0, using V_k(0) pattern)
//
// For backward, we compute grad_c[i] = sum_k (da_k/dc_i) * grad_a[k]
//
// Direct contributions:
//   da_N/dc_{N-1} = 1/(2*N)
//   da_k/dc_{k-1} = 1/(2*k) for k = 2..N-1
//   da_k/dc_{k+1} = 1/(2*k) for k = 2..N-1  (note: + instead of - for V)
//   da_1/dc_0 = 1/2
//   da_1/dc_2 = 1/4
//
// a_0 depends on all a_k, creating indirect contributions:
//   da_0/da_k = -sign_k where sign_k follows V_k(0) pattern
//
// Parameters:
//   grad_coeffs: output gradient for input coefficients, size N
//   grad_output: incoming gradient for antiderivative coefficients, size N+1
//   coeffs: original input coefficients
//   N: number of input coefficients
template <typename T>
void chebyshev_polynomial_v_antiderivative_backward(
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

    // Helper lambda for V_k(0) sign: pattern {1, -1, -1, 1} repeating
    auto v_sign = [](int64_t k) -> int64_t {
        int64_t mod4 = k % 4;
        return (mod4 == 0 || mod4 == 3) ? 1 : -1;
    };

    // Contribution from a_0's dependence on all a_k
    T grad_a0 = grad_output[0];

    // Direct contribution from a_N = c_{N-1} / (2*N)
    grad_coeffs[N - 1] += grad_output[N] / (T(2) * T(N));

    // Also, a_N affects a_0:
    // a_0 depends on a_N with coefficient -sign_N
    int64_t sign_N = v_sign(N);
    grad_coeffs[N - 1] += grad_a0 * T(-sign_N) / (T(2) * T(N));

    // Contributions from a_k = (c_{k-1} + c_{k+1}) / (2*k) for k = 2..N-1
    for (int64_t k = 2; k < N; ++k) {
        T scale = T(1) / (T(2) * T(k));

        // Direct contributions
        // da_k/dc_{k-1} = 1/(2*k)
        grad_coeffs[k - 1] += grad_output[k] * scale;

        // da_k/dc_{k+1} = 1/(2*k) if k+1 < N
        if (k + 1 < N) {
            grad_coeffs[k + 1] += grad_output[k] * scale;
        }

        // Indirect contribution through a_0
        int64_t sign_k = v_sign(k);
        // da_0/da_k = -sign_k
        // da_0/dc_{k-1} = -sign_k * (1/(2*k))
        // da_0/dc_{k+1} = -sign_k * (1/(2*k))
        grad_coeffs[k - 1] += grad_a0 * T(-sign_k) * scale;
        if (k + 1 < N) {
            grad_coeffs[k + 1] += grad_a0 * T(-sign_k) * scale;
        }
    }

    // Contribution from a_1 = c_0/2 + c_2/4
    // da_1/dc_0 = 1/2
    grad_coeffs[0] += grad_output[1] * T(0.5);

    // da_1/dc_2 = 1/4 if N > 2
    if (N > 2) {
        grad_coeffs[2] += grad_output[1] * T(0.25);
    }

    // a_0 depends on a_1 with sign_1 = -1
    // da_0/dc_0 = -sign_1 * (1/2) = 0.5
    // da_0/dc_2 = -sign_1 * (1/4) = 0.25
    grad_coeffs[0] += grad_a0 * T(0.5);  // -(-1) * 0.5 = 0.5
    if (N > 2) {
        grad_coeffs[2] += grad_a0 * T(0.25);  // -(-1) * 0.25 = 0.25
    }
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_v_antiderivative_backward(
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

    auto v_sign = [](int64_t k) -> int64_t {
        int64_t mod4 = k % 4;
        return (mod4 == 0 || mod4 == 3) ? 1 : -1;
    };

    C grad_a0 = grad_output[0];

    // Direct contribution from a_N = c_{N-1} / (2*N)
    grad_coeffs[N - 1] += grad_output[N] / C(T(2) * T(N), T(0));

    // a_N affects a_0
    int64_t sign_N = v_sign(N);
    grad_coeffs[N - 1] += grad_a0 * C(T(-sign_N), T(0)) / C(T(2) * T(N), T(0));

    // Contributions from a_k for k = 2..N-1
    for (int64_t k = 2; k < N; ++k) {
        C scale = C(T(1), T(0)) / C(T(2) * T(k), T(0));

        grad_coeffs[k - 1] += grad_output[k] * scale;

        if (k + 1 < N) {
            grad_coeffs[k + 1] += grad_output[k] * scale;
        }

        int64_t sign_k = v_sign(k);
        grad_coeffs[k - 1] += grad_a0 * C(T(-sign_k), T(0)) * scale;
        if (k + 1 < N) {
            grad_coeffs[k + 1] += grad_a0 * C(T(-sign_k), T(0)) * scale;
        }
    }

    // Contribution from a_1 = c_0/2 + c_2/4
    grad_coeffs[0] += grad_output[1] * C(T(0.5), T(0));

    if (N > 2) {
        grad_coeffs[2] += grad_output[1] * C(T(0.25), T(0));
    }

    // a_0 depends on a_1
    grad_coeffs[0] += grad_a0 * C(T(0.5), T(0));
    if (N > 2) {
        grad_coeffs[2] += grad_a0 * C(T(0.25), T(0));
    }
}

} // namespace torchscience::kernel::polynomial
