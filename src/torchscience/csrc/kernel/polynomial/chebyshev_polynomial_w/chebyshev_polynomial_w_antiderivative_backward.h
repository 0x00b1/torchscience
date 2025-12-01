#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for Chebyshev W polynomial antiderivative.
//
// Forward computes a[0..N] from c[0..N-1]:
//   a_N = c_{N-1} / (2*N)
//   a_k = (c_{k-1} - c_{k+1}) / (2*k)  for k = 2..N-1
//   a_1 = c_0 - 0.5*c_2
//   a_0 = -sum_k sign_k * a_k  (to make P(0) = 0, where sign_k = (k%4 < 2) ? 1 : -1)
//
// For backward, we compute grad_c[i] = sum_k (da_k/dc_i) * grad_a[k]
//
// Direct contributions:
//   da_N/dc_{N-1} = 1/(2*N)
//   da_k/dc_{k-1} = 1/(2*k) for k = 2..N-1
//   da_k/dc_{k+1} = -1/(2*k) for k = 2..N-1
//   da_1/dc_0 = 1
//   da_1/dc_2 = -0.5
//
// a_0 depends on all a_k (k >= 1), creating indirect contributions:
//   da_0/da_k = -sign_k for all k
//   So da_0/dc_i = sum_k (-sign_k) * (da_k/dc_i)
//
// Parameters:
//   grad_coeffs: output gradient for input coefficients, size N
//   grad_output: incoming gradient for antiderivative coefficients, size N+1
//   coeffs: original input coefficients
//   N: number of input coefficients
template <typename T>
void chebyshev_polynomial_w_antiderivative_backward(
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

    // Compute contribution from a_0's dependence on all a_k
    // a_0 = -sum_{k=0}^{N} sign_k * a_k where sign_k = (k%4 < 2) ? 1 : -1
    // So grad_output[0] flows to grad_coeffs through the chain rule
    T grad_a0 = grad_output[0];

    // Helper to get sign for W polynomial at 0
    auto get_sign = [](int64_t k) -> int64_t {
        return ((k % 4) < 2) ? 1 : -1;
    };

    // Direct contribution from a_N = c_{N-1} / (2*N)
    // grad_c[N-1] += grad_a[N] / (2*N)
    grad_coeffs[N - 1] += grad_output[N] / (T(2) * T(N));

    // a_N affects a_0:
    // a_0 depends on a_N with coefficient -sign_N
    int64_t sign_N = get_sign(N);
    grad_coeffs[N - 1] += grad_a0 * T(-sign_N) / (T(2) * T(N));

    // Contributions from a_k = (c_{k-1} - c_{k+1}) / (2*k) for k = 2..N-1
    for (int64_t k = 2; k < N; ++k) {
        T scale = T(1) / (T(2) * T(k));

        // Direct contributions
        // da_k/dc_{k-1} = 1/(2*k)
        grad_coeffs[k - 1] += grad_output[k] * scale;

        // da_k/dc_{k+1} = -1/(2*k) if k+1 < N
        if (k + 1 < N) {
            grad_coeffs[k + 1] -= grad_output[k] * scale;
        }

        // Indirect contribution through a_0
        int64_t sign_k = get_sign(k);
        // da_0/da_k = -sign_k
        // da_0/dc_{k-1} = -sign_k * (1/(2*k))
        // da_0/dc_{k+1} = -sign_k * (-1/(2*k)) = sign_k / (2*k)
        grad_coeffs[k - 1] += grad_a0 * T(-sign_k) * scale;
        if (k + 1 < N) {
            grad_coeffs[k + 1] += grad_a0 * T(sign_k) * scale;
        }
    }

    // Contribution from a_1 = c_0 - 0.5*c_2
    // da_1/dc_0 = 1
    grad_coeffs[0] += grad_output[1];

    // da_1/dc_2 = -0.5 if N > 2
    if (N > 2) {
        grad_coeffs[2] -= T(0.5) * grad_output[1];
    }

    // Indirect contribution from a_1 through a_0
    int64_t sign_1 = get_sign(1);  // = 1
    grad_coeffs[0] += grad_a0 * T(-sign_1);  // da_0/dc_0 through a_1
    if (N > 2) {
        grad_coeffs[2] += grad_a0 * T(sign_1) * T(0.5);  // da_0/dc_2 through a_1
    }
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_w_antiderivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    using C = c10::complex<T>;
    const C zero(T(0), T(0));
    const C half(T(0.5), T(0));

    // Initialize gradients to zero
    for (int64_t i = 0; i < N; ++i) {
        grad_coeffs[i] = zero;
    }

    if (N == 0) {
        return;
    }

    C grad_a0 = grad_output[0];

    auto get_sign = [](int64_t k) -> int64_t {
        return ((k % 4) < 2) ? 1 : -1;
    };

    // Direct contribution from a_N = c_{N-1} / (2*N)
    grad_coeffs[N - 1] += grad_output[N] / C(T(2) * T(N), T(0));

    // a_N affects a_0
    int64_t sign_N = get_sign(N);
    grad_coeffs[N - 1] += grad_a0 * C(T(-sign_N), T(0)) / C(T(2) * T(N), T(0));

    // Contributions from a_k for k = 2..N-1
    for (int64_t k = 2; k < N; ++k) {
        C scale = C(T(1), T(0)) / C(T(2) * T(k), T(0));

        grad_coeffs[k - 1] += grad_output[k] * scale;

        if (k + 1 < N) {
            grad_coeffs[k + 1] -= grad_output[k] * scale;
        }

        int64_t sign_k = get_sign(k);
        grad_coeffs[k - 1] += grad_a0 * C(T(-sign_k), T(0)) * scale;
        if (k + 1 < N) {
            grad_coeffs[k + 1] += grad_a0 * C(T(sign_k), T(0)) * scale;
        }
    }

    // Contribution from a_1 = c_0 - 0.5*c_2
    grad_coeffs[0] += grad_output[1];

    if (N > 2) {
        grad_coeffs[2] -= half * grad_output[1];
    }

    // Indirect contribution from a_1 through a_0
    int64_t sign_1 = get_sign(1);
    grad_coeffs[0] += grad_a0 * C(T(-sign_1), T(0));
    if (N > 2) {
        grad_coeffs[2] += grad_a0 * C(T(sign_1), T(0)) * half;
    }
}

} // namespace torchscience::kernel::polynomial
