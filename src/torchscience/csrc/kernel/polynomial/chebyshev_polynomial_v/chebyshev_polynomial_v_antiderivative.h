#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Chebyshev V polynomial antiderivative
//
// Given coefficients c[0..N-1] representing p(x) = sum_{k=0}^{N-1} c_k * V_k(x),
// computes the antiderivative P(x) = integral p(x) dx such that P(0) = 0.
//
// The antiderivative has coefficients a[0..N]:
// Using the integral identity for Chebyshev V polynomials:
//   integral V_n(x) dx = (V_{n+1}(x) - V_{n-1}(x)) / (2*(n+1)) + constant  for n >= 1
//   integral V_0(x) dx = x = (V_1(x) + 1) / 2
//
// For the coefficient form:
//   a_N = c_{N-1} / (2*N)
//   a_k = (c_{k-1} + c_{k+1}) / (2*k)  for k = 2..N-1  (note: + instead of -)
//   a_1 = c_0/2 + c_2/(2*2)
//   a_0 = constant (chosen so P(0) = 0)
//
// The constant a_0 is chosen so that P(0) = 0, using V_k(0) values.
//
// Parameters:
//   output: array of size N + 1 to store antiderivative coefficients
//   coeffs: input coefficients [c_0, c_1, ..., c_{N-1}]
//   N: number of input coefficients
//
// Returns: size of output (N + 1)
template <typename T>
int64_t chebyshev_polynomial_v_antiderivative(
    T* output,
    const T* coeffs,
    int64_t N
) {
    if (N == 0) {
        output[0] = T(0);
        return 1;
    }

    const int64_t output_size = N + 1;

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = T(0);
    }

    // a_N = c_{N-1} / (2*N)
    output[N] = coeffs[N - 1] / (T(2) * T(N));

    // a_k = (c_{k-1} + c_{k+1}) / (2*k) for k = 2..N-1
    // Note: For V polynomials, the recurrence involves + not -
    for (int64_t k = N - 1; k >= 2; --k) {
        T c_km1 = coeffs[k - 1];
        T c_kp1 = (k + 1 < N) ? coeffs[k + 1] : T(0);
        output[k] = (c_km1 + c_kp1) / (T(2) * T(k));
    }

    // a_1 = c_0/2 + c_2/4 (if N > 2)
    if (N >= 1) {
        T c_2 = (N > 2) ? coeffs[2] : T(0);
        output[1] = coeffs[0] / T(2) + c_2 / T(4);
    }

    // Compute a_0 so that P(0) = 0
    // V_0(0) = 1, V_1(0) = -1, V_2(0) = -1, V_3(0) = 1, V_4(0) = 1, V_5(0) = -1, ...
    // V_n(0) = (-1)^n for even, (-1)^((n+1)/2) for odd...
    // Actually V_n(0) follows: 1, -1, -1, 1, 1, -1, -1, 1, 1, ...
    // Pattern repeats with period 4: {1, -1, -1, 1}
    T p_at_zero = T(0);
    for (int64_t k = 0; k <= N; ++k) {
        // V_k(0) pattern: k % 4 -> {0: 1, 1: -1, 2: -1, 3: 1}
        int64_t mod4 = k % 4;
        int64_t sign = (mod4 == 0 || mod4 == 3) ? 1 : -1;
        p_at_zero += T(sign) * output[k];
    }
    // We want P(0) = 0, so a_0 = -rest (where rest = p_at_zero - output[0])
    // Since V_0(0) = 1, we have: output[0] + rest = 0 => output[0] = -rest
    // But output[0] is currently 0, so:
    output[0] = -p_at_zero;

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_v_antiderivative(
    c10::complex<T>* output,
    const c10::complex<T>* coeffs,
    int64_t N
) {
    using C = c10::complex<T>;

    if (N == 0) {
        output[0] = C(T(0), T(0));
        return 1;
    }

    const int64_t output_size = N + 1;
    const C zero(T(0), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    // a_N = c_{N-1} / (2*N)
    output[N] = coeffs[N - 1] / C(T(2) * T(N), T(0));

    // a_k = (c_{k-1} + c_{k+1}) / (2*k) for k = 2..N-1
    for (int64_t k = N - 1; k >= 2; --k) {
        C c_km1 = coeffs[k - 1];
        C c_kp1 = (k + 1 < N) ? coeffs[k + 1] : zero;
        output[k] = (c_km1 + c_kp1) / C(T(2) * T(k), T(0));
    }

    // a_1 = c_0/2 + c_2/4 (if N > 2)
    if (N >= 1) {
        C c_2 = (N > 2) ? coeffs[2] : zero;
        output[1] = coeffs[0] / C(T(2), T(0)) + c_2 / C(T(4), T(0));
    }

    // Compute a_0 so that P(0) = 0
    C p_at_zero = zero;
    for (int64_t k = 0; k <= N; ++k) {
        int64_t mod4 = k % 4;
        int64_t sign = (mod4 == 0 || mod4 == 3) ? 1 : -1;
        p_at_zero += C(T(sign), T(0)) * output[k];
    }
    output[0] = -p_at_zero;

    return output_size;
}

} // namespace torchscience::kernel::polynomial
