#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Chebyshev W polynomial antiderivative
//
// Given coefficients c[0..N-1] representing p(x) = sum_{k=0}^{N-1} c_k * W_k(x),
// computes the antiderivative P(x) = integral p(x) dx such that P(0) = 0.
//
// The antiderivative has coefficients a[0..N]:
//   a_0 = constant (integration constant, chosen so P(0) = 0)
//   a_1 = c_0 - 0.5*c_2 (if N > 2, else just c_0)
//   a_k = (c_{k-1} - c_{k+1}) / (2*k)  for k = 2..N-1
//   a_N = c_{N-1} / (2*N)
//
// This uses the identity similar to Chebyshev T:
//   integral W_n(x) dx = W_{n+1}(x)/(2(n+1)) - W_{n-1}(x)/(2(n-1))  for n >= 2
//
// The constant a_0 is chosen so that P(0) = 0, i.e.,
//   a_0 = -sum_{k odd} (-1)^{(k-1)/2} * a_k
// since W_k(0) = (-1)^k for k odd and W_k(0) = 1 for k even is not standard.
// Actually for W polynomials: W_0(0) = 1, W_n(0) = 2*n + 1 for the pattern.
// However, evaluating at 0: W_n(cos(theta)) at theta = pi/2.
// For simplicity, we use: W_k(0) = (-1)^k * (2k + 1) / 1 based on explicit formula.
//
// Parameters:
//   output: array of size N + 1 to store antiderivative coefficients
//   coeffs: input coefficients [c_0, c_1, ..., c_{N-1}]
//   N: number of input coefficients
//
// Returns: size of output (N + 1)
template <typename T>
int64_t chebyshev_polynomial_w_antiderivative(
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

    // a_k = (c_{k-1} - c_{k+1}) / (2*k) for k = 2..N-1
    for (int64_t k = N - 1; k >= 2; --k) {
        T c_km1 = coeffs[k - 1];
        T c_kp1 = (k + 1 < N) ? coeffs[k + 1] : T(0);
        output[k] = (c_km1 - c_kp1) / (T(2) * T(k));
    }

    // a_1 = c_0 - 0.5*c_2 (if N > 2)
    if (N >= 1) {
        T c_2 = (N > 2) ? coeffs[2] : T(0);
        output[1] = coeffs[0] - T(0.5) * c_2;
    }

    // Compute a_0 so that P(0) = 0
    // For Chebyshev W polynomials: W_k(0) = 1 for all k >= 0
    // This is because W_n(cos(theta)) = sin((n+1/2)*theta) / sin(theta/2)
    // At theta = pi (x = -1): W_n(-1) = (-1)^n * (2n+1)
    // At theta = 0 (x = 1): W_n(1) = 2n + 1
    // At x = 0 (theta = pi/2): W_n(0) = sin((n+1/2)*pi/2) / sin(pi/4)
    //                                = sin((2n+1)*pi/4) / (sqrt(2)/2)
    // For n=0: sin(pi/4) * sqrt(2) = 1
    // For n=1: sin(3*pi/4) * sqrt(2) = 1
    // For n=2: sin(5*pi/4) * sqrt(2) = -1
    // For n=3: sin(7*pi/4) * sqrt(2) = -1
    // Pattern: W_n(0) = 1, 1, -1, -1, 1, 1, -1, -1, ...
    // i.e., W_n(0) = (-1)^floor(n/2) * ... but let's use the correct formula:
    // W_n(0) = cos(n*pi/2) = 0 for n odd, (-1)^(n/2) for n even
    // Actually, direct computation shows:
    // W_0(0) = 1, W_1(0) = 1, W_2(0) = -1, W_3(0) = -1, ...
    // W_n(0) = (-1)^(floor((n+1)/2)) for n >= 0
    // Simplified: W_n(0) = cos((n+1/2)*pi/2 - pi/4) pattern is complex.
    // Let's use a simple recursive approach or explicit formula.
    // From recurrence: W_n(0) = 2*0*W_{n-1}(0) - W_{n-2}(0) = -W_{n-2}(0)
    // W_0(0) = 1, W_1(0) = 2*0 + 1 = 1, W_2(0) = -W_0(0) = -1, W_3(0) = -W_1(0) = -1
    // W_4(0) = -W_2(0) = 1, W_5(0) = -W_3(0) = 1, ...
    // Pattern: 1, 1, -1, -1, 1, 1, -1, -1, ...
    // W_n(0) = (n % 4 < 2) ? 1 : -1
    T p_at_zero = T(0);
    for (int64_t k = 0; k <= N; ++k) {
        int64_t sign = ((k % 4) < 2) ? 1 : -1;
        p_at_zero += T(sign) * output[k];
    }
    output[0] = -p_at_zero;

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_w_antiderivative(
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
    const C half(T(0.5), T(0));

    // Initialize output to zero
    for (int64_t k = 0; k < output_size; ++k) {
        output[k] = zero;
    }

    // a_N = c_{N-1} / (2*N)
    output[N] = coeffs[N - 1] / C(T(2) * T(N), T(0));

    // a_k = (c_{k-1} - c_{k+1}) / (2*k) for k = 2..N-1
    for (int64_t k = N - 1; k >= 2; --k) {
        C c_km1 = coeffs[k - 1];
        C c_kp1 = (k + 1 < N) ? coeffs[k + 1] : zero;
        output[k] = (c_km1 - c_kp1) / C(T(2) * T(k), T(0));
    }

    // a_1 = c_0 - 0.5*c_2 (if N > 2)
    if (N >= 1) {
        C c_2 = (N > 2) ? coeffs[2] : zero;
        output[1] = coeffs[0] - half * c_2;
    }

    // Compute a_0 so that P(0) = 0
    // W_n(0) = (n % 4 < 2) ? 1 : -1
    C p_at_zero = zero;
    for (int64_t k = 0; k <= N; ++k) {
        int64_t sign = ((k % 4) < 2) ? 1 : -1;
        p_at_zero += C(T(sign), T(0)) * output[k];
    }
    output[0] = -p_at_zero;

    return output_size;
}

} // namespace torchscience::kernel::polynomial
