#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Chebyshev U polynomial antiderivative
//
// Given coefficients c[0..N-1] representing p(x) = sum_{k=0}^{N-1} c_k * U_k(x),
// computes the antiderivative P(x) = integral p(x) dx such that P(0) = 0.
//
// The integral of U_n(x) is:
//   integral U_n(x) dx = T_{n+1}(x) / (n+1)
//
// However, to express the antiderivative in the Chebyshev U basis, we use:
//   integral U_n(x) dx = U_{n+1}(x) / (2*(n+2)) - U_{n-1}(x) / (2*n)  for n >= 1
//   integral U_0(x) dx = U_1(x) / 4 + constant
//
// The antiderivative has coefficients a[0..N]:
//   The integration constant a_0 is chosen so P(0) = 0.
//
// Using the Chebyshev T representation is simpler:
//   integral U_n(x) dx = T_{n+1}(x) / (n+1)
//
// So for p(x) = sum c_k U_k(x), we have:
//   P(x) = sum c_k T_{k+1}(x) / (k+1) + C
//
// To express in Chebyshev U basis, we note that:
//   T_n(x) = (U_n(x) - U_{n-2}(x)) / 2  for n >= 2
//   T_1(x) = U_1(x) / 2
//   T_0(x) = U_0(x)
//
// Parameters:
//   output: array of size N + 1 to store antiderivative coefficients
//   coeffs: input coefficients [c_0, c_1, ..., c_{N-1}]
//   N: number of input coefficients
//
// Returns: size of output (N + 1)
template <typename T>
int64_t chebyshev_polynomial_u_antiderivative(
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

    // For Chebyshev U antiderivative, we use:
    // integral c_k U_k(x) dx contributes:
    //   For k = 0: integral U_0(x) dx = x = U_1(x)/2 + U_{-1}(x)/2
    //              Since U_{-1}(x) = 0, this is U_1(x)/2
    //              So c_0 contributes c_0/2 to coefficient of U_1
    //
    //   For k >= 1: integral U_k(x) dx = U_{k+1}(x)/(2(k+2)) - U_{k-1}(x)/(2k)
    //              So c_k contributes:
    //                c_k/(2(k+2)) to coefficient of U_{k+1}
    //                -c_k/(2k) to coefficient of U_{k-1}

    // c_0 contributes c_0/2 to output[1]
    output[1] += coeffs[0] / T(2);

    // For k >= 1
    for (int64_t k = 1; k < N; ++k) {
        // Contribution to U_{k+1}
        output[k + 1] += coeffs[k] / (T(2) * T(k + 2));
        // Contribution to U_{k-1}
        output[k - 1] -= coeffs[k] / (T(2) * T(k));
    }

    // Compute a_0 so that P(0) = 0
    // U_k(0) = (-1)^k * (k+1) for even k, and 0 for odd k? No.
    // Actually, U_k(0) = sin((k+1)*pi/2) / sin(pi/2) = sin((k+1)*pi/2)
    //   U_0(0) = 1, U_1(0) = 0, U_2(0) = -1, U_3(0) = 0, U_4(0) = 1, ...
    // So U_k(0) = (-1)^(k/2) for even k, 0 for odd k
    // P(0) = sum_{k even} a_k * (-1)^(k/2)
    T p_at_zero = T(0);
    for (int64_t k = 0; k <= N; k += 2) {
        int64_t sign = ((k / 2) % 2 == 0) ? 1 : -1;
        p_at_zero += T(sign) * output[k];
    }
    // We want P(0) = 0, so adjust a_0
    // Currently p_at_zero includes output[0] with sign +1
    // So output[0] + rest = 0 => output[0] = -rest
    output[0] = -p_at_zero;

    return output_size;
}

// Complex specialization
template <typename T>
int64_t chebyshev_polynomial_u_antiderivative(
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

    // c_0 contributes c_0/2 to output[1]
    output[1] += coeffs[0] / C(T(2), T(0));

    // For k >= 1
    for (int64_t k = 1; k < N; ++k) {
        // Contribution to U_{k+1}
        output[k + 1] += coeffs[k] / C(T(2) * T(k + 2), T(0));
        // Contribution to U_{k-1}
        output[k - 1] -= coeffs[k] / C(T(2) * T(k), T(0));
    }

    // Compute a_0 so that P(0) = 0
    C p_at_zero = zero;
    for (int64_t k = 0; k <= N; k += 2) {
        int64_t sign = ((k / 2) % 2 == 0) ? 1 : -1;
        p_at_zero += C(T(sign), T(0)) * output[k];
    }
    output[0] = -p_at_zero;

    return output_size;
}

} // namespace torchscience::kernel::polynomial
