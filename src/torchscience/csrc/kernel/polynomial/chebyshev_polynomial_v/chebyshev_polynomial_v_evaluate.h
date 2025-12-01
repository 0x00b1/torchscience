#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Clenshaw's algorithm for Chebyshev V polynomial evaluation
// Evaluates p(x) = sum_{k=0}^{N-1} c_k * V_k(x)
//
// V polynomials: V_0(x) = 1, V_1(x) = 2x - 1, V_{n+1}(x) = 2x*V_n(x) - V_{n-1}(x)
//
// Using: b_{n+1} = b_{n+2} = 0
//        b_k = c_k + 2*x*b_{k+1} - b_{k+2}  for k = n-1, n-2, ..., 1
//        f(x) = c_0 + (2*x - 1)*b_1 - b_2
//
// Parameters:
//   coeffs: pointer to N coefficients [c_0, c_1, ..., c_{N-1}]
//   x: evaluation point
//   N: number of coefficients
//
// Returns: p(x)
template <typename T>
T chebyshev_polynomial_v_evaluate(const T* coeffs, T x, int64_t N) {
    if (N == 0) {
        return T(0);
    }
    if (N == 1) {
        return coeffs[0];
    }

    T b_kp2 = T(0);  // b_{n+1}
    T b_kp1 = T(0);  // b_{n}

    // Clenshaw recurrence for k = N-1, N-2, ..., 1
    for (int64_t k = N - 1; k >= 1; --k) {
        T b_k = coeffs[k] + T(2) * x * b_kp1 - b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }

    // Final step for V polynomials: f(x) = c_0 + (2*x - 1)*b_1 - b_2
    return coeffs[0] + (T(2) * x - T(1)) * b_kp1 - b_kp2;
}

// Complex specialization
template <typename T>
c10::complex<T> chebyshev_polynomial_v_evaluate(
    const c10::complex<T>* coeffs,
    c10::complex<T> x,
    int64_t N
) {
    using C = c10::complex<T>;

    if (N == 0) {
        return C(T(0), T(0));
    }
    if (N == 1) {
        return coeffs[0];
    }

    C b_kp2(T(0), T(0));
    C b_kp1(T(0), T(0));

    for (int64_t k = N - 1; k >= 1; --k) {
        C b_k = coeffs[k] + C(T(2), T(0)) * x * b_kp1 - b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }

    return coeffs[0] + (C(T(2), T(0)) * x - C(T(1), T(0))) * b_kp1 - b_kp2;
}

} // namespace torchscience::kernel::polynomial
