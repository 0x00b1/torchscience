#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Clenshaw's algorithm for Chebyshev U polynomial evaluation
// Evaluates p(x) = sum_{k=0}^{N-1} c_k * U_k(x)
// Using: b_{n+1} = b_{n+2} = 0
//        b_k = c_k + 2*x*b_{k+1} - b_{k+2}  for k = n-1, n-2, ..., 0
//        f(x) = b_0
//
// Parameters:
//   coeffs: pointer to N coefficients [c_0, c_1, ..., c_{N-1}]
//   x: evaluation point
//   N: number of coefficients
//
// Returns: p(x)
template <typename T>
T chebyshev_polynomial_u_evaluate(const T* coeffs, T x, int64_t N) {
    if (N == 0) {
        return T(0);
    }
    if (N == 1) {
        return coeffs[0];
    }

    T b_kp2 = T(0);  // b_{n+1}
    T b_kp1 = T(0);  // b_{n}

    for (int64_t k = N - 1; k >= 0; --k) {
        T b_k = coeffs[k] + T(2) * x * b_kp1 - b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }

    // f(x) = b_0
    return b_kp1;
}

// Complex specialization
template <typename T>
c10::complex<T> chebyshev_polynomial_u_evaluate(
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

    for (int64_t k = N - 1; k >= 0; --k) {
        C b_k = coeffs[k] + C(T(2), T(0)) * x * b_kp1 - b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }

    return b_kp1;
}

} // namespace torchscience::kernel::polynomial
