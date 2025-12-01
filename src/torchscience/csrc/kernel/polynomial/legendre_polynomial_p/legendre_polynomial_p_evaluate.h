#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Clenshaw's algorithm for Legendre P polynomial evaluation
// Evaluates p(x) = sum_{k=0}^{N-1} c_k * P_k(x)
//
// Legendre polynomials satisfy:
//   P_0(x) = 1
//   P_1(x) = x
//   P_{k+1}(x) = ((2k+1)/(k+1)) * x * P_k(x) - (k/(k+1)) * P_{k-1}(x)
//
// Clenshaw backward recurrence:
//   b_{n+1} = b_{n+2} = 0
//   b_k = c_k + A_k * x * b_{k+1} - C_{k+1} * b_{k+2}
//   where A_k = (2k+1)/(k+1), C_{k+1} = (k+1)/(k+2)
//   f(x) = b_0
//
// Parameters:
//   coeffs: pointer to N coefficients [c_0, c_1, ..., c_{N-1}]
//   x: evaluation point
//   N: number of coefficients
//
// Returns: p(x)
template <typename T>
T legendre_polynomial_p_evaluate(const T* coeffs, T x, int64_t N) {
    if (N == 0) {
        return T(0);
    }
    if (N == 1) {
        return coeffs[0];
    }

    T b_kp2 = T(0);  // b_{n+1}
    T b_kp1 = coeffs[N - 1];  // b_{n-1} = c_{n-1}

    // Clenshaw backward recurrence
    for (int64_t k = N - 2; k >= 0; --k) {
        // A_k = (2k+1)/(k+1)
        T a_k = T(2 * k + 1) / T(k + 1);
        // C_{k+1} = (k+1)/(k+2)
        T c_kp1 = T(k + 1) / T(k + 2);

        T b_k = coeffs[k] + a_k * x * b_kp1 - c_kp1 * b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }

    return b_kp1;
}

// Complex specialization
template <typename T>
c10::complex<T> legendre_polynomial_p_evaluate(
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
    C b_kp1 = coeffs[N - 1];

    for (int64_t k = N - 2; k >= 0; --k) {
        C a_k(T(2 * k + 1) / T(k + 1), T(0));
        C c_kp1(T(k + 1) / T(k + 2), T(0));

        C b_k = coeffs[k] + a_k * x * b_kp1 - c_kp1 * b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }

    return b_kp1;
}

} // namespace torchscience::kernel::polynomial
