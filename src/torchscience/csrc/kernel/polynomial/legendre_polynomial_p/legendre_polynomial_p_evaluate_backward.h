#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::polynomial {

// Backward pass for Legendre P polynomial evaluation
// For y = sum_{k=0}^{N-1} c_k * P_k(x):
//   dy/dc_k = P_k(x)
//   dy/dx = sum_{k=1}^{N-1} c_k * P'_k(x)
//
// P polynomials: P_0=1, P_1=x, P_{k+1}=((2k+1)x*P_k - k*P_{k-1})/(k+1)
// P'_0 = 0, P'_1 = 1
// P'_{k+1} = ((2k+1)/(k+1)) * (P_k + x*P'_k) - (k/(k+1)) * P'_{k-1}
//
// Parameters:
//   grad_coeffs: output array for gradient w.r.t. coefficients (size N)
//   grad_output: upstream gradient (scalar)
//   coeffs: original coefficients [c_0, c_1, ..., c_{N-1}]
//   x: evaluation point
//   N: number of coefficients
//
// Returns: grad_x (gradient w.r.t. x)
template <typename T>
T legendre_polynomial_p_evaluate_backward(
    T* grad_coeffs,
    T grad_output,
    const T* coeffs,
    T x,
    int64_t N
) {
    if (N == 0) {
        return T(0);
    }

    // Compute Legendre basis values: P_0=1, P_1=x, P_{k+1}=((2k+1)x*P_k - k*P_{k-1})/(k+1)
    T P_km2 = T(1);  // P_0
    T P_km1 = x;     // P_1

    grad_coeffs[0] = grad_output * P_km2;  // P_0(x) = 1

    if (N >= 2) {
        grad_coeffs[1] = grad_output * P_km1;  // P_1(x) = x
    }

    for (int64_t k = 2; k < N; ++k) {
        // P_k = ((2(k-1)+1)*x*P_{k-1} - (k-1)*P_{k-2}) / k
        T P_k = (T(2 * k - 1) * x * P_km1 - T(k - 1) * P_km2) / T(k);
        grad_coeffs[k] = grad_output * P_k;
        P_km2 = P_km1;
        P_km1 = P_k;
    }

    // Compute grad_x = grad_output * sum_{k=1}^{N-1} c_k * P'_k(x)
    if (N == 1) {
        return T(0);  // P_0 is constant
    }

    T deriv = T(0);

    // P'_0 = 0, P'_1 = 1
    T Pp_km2 = T(0);  // P'_0
    T Pp_km1 = T(1);  // P'_1

    // Reset P values
    P_km2 = T(1);  // P_0
    P_km1 = x;     // P_1

    deriv += coeffs[1] * Pp_km1;  // k=1: c_1 * P'_1(x) = c_1 * 1

    for (int64_t k = 2; k < N; ++k) {
        T P_k = (T(2 * k - 1) * x * P_km1 - T(k - 1) * P_km2) / T(k);
        // P'_k = ((2(k-1)+1)/k) * (P_{k-1} + x*P'_{k-1}) - ((k-1)/k) * P'_{k-2}
        T Pp_k = (T(2 * k - 1) / T(k)) * (P_km1 + x * Pp_km1) - (T(k - 1) / T(k)) * Pp_km2;

        deriv += coeffs[k] * Pp_k;

        P_km2 = P_km1;
        P_km1 = P_k;
        Pp_km2 = Pp_km1;
        Pp_km1 = Pp_k;
    }

    return grad_output * deriv;
}

// Complex specialization
template <typename T>
c10::complex<T> legendre_polynomial_p_evaluate_backward(
    c10::complex<T>* grad_coeffs,
    c10::complex<T> grad_output,
    const c10::complex<T>* coeffs,
    c10::complex<T> x,
    int64_t N
) {
    using C = c10::complex<T>;

    if (N == 0) {
        return C(T(0), T(0));
    }

    C P_km2(T(1), T(0));
    C P_km1 = x;

    grad_coeffs[0] = grad_output * P_km2;

    if (N >= 2) {
        grad_coeffs[1] = grad_output * P_km1;
    }

    for (int64_t k = 2; k < N; ++k) {
        C P_k = (C(T(2 * k - 1), T(0)) * x * P_km1 - C(T(k - 1), T(0)) * P_km2) / C(T(k), T(0));
        grad_coeffs[k] = grad_output * P_k;
        P_km2 = P_km1;
        P_km1 = P_k;
    }

    if (N == 1) {
        return C(T(0), T(0));
    }

    C deriv(T(0), T(0));

    C Pp_km2(T(0), T(0));
    C Pp_km1(T(1), T(0));

    P_km2 = C(T(1), T(0));
    P_km1 = x;

    deriv += coeffs[1] * Pp_km1;

    for (int64_t k = 2; k < N; ++k) {
        C P_k = (C(T(2 * k - 1), T(0)) * x * P_km1 - C(T(k - 1), T(0)) * P_km2) / C(T(k), T(0));
        C Pp_k = (C(T(2 * k - 1), T(0)) / C(T(k), T(0))) * (P_km1 + x * Pp_km1) -
                 (C(T(k - 1), T(0)) / C(T(k), T(0))) * Pp_km2;

        deriv += coeffs[k] * Pp_k;

        P_km2 = P_km1;
        P_km1 = P_k;
        Pp_km2 = Pp_km1;
        Pp_km1 = Pp_k;
    }

    return grad_output * deriv;
}

} // namespace torchscience::kernel::polynomial
