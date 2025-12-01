#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::polynomial {

// Backward pass for Chebyshev U polynomial evaluation
// For y = sum_{k=0}^{N-1} c_k * U_k(x):
//   dy/dc_k = U_k(x)  (Chebyshev U basis function)
//   dy/dx = sum_{k=1}^{N-1} c_k * U'_k(x)
//
// where U_k: U_0=1, U_1=2x, U_k=2x*U_{k-1}-U_{k-2}
// and U'_k(x) = [(k+1)*T_{k+1}(x) - x*U_k(x)] / (x^2 - 1)
// where T_k: T_0=1, T_1=x, T_k=2x*T_{k-1}-T_{k-2}
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
T chebyshev_polynomial_u_evaluate_backward(
    T* grad_coeffs,
    T grad_output,
    const T* coeffs,
    T x,
    int64_t N
) {
    if (N == 0) {
        return T(0);
    }

    // Compute Chebyshev U basis: U_0=1, U_1=2x, U_k=2x*U_{k-1}-U_{k-2}
    // grad_coeffs[k] = grad_output * U_k(x)
    T U_km2 = T(1);      // U_0
    T U_km1 = T(2) * x;  // U_1

    grad_coeffs[0] = grad_output * U_km2;  // U_0(x) = 1

    if (N >= 2) {
        grad_coeffs[1] = grad_output * U_km1;  // U_1(x) = 2x
    }

    for (int64_t k = 2; k < N; ++k) {
        T U_k = T(2) * x * U_km1 - U_km2;
        grad_coeffs[k] = grad_output * U_k;
        U_km2 = U_km1;
        U_km1 = U_k;
    }

    // Compute grad_x = grad_output * sum_{k=1}^{N-1} c_k * U'_k(x)
    // U'_k(x) = [(k+1)*T_{k+1}(x) - x*U_k(x)] / (x^2 - 1)
    if (N == 1) {
        return T(0);  // U_0 is constant, derivative is 0
    }

    T x2m1 = x * x - T(1);
    T deriv = T(0);

    // Reset U values
    U_km2 = T(1);      // U_0
    U_km1 = T(2) * x;  // U_1

    // Compute T values: T_0=1, T_1=x, T_k=2x*T_{k-1}-T_{k-2}
    T T_km2 = T(1);  // T_0
    T T_km1 = x;     // T_1
    T T_k = T(2) * x * T_km1 - T_km2;  // T_2

    // k=1: U'_1(x) = [2*T_2 - x*U_1] / (x^2-1) = [2*(2x^2-1) - x*2x] / (x^2-1) = 2
    // Special case for numerical stability
    deriv += coeffs[1] * T(2);

    if (N >= 3) {
        // For k >= 2, use the formula
        T_km2 = T_km1;  // T_1
        T_km1 = T_k;    // T_2

        for (int64_t k = 2; k < N; ++k) {
            T U_k = T(2) * x * U_km1 - U_km2;
            T_k = T(2) * x * T_km1 - T_km2;  // T_{k+1}

            // U'_k = [(k+1)*T_{k+1} - x*U_k] / (x^2-1)
            if (std::abs(x2m1) > T(1e-10)) {
                T U_deriv = (T(k + 1) * T_k - x * U_k) / x2m1;
                deriv += coeffs[k] * U_deriv;
            } else {
                // At x = ±1, use L'Hopital or limit formula
                // U_k(1) = k+1, U_k(-1) = (-1)^k * (k+1)
                // U'_k(±1) has a specific value
                // For simplicity, use numerical approximation near boundary
                T eps = T(1e-6);
                T x_plus = x + eps;
                T x_minus = x - eps;

                // Recompute U_k at perturbed points
                T U_p_km2 = T(1), U_p_km1 = T(2) * x_plus;
                T U_m_km2 = T(1), U_m_km1 = T(2) * x_minus;
                for (int64_t j = 2; j <= k; ++j) {
                    T U_p = T(2) * x_plus * U_p_km1 - U_p_km2;
                    T U_m = T(2) * x_minus * U_m_km1 - U_m_km2;
                    U_p_km2 = U_p_km1; U_p_km1 = U_p;
                    U_m_km2 = U_m_km1; U_m_km1 = U_m;
                }
                T U_deriv = (U_p_km1 - U_m_km1) / (T(2) * eps);
                deriv += coeffs[k] * U_deriv;
            }

            U_km2 = U_km1;
            U_km1 = U_k;
            T_km2 = T_km1;
            T_km1 = T_k;
        }
    }

    return grad_output * deriv;
}

// Complex specialization
template <typename T>
c10::complex<T> chebyshev_polynomial_u_evaluate_backward(
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

    C U_km2(T(1), T(0));
    C U_km1 = C(T(2), T(0)) * x;

    grad_coeffs[0] = grad_output * U_km2;

    if (N >= 2) {
        grad_coeffs[1] = grad_output * U_km1;
    }

    for (int64_t k = 2; k < N; ++k) {
        C U_k = C(T(2), T(0)) * x * U_km1 - U_km2;
        grad_coeffs[k] = grad_output * U_k;
        U_km2 = U_km1;
        U_km1 = U_k;
    }

    if (N == 1) {
        return C(T(0), T(0));
    }

    C x2m1 = x * x - C(T(1), T(0));
    C deriv(T(0), T(0));

    U_km2 = C(T(1), T(0));
    U_km1 = C(T(2), T(0)) * x;

    C T_km2(T(1), T(0));
    C T_km1 = x;
    C T_k = C(T(2), T(0)) * x * T_km1 - T_km2;

    deriv += coeffs[1] * C(T(2), T(0));

    if (N >= 3) {
        T_km2 = T_km1;
        T_km1 = T_k;

        for (int64_t k = 2; k < N; ++k) {
            C U_k = C(T(2), T(0)) * x * U_km1 - U_km2;
            T_k = C(T(2), T(0)) * x * T_km1 - T_km2;

            if (std::abs(x2m1) > T(1e-10)) {
                C U_deriv = (C(T(k + 1), T(0)) * T_k - x * U_k) / x2m1;
                deriv += coeffs[k] * U_deriv;
            }

            U_km2 = U_km1;
            U_km1 = U_k;
            T_km2 = T_km1;
            T_km1 = T_k;
        }
    }

    return grad_output * deriv;
}

} // namespace torchscience::kernel::polynomial
