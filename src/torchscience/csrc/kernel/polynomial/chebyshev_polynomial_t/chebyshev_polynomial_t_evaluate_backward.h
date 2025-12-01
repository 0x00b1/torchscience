#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for Chebyshev T polynomial evaluation
// For y = sum_{k=0}^{N-1} c_k * T_k(x):
//   dy/dc_k = T_k(x)  (Chebyshev basis function)
//   dy/dx = sum_{k=1}^{N-1} c_k * k * U_{k-1}(x)  (derivative)
//
// where T_k is Chebyshev of first kind: T_0=1, T_1=x, T_k=2x*T_{k-1}-T_{k-2}
// and U_k is Chebyshev of second kind: U_0=1, U_1=2x, U_k=2x*U_{k-1}-U_{k-2}
// with T'_k(x) = k * U_{k-1}(x)
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
T chebyshev_polynomial_t_evaluate_backward(
    T* grad_coeffs,
    T grad_output,
    const T* coeffs,
    T x,
    int64_t N
) {
    if (N == 0) {
        return T(0);
    }

    // Compute Chebyshev T basis: T_0=1, T_1=x, T_k=2x*T_{k-1}-T_{k-2}
    // grad_coeffs[k] = grad_output * T_k(x)
    T T_km2 = T(1);  // T_0
    T T_km1 = x;     // T_1

    grad_coeffs[0] = grad_output * T_km2;  // T_0(x) = 1

    if (N >= 2) {
        grad_coeffs[1] = grad_output * T_km1;  // T_1(x) = x
    }

    for (int64_t k = 2; k < N; ++k) {
        T T_k = T(2) * x * T_km1 - T_km2;
        grad_coeffs[k] = grad_output * T_k;
        T_km2 = T_km1;
        T_km1 = T_k;
    }

    // Compute grad_x = grad_output * sum_{k=1}^{N-1} c_k * k * U_{k-1}(x)
    // where U_0=1, U_1=2x, U_k=2x*U_{k-1}-U_{k-2}
    if (N == 1) {
        return T(0);  // Constant polynomial, derivative is 0
    }

    T deriv = T(0);

    // k=1: c_1 * 1 * U_0 = c_1 * 1 * 1 = c_1
    deriv += coeffs[1] * T(1);

    if (N >= 3) {
        // k=2: c_2 * 2 * U_1 = c_2 * 2 * 2x = 4*c_2*x
        T U_km2 = T(1);      // U_0
        T U_km1 = T(2) * x;  // U_1
        deriv += coeffs[2] * T(2) * U_km1;

        for (int64_t k = 3; k < N; ++k) {
            // U_{k-1} = 2x * U_{k-2} - U_{k-3}
            T U_k = T(2) * x * U_km1 - U_km2;
            deriv += coeffs[k] * T(k) * U_k;
            U_km2 = U_km1;
            U_km1 = U_k;
        }
    }

    return grad_output * deriv;
}

// Complex specialization
template <typename T>
c10::complex<T> chebyshev_polynomial_t_evaluate_backward(
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

    // Compute Chebyshev T basis
    C T_km2(T(1), T(0));
    C T_km1 = x;

    grad_coeffs[0] = grad_output * T_km2;

    if (N >= 2) {
        grad_coeffs[1] = grad_output * T_km1;
    }

    for (int64_t k = 2; k < N; ++k) {
        C T_k = C(T(2), T(0)) * x * T_km1 - T_km2;
        grad_coeffs[k] = grad_output * T_k;
        T_km2 = T_km1;
        T_km1 = T_k;
    }

    if (N == 1) {
        return C(T(0), T(0));
    }

    C deriv(T(0), T(0));

    deriv += coeffs[1] * C(T(1), T(0));

    if (N >= 3) {
        C U_km2(T(1), T(0));
        C U_km1 = C(T(2), T(0)) * x;
        deriv += coeffs[2] * C(T(2), T(0)) * U_km1;

        for (int64_t k = 3; k < N; ++k) {
            C U_k = C(T(2), T(0)) * x * U_km1 - U_km2;
            deriv += coeffs[k] * C(T(k), T(0)) * U_k;
            U_km2 = U_km1;
            U_km1 = U_k;
        }
    }

    return grad_output * deriv;
}

} // namespace torchscience::kernel::polynomial
