#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::polynomial {

// Backward pass for Chebyshev W polynomial evaluation
// For y = sum_{k=0}^{N-1} c_k * W_k(x):
//   dy/dc_k = W_k(x)  (Chebyshev W basis function)
//   dy/dx = sum_{k=1}^{N-1} c_k * W'_k(x)
//
// W polynomials: W_0=1, W_1=2x+1, W_k=2x*W_{k-1}-W_{k-2}
// W'_0(x) = 0
// W'_1(x) = 2
// W'_k(x) = 2*W_{k-1}(x) + 2x*W'_{k-1}(x) - W'_{k-2}(x)
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
T chebyshev_polynomial_w_evaluate_backward(
    T* grad_coeffs,
    T grad_output,
    const T* coeffs,
    T x,
    int64_t N
) {
    if (N == 0) {
        return T(0);
    }

    // Compute Chebyshev W basis: W_0=1, W_1=2x+1, W_k=2x*W_{k-1}-W_{k-2}
    T W_km2 = T(1);           // W_0
    T W_km1 = T(2) * x + T(1); // W_1

    grad_coeffs[0] = grad_output * W_km2;  // W_0(x) = 1

    if (N >= 2) {
        grad_coeffs[1] = grad_output * W_km1;  // W_1(x) = 2x + 1
    }

    for (int64_t k = 2; k < N; ++k) {
        T W_k = T(2) * x * W_km1 - W_km2;
        grad_coeffs[k] = grad_output * W_k;
        W_km2 = W_km1;
        W_km1 = W_k;
    }

    // Compute grad_x = grad_output * sum_{k=1}^{N-1} c_k * W'_k(x)
    if (N == 1) {
        return T(0);  // W_0 is constant
    }

    T deriv = T(0);

    // W'_0 = 0, W'_1 = 2
    T Wp_km2 = T(0);  // W'_0
    T Wp_km1 = T(2);  // W'_1

    // Reset W values
    W_km2 = T(1);           // W_0
    W_km1 = T(2) * x + T(1); // W_1

    deriv += coeffs[1] * Wp_km1;  // k=1: c_1 * W'_1(x) = c_1 * 2

    for (int64_t k = 2; k < N; ++k) {
        T W_k = T(2) * x * W_km1 - W_km2;
        // W'_k = 2*W_{k-1} + 2x*W'_{k-1} - W'_{k-2}
        T Wp_k = T(2) * W_km1 + T(2) * x * Wp_km1 - Wp_km2;

        deriv += coeffs[k] * Wp_k;

        W_km2 = W_km1;
        W_km1 = W_k;
        Wp_km2 = Wp_km1;
        Wp_km1 = Wp_k;
    }

    return grad_output * deriv;
}

// Complex specialization
template <typename T>
c10::complex<T> chebyshev_polynomial_w_evaluate_backward(
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

    C W_km2(T(1), T(0));
    C W_km1 = C(T(2), T(0)) * x + C(T(1), T(0));

    grad_coeffs[0] = grad_output * W_km2;

    if (N >= 2) {
        grad_coeffs[1] = grad_output * W_km1;
    }

    for (int64_t k = 2; k < N; ++k) {
        C W_k = C(T(2), T(0)) * x * W_km1 - W_km2;
        grad_coeffs[k] = grad_output * W_k;
        W_km2 = W_km1;
        W_km1 = W_k;
    }

    if (N == 1) {
        return C(T(0), T(0));
    }

    C deriv(T(0), T(0));

    C Wp_km2(T(0), T(0));
    C Wp_km1(T(2), T(0));

    W_km2 = C(T(1), T(0));
    W_km1 = C(T(2), T(0)) * x + C(T(1), T(0));

    deriv += coeffs[1] * Wp_km1;

    for (int64_t k = 2; k < N; ++k) {
        C W_k = C(T(2), T(0)) * x * W_km1 - W_km2;
        C Wp_k = C(T(2), T(0)) * W_km1 + C(T(2), T(0)) * x * Wp_km1 - Wp_km2;

        deriv += coeffs[k] * Wp_k;

        W_km2 = W_km1;
        W_km1 = W_k;
        Wp_km2 = Wp_km1;
        Wp_km1 = Wp_k;
    }

    return grad_output * deriv;
}

} // namespace torchscience::kernel::polynomial
