#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for polynomial evaluation
// For y = p(x) = sum_{k=0}^{N-1} c_k * x^k:
//   dy/dc_k = x^k  (basis function at x)
//   dy/dx = sum_{k=1}^{N-1} k * c_k * x^{k-1} = p'(x)  (derivative polynomial)
//
// This kernel computes gradients for a single (coeffs, x) pair.
// The CPU backend handles batching.
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
T polynomial_evaluate_backward(
    T* grad_coeffs,
    T grad_output,
    const T* coeffs,
    T x,
    int64_t N
) {
    if (N == 0) {
        return T(0);
    }

    // grad_coeffs[k] = grad_output * x^k
    T x_pow = T(1);
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output * x_pow;
        x_pow *= x;
    }

    // grad_x = grad_output * p'(x)
    // For constant polynomial (N == 1), derivative is 0
    if (N == 1) {
        return T(0);
    }

    // Compute p'(x) using Horner's method on derivative coefficients
    // p'(x) = sum_{k=1}^{N-1} k * c_k * x^{k-1}
    //       = c_1 + 2*c_2*x + 3*c_3*x^2 + ... + (N-1)*c_{N-1}*x^{N-2}
    // Using Horner: start with (N-1)*c_{N-1}, then multiply by x and add k*c_k
    T deriv_result = T(N - 1) * coeffs[N - 1];
    for (int64_t k = N - 2; k >= 1; --k) {
        deriv_result = deriv_result * x + T(k) * coeffs[k];
    }

    return grad_output * deriv_result;
}

// Complex specialization
template <typename T>
c10::complex<T> polynomial_evaluate_backward(
    c10::complex<T>* grad_coeffs,
    c10::complex<T> grad_output,
    const c10::complex<T>* coeffs,
    c10::complex<T> x,
    int64_t N
) {
    if (N == 0) {
        return c10::complex<T>(T(0), T(0));
    }

    // grad_coeffs[k] = grad_output * x^k
    c10::complex<T> x_pow(T(1), T(0));
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output * x_pow;
        x_pow *= x;
    }

    if (N == 1) {
        return c10::complex<T>(T(0), T(0));
    }

    // Compute p'(x) using Horner's method
    c10::complex<T> deriv_result = c10::complex<T>(T(N - 1), T(0)) * coeffs[N - 1];
    for (int64_t k = N - 2; k >= 1; --k) {
        deriv_result = deriv_result * x + c10::complex<T>(T(k), T(0)) * coeffs[k];
    }

    return grad_output * deriv_result;
}

} // namespace torchscience::kernel::polynomial
