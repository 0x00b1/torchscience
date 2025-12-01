#pragma once

#include <c10/util/complex.h>
#include <tuple>

namespace torchscience::kernel::polynomial {

// Second-order backward (backward of backward) for polynomial evaluation
//
// Let L be the final loss.
// From forward: y = p(x) = sum c_k * x^k
// From backward:
//   grad_coeffs[k] = grad_output * x^k
//   grad_x = grad_output * p'(x)
//
// This computes gradients of L w.r.t. (grad_output, coeffs, x) given
// gradients of L w.r.t. (grad_coeffs, grad_x).
//
// Inputs:
//   gg_coeffs: dL/d(grad_coeffs), array of size N
//   gg_x: dL/d(grad_x), scalar
//   grad_output: original upstream gradient
//   coeffs: original coefficients
//   x: original evaluation point
//   N: number of coefficients
//
// Outputs:
//   grad_grad_output: dL/d(grad_output)
//   grad_coeffs_out: dL/d(coeffs), array of size N
//   grad_x_out: dL/d(x)
//
// Derivations:
//   grad_coeffs[k] = grad_output * x^k
//     => dL/d(grad_output) += gg_coeffs[k] * x^k = sum_k gg_coeffs[k] * x^k
//     => dL/d(x) += grad_output * k * gg_coeffs[k] * x^{k-1}
//
//   grad_x = grad_output * p'(x)
//     => dL/d(grad_output) += gg_x * p'(x)
//     => dL/d(coeffs[k]) += gg_x * grad_output * k * x^{k-1}  (for k >= 1)
//     => dL/d(x) += gg_x * grad_output * p''(x)

template <typename T>
std::tuple<T, T> polynomial_evaluate_backward_backward(
    T* grad_coeffs_out,
    const T* gg_coeffs,
    T gg_x,
    T grad_output,
    const T* coeffs,
    T x,
    int64_t N
) {
    T grad_grad_output = T(0);
    T grad_x_out = T(0);

    // Initialize grad_coeffs_out to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs_out[k] = T(0);
    }

    if (N == 0) {
        return {grad_grad_output, grad_x_out};
    }

    // Contribution from grad_coeffs[k] = grad_output * x^k
    // dL/d(grad_output) += sum_k gg_coeffs[k] * x^k
    // dL/d(x) += sum_{k>=1} grad_output * k * gg_coeffs[k] * x^{k-1}
    T x_pow = T(1);
    for (int64_t k = 0; k < N; ++k) {
        grad_grad_output += gg_coeffs[k] * x_pow;
        if (k >= 1) {
            grad_x_out += grad_output * T(k) * gg_coeffs[k] * (x_pow / x);
        }
        x_pow *= x;
    }

    // Handle x=0 case for the x^{k-1} terms
    // When x=0, only k=1 term contributes (x^0 = 1)
    if (x == T(0)) {
        grad_x_out = T(0);
        if (N >= 2) {
            grad_x_out = grad_output * gg_coeffs[1];
        }
    }

    if (N == 1) {
        // p'(x) = 0 for constant, no gg_x contribution
        return {grad_grad_output, grad_x_out};
    }

    // Contribution from grad_x = grad_output * p'(x)
    // p'(x) = sum_{k=1}^{N-1} k * c_k * x^{k-1}

    // dL/d(grad_output) += gg_x * p'(x)
    T deriv_at_x = T(N - 1) * coeffs[N - 1];
    for (int64_t k = N - 2; k >= 1; --k) {
        deriv_at_x = deriv_at_x * x + T(k) * coeffs[k];
    }
    grad_grad_output += gg_x * deriv_at_x;

    // dL/d(coeffs[k]) += gg_x * grad_output * k * x^{k-1}  for k >= 1
    x_pow = T(1);  // x^0
    for (int64_t k = 1; k < N; ++k) {
        grad_coeffs_out[k] += gg_x * grad_output * T(k) * x_pow;
        x_pow *= x;
    }

    // dL/d(x) += gg_x * grad_output * p''(x)
    // p''(x) = sum_{k=2}^{N-1} k*(k-1) * c_k * x^{k-2}
    if (N > 2) {
        T second_deriv = T(N - 1) * T(N - 2) * coeffs[N - 1];
        for (int64_t k = N - 2; k >= 2; --k) {
            second_deriv = second_deriv * x + T(k) * T(k - 1) * coeffs[k];
        }
        grad_x_out += gg_x * grad_output * second_deriv;
    }

    return {grad_grad_output, grad_x_out};
}

// Complex specialization
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> polynomial_evaluate_backward_backward(
    c10::complex<T>* grad_coeffs_out,
    const c10::complex<T>* gg_coeffs,
    c10::complex<T> gg_x,
    c10::complex<T> grad_output,
    const c10::complex<T>* coeffs,
    c10::complex<T> x,
    int64_t N
) {
    using C = c10::complex<T>;
    C grad_grad_output(T(0), T(0));
    C grad_x_out(T(0), T(0));

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs_out[k] = C(T(0), T(0));
    }

    if (N == 0) {
        return {grad_grad_output, grad_x_out};
    }

    // Contribution from grad_coeffs[k] = grad_output * x^k
    C x_pow(T(1), T(0));
    for (int64_t k = 0; k < N; ++k) {
        grad_grad_output += gg_coeffs[k] * x_pow;
        if (k >= 1 && std::abs(x) > T(0)) {
            grad_x_out += grad_output * C(T(k), T(0)) * gg_coeffs[k] * (x_pow / x);
        }
        x_pow *= x;
    }

    // Handle x=0 case
    if (std::abs(x) == T(0) && N >= 2) {
        grad_x_out = grad_output * gg_coeffs[1];
    }

    if (N == 1) {
        return {grad_grad_output, grad_x_out};
    }

    // Contribution from grad_x = grad_output * p'(x)
    C deriv_at_x = C(T(N - 1), T(0)) * coeffs[N - 1];
    for (int64_t k = N - 2; k >= 1; --k) {
        deriv_at_x = deriv_at_x * x + C(T(k), T(0)) * coeffs[k];
    }
    grad_grad_output += gg_x * deriv_at_x;

    // dL/d(coeffs[k]) += gg_x * grad_output * k * x^{k-1}
    x_pow = C(T(1), T(0));
    for (int64_t k = 1; k < N; ++k) {
        grad_coeffs_out[k] += gg_x * grad_output * C(T(k), T(0)) * x_pow;
        x_pow *= x;
    }

    // dL/d(x) += gg_x * grad_output * p''(x)
    if (N > 2) {
        C second_deriv = C(T(N - 1) * T(N - 2), T(0)) * coeffs[N - 1];
        for (int64_t k = N - 2; k >= 2; --k) {
            second_deriv = second_deriv * x + C(T(k) * T(k - 1), T(0)) * coeffs[k];
        }
        grad_x_out += gg_x * grad_output * second_deriv;
    }

    return {grad_grad_output, grad_x_out};
}

} // namespace torchscience::kernel::polynomial
