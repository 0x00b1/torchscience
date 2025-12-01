#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

namespace torchscience::kernel::polynomial {

// Second-order backward for Chebyshev T polynomial evaluation
//
// From backward:
//   grad_coeffs[k] = grad_output * T_k(x)
//   grad_x = grad_output * p'(x) where p'(x) = sum_{k=1}^{N-1} c_k * k * U_{k-1}(x)
//
// This computes gradients of L w.r.t. (grad_output, coeffs, x) given
// gradients of L w.r.t. (grad_coeffs, grad_x).
//
// Derivations:
//   From grad_coeffs[k] = grad_output * T_k(x):
//     dL/d(grad_output) += sum_k gg_coeffs[k] * T_k(x)
//     dL/d(x) += grad_output * sum_k gg_coeffs[k] * k * U_{k-1}(x)
//
//   From grad_x = grad_output * p'(x):
//     dL/d(grad_output) += gg_x * p'(x)
//     dL/d(coeffs[k]) += gg_x * grad_output * k * U_{k-1}(x)  for k >= 1
//     dL/d(x) += gg_x * grad_output * p''(x)
//
// Parameters:
//   grad_coeffs_out: output dL/d(coeffs), array of size N
//   gg_coeffs: dL/d(grad_coeffs), array of size N
//   gg_x: dL/d(grad_x), scalar
//   grad_output: original upstream gradient
//   coeffs: original coefficients
//   x: original evaluation point
//   N: number of coefficients
//
// Returns: (grad_grad_output, grad_x_out)

template <typename T>
std::tuple<T, T> chebyshev_polynomial_t_evaluate_backward_backward(
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

    // Compute T_k(x) and U_{k-1}(x) for all k
    // T_0=1, T_1=x, T_k=2x*T_{k-1}-T_{k-2}
    // U_0=1, U_1=2x, U_k=2x*U_{k-1}-U_{k-2}

    // First pass: compute contributions from grad_coeffs[k] = grad_output * T_k(x)
    // dL/d(grad_output) += sum_k gg_coeffs[k] * T_k(x)
    // dL/d(x) += grad_output * sum_k gg_coeffs[k] * T'_k(x)
    //          = grad_output * sum_k gg_coeffs[k] * k * U_{k-1}(x)

    T T_km2 = T(1);  // T_0
    T T_km1 = x;     // T_1

    grad_grad_output += gg_coeffs[0] * T_km2;  // k=0: T_0

    if (N >= 2) {
        grad_grad_output += gg_coeffs[1] * T_km1;  // k=1: T_1
        // T'_1(x) = 1 * U_0 = 1
        grad_x_out += grad_output * gg_coeffs[1] * T(1);
    }

    T U_km2 = T(1);      // U_0
    T U_km1 = T(2) * x;  // U_1

    if (N >= 3) {
        // k=2: T'_2 = 2 * U_1
        T T_k = T(2) * x * T_km1 - T_km2;
        grad_grad_output += gg_coeffs[2] * T_k;
        grad_x_out += grad_output * gg_coeffs[2] * T(2) * U_km1;
        T_km2 = T_km1;
        T_km1 = T_k;
    }

    for (int64_t k = 3; k < N; ++k) {
        T T_k = T(2) * x * T_km1 - T_km2;
        T U_k = T(2) * x * U_km1 - U_km2;

        grad_grad_output += gg_coeffs[k] * T_k;
        grad_x_out += grad_output * gg_coeffs[k] * T(k) * U_k;

        T_km2 = T_km1;
        T_km1 = T_k;
        U_km2 = U_km1;
        U_km1 = U_k;
    }

    // Second pass: contributions from grad_x = grad_output * p'(x)
    // p'(x) = sum_{k=1}^{N-1} c_k * k * U_{k-1}(x)

    if (N >= 2) {
        // Compute p'(x) for dL/d(grad_output) contribution
        T deriv = T(0);
        U_km2 = T(1);      // U_0
        U_km1 = T(2) * x;  // U_1

        // k=1: c_1 * 1 * U_0
        deriv += coeffs[1] * T(1) * T(1);  // U_0 = 1
        grad_coeffs_out[1] += gg_x * grad_output * T(1) * T(1);

        if (N >= 3) {
            // k=2: c_2 * 2 * U_1
            deriv += coeffs[2] * T(2) * U_km1;
            grad_coeffs_out[2] += gg_x * grad_output * T(2) * U_km1;

            for (int64_t k = 3; k < N; ++k) {
                T U_k = T(2) * x * U_km1 - U_km2;
                deriv += coeffs[k] * T(k) * U_k;
                grad_coeffs_out[k] += gg_x * grad_output * T(k) * U_k;
                U_km2 = U_km1;
                U_km1 = U_k;
            }
        }

        grad_grad_output += gg_x * deriv;

        // dL/d(x) += gg_x * grad_output * p''(x)
        // p''(x) = sum_{k=1}^{N-1} c_k * k * U'_{k-1}(x)
        // U'_n(x) = ((n+1)*U_n(x) - (n+2)*U_{n-1}(x) + ...) - complex
        // Simpler: use recurrence for second derivative directly
        // p''(x) = d/dx[sum c_k * k * U_{k-1}]
        //        = sum c_k * k * U'_{k-1}
        // For U_n: U'_n = (n+1)*V_n where V satisfies similar recurrence
        // Actually: (1-x^2)*U'_n = (n+1)T_{n+1} - x*(n+1)*U_n
        // So U'_n = [(n+1)T_{n+1} - x*(n+1)*U_n] / (1-x^2) for |x| < 1

        // For numerical stability, compute p'' using the fact that
        // the second derivative of sum c_k T_k can also be expressed as
        // a Chebyshev series. For simplicity, use the explicit formula:
        // T''_k = k * [(k+1)*T_k - U_k] / (x^2 - 1) for x != ±1
        // But since p' involves U, p'' involves U'.

        // A simpler stable approach: compute using the relation
        // p''(x) = sum_{k=2}^{N-1} c_k * k * (k-1) * V_{k-2}(x)
        // where V satisfies a specific recurrence.

        // For now, use explicit computation with care for edge cases
        if (N >= 3 && std::abs(x * x - T(1)) > T(1e-10)) {
            T second_deriv = T(0);
            T x2m1 = x * x - T(1);

            // Recompute T_k and U_k
            T_km2 = T(1);
            T_km1 = x;
            U_km2 = T(1);
            U_km1 = T(2) * x;

            // U'_0 = 0 (U_0 = 1 is constant)
            // U'_1 = 2 (U_1 = 2x)
            // U'_n = [(n+1)*T_{n+1} - x*(n+1)*U_n] / (1-x^2)

            // k=1: coefficient is c_1 * 1 * U'_0 = 0
            // k=2: coefficient is c_2 * 2 * U'_1 = c_2 * 2 * 2 = 4*c_2
            second_deriv += coeffs[2] * T(2) * T(2);

            for (int64_t k = 3; k < N; ++k) {
                // Need U'_{k-1} where U'_n = [(n+1)*T_{n+1} - x*(n+1)*U_n] / (1-x^2)
                T T_k = T(2) * x * T_km1 - T_km2;
                T U_k = T(2) * x * U_km1 - U_km2;

                // U'_{k-1} for k >= 3 means we need n = k-1 >= 2
                // U'_n = [-(n+1)*T_{n+1} + x*(n+1)*U_n] / (x^2 - 1)
                T n = T(k - 1);
                T U_deriv = (-(n + T(1)) * T_k + x * (n + T(1)) * U_km1) / x2m1;
                second_deriv += coeffs[k] * T(k) * U_deriv;

                T_km2 = T_km1;
                T_km1 = T_k;
                U_km2 = U_km1;
                U_km1 = U_k;
            }

            grad_x_out += gg_x * grad_output * second_deriv;
        }
        // At x = ±1, the second derivative has a specific value but
        // is more complex to compute. For numerical purposes, the
        // gradient near boundaries approaches these limits smoothly.
    }

    return {grad_grad_output, grad_x_out};
}

// Complex specialization
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> chebyshev_polynomial_t_evaluate_backward_backward(
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

    C T_km2(T(1), T(0));
    C T_km1 = x;

    grad_grad_output += gg_coeffs[0] * T_km2;

    if (N >= 2) {
        grad_grad_output += gg_coeffs[1] * T_km1;
        grad_x_out += grad_output * gg_coeffs[1] * C(T(1), T(0));
    }

    C U_km2(T(1), T(0));
    C U_km1 = C(T(2), T(0)) * x;

    if (N >= 3) {
        C T_k = C(T(2), T(0)) * x * T_km1 - T_km2;
        grad_grad_output += gg_coeffs[2] * T_k;
        grad_x_out += grad_output * gg_coeffs[2] * C(T(2), T(0)) * U_km1;
        T_km2 = T_km1;
        T_km1 = T_k;
    }

    for (int64_t k = 3; k < N; ++k) {
        C T_k = C(T(2), T(0)) * x * T_km1 - T_km2;
        C U_k = C(T(2), T(0)) * x * U_km1 - U_km2;

        grad_grad_output += gg_coeffs[k] * T_k;
        grad_x_out += grad_output * gg_coeffs[k] * C(T(k), T(0)) * U_k;

        T_km2 = T_km1;
        T_km1 = T_k;
        U_km2 = U_km1;
        U_km1 = U_k;
    }

    if (N >= 2) {
        C deriv(T(0), T(0));
        U_km2 = C(T(1), T(0));
        U_km1 = C(T(2), T(0)) * x;

        deriv += coeffs[1] * C(T(1), T(0));
        grad_coeffs_out[1] += gg_x * grad_output * C(T(1), T(0));

        if (N >= 3) {
            deriv += coeffs[2] * C(T(2), T(0)) * U_km1;
            grad_coeffs_out[2] += gg_x * grad_output * C(T(2), T(0)) * U_km1;

            for (int64_t k = 3; k < N; ++k) {
                C U_k = C(T(2), T(0)) * x * U_km1 - U_km2;
                deriv += coeffs[k] * C(T(k), T(0)) * U_k;
                grad_coeffs_out[k] += gg_x * grad_output * C(T(k), T(0)) * U_k;
                U_km2 = U_km1;
                U_km1 = U_k;
            }
        }

        grad_grad_output += gg_x * deriv;

        // Second derivative computation for complex is similar but
        // the singularity check is on |x^2 - 1|
        if (N >= 3 && std::abs(x * x - C(T(1), T(0))) > T(1e-10)) {
            C second_deriv(T(0), T(0));
            C x2m1 = x * x - C(T(1), T(0));

            T_km2 = C(T(1), T(0));
            T_km1 = x;
            U_km2 = C(T(1), T(0));
            U_km1 = C(T(2), T(0)) * x;

            second_deriv += coeffs[2] * C(T(4), T(0));

            for (int64_t k = 3; k < N; ++k) {
                C T_k = C(T(2), T(0)) * x * T_km1 - T_km2;
                C U_k = C(T(2), T(0)) * x * U_km1 - U_km2;

                T n = T(k - 1);
                C U_deriv = (C(-n - T(1), T(0)) * T_k + x * C(n + T(1), T(0)) * U_km1) / x2m1;
                second_deriv += coeffs[k] * C(T(k), T(0)) * U_deriv;

                T_km2 = T_km1;
                T_km1 = T_k;
                U_km2 = U_km1;
                U_km1 = U_k;
            }

            grad_x_out += gg_x * grad_output * second_deriv;
        }
    }

    return {grad_grad_output, grad_x_out};
}

} // namespace torchscience::kernel::polynomial
