#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

namespace torchscience::kernel::polynomial {

// Second-order backward for Chebyshev W polynomial evaluation
//
// From backward:
//   grad_coeffs[k] = grad_output * W_k(x)
//   grad_x = grad_output * p'(x)
//
// This computes gradients of L w.r.t. (grad_output, coeffs, x) given
// gradients of L w.r.t. (grad_coeffs, grad_x).

template <typename T>
std::tuple<T, T> chebyshev_polynomial_w_evaluate_backward_backward(
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

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs_out[k] = T(0);
    }

    if (N == 0) {
        return {grad_grad_output, grad_x_out};
    }

    // Compute W_k(x) and W'_k(x) basis
    T W_km2 = T(1);           // W_0
    T W_km1 = T(2) * x + T(1); // W_1
    T Wp_km2 = T(0);          // W'_0
    T Wp_km1 = T(2);          // W'_1

    // Contribution from grad_coeffs[k] = grad_output * W_k(x)
    // dL/d(grad_output) += sum_k gg_coeffs[k] * W_k(x)
    // dL/d(x) += grad_output * sum_k gg_coeffs[k] * W'_k(x)

    grad_grad_output += gg_coeffs[0] * W_km2;  // k=0: W_0

    if (N >= 2) {
        grad_grad_output += gg_coeffs[1] * W_km1;  // k=1: W_1
        grad_x_out += grad_output * gg_coeffs[1] * Wp_km1;
    }

    for (int64_t k = 2; k < N; ++k) {
        T W_k = T(2) * x * W_km1 - W_km2;
        T Wp_k = T(2) * W_km1 + T(2) * x * Wp_km1 - Wp_km2;

        grad_grad_output += gg_coeffs[k] * W_k;
        grad_x_out += grad_output * gg_coeffs[k] * Wp_k;

        W_km2 = W_km1;
        W_km1 = W_k;
        Wp_km2 = Wp_km1;
        Wp_km1 = Wp_k;
    }

    // Contribution from grad_x = grad_output * p'(x)
    if (N >= 2) {
        T deriv = T(0);

        // Reset values
        W_km2 = T(1);
        W_km1 = T(2) * x + T(1);
        Wp_km2 = T(0);
        Wp_km1 = T(2);

        // k=1: W'_1 = 2
        deriv += coeffs[1] * Wp_km1;
        grad_coeffs_out[1] += gg_x * grad_output * Wp_km1;

        for (int64_t k = 2; k < N; ++k) {
            T W_k = T(2) * x * W_km1 - W_km2;
            T Wp_k = T(2) * W_km1 + T(2) * x * Wp_km1 - Wp_km2;

            deriv += coeffs[k] * Wp_k;
            grad_coeffs_out[k] += gg_x * grad_output * Wp_k;

            W_km2 = W_km1;
            W_km1 = W_k;
            Wp_km2 = Wp_km1;
            Wp_km1 = Wp_k;
        }

        grad_grad_output += gg_x * deriv;

        // dL/d(x) += gg_x * grad_output * p''(x)
        // Compute second derivative numerically
        if (N >= 3) {
            T eps = T(1e-6);
            T x_plus = x + eps;
            T x_minus = x - eps;

            auto compute_deriv = [&](T xv) -> T {
                T d = T(0);
                T Wkm2 = T(1);
                T Wkm1 = T(2) * xv + T(1);
                T Wpkm2 = T(0);
                T Wpkm1 = T(2);

                d += coeffs[1] * Wpkm1;

                for (int64_t k = 2; k < N; ++k) {
                    T Wk = T(2) * xv * Wkm1 - Wkm2;
                    T Wpk = T(2) * Wkm1 + T(2) * xv * Wpkm1 - Wpkm2;

                    d += coeffs[k] * Wpk;

                    Wkm2 = Wkm1;
                    Wkm1 = Wk;
                    Wpkm2 = Wpkm1;
                    Wpkm1 = Wpk;
                }
                return d;
            };

            T deriv_plus = compute_deriv(x_plus);
            T deriv_minus = compute_deriv(x_minus);
            T second_deriv = (deriv_plus - deriv_minus) / (T(2) * eps);

            grad_x_out += gg_x * grad_output * second_deriv;
        }
    }

    return {grad_grad_output, grad_x_out};
}

// Complex specialization
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> chebyshev_polynomial_w_evaluate_backward_backward(
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

    C W_km2(T(1), T(0));
    C W_km1 = C(T(2), T(0)) * x + C(T(1), T(0));
    C Wp_km2(T(0), T(0));
    C Wp_km1(T(2), T(0));

    grad_grad_output += gg_coeffs[0] * W_km2;

    if (N >= 2) {
        grad_grad_output += gg_coeffs[1] * W_km1;
        grad_x_out += grad_output * gg_coeffs[1] * Wp_km1;
    }

    for (int64_t k = 2; k < N; ++k) {
        C W_k = C(T(2), T(0)) * x * W_km1 - W_km2;
        C Wp_k = C(T(2), T(0)) * W_km1 + C(T(2), T(0)) * x * Wp_km1 - Wp_km2;

        grad_grad_output += gg_coeffs[k] * W_k;
        grad_x_out += grad_output * gg_coeffs[k] * Wp_k;

        W_km2 = W_km1;
        W_km1 = W_k;
        Wp_km2 = Wp_km1;
        Wp_km1 = Wp_k;
    }

    if (N >= 2) {
        C deriv(T(0), T(0));

        W_km2 = C(T(1), T(0));
        W_km1 = C(T(2), T(0)) * x + C(T(1), T(0));
        Wp_km2 = C(T(0), T(0));
        Wp_km1 = C(T(2), T(0));

        deriv += coeffs[1] * Wp_km1;
        grad_coeffs_out[1] += gg_x * grad_output * Wp_km1;

        for (int64_t k = 2; k < N; ++k) {
            C W_k = C(T(2), T(0)) * x * W_km1 - W_km2;
            C Wp_k = C(T(2), T(0)) * W_km1 + C(T(2), T(0)) * x * Wp_km1 - Wp_km2;

            deriv += coeffs[k] * Wp_k;
            grad_coeffs_out[k] += gg_x * grad_output * Wp_k;

            W_km2 = W_km1;
            W_km1 = W_k;
            Wp_km2 = Wp_km1;
            Wp_km1 = Wp_k;
        }

        grad_grad_output += gg_x * deriv;
    }

    return {grad_grad_output, grad_x_out};
}

} // namespace torchscience::kernel::polynomial
