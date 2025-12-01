#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

namespace torchscience::kernel::polynomial {

// Second-order backward for Chebyshev V polynomial evaluation
//
// From backward:
//   grad_coeffs[k] = grad_output * V_k(x)
//   grad_x = grad_output * p'(x)
//
// This computes gradients of L w.r.t. (grad_output, coeffs, x) given
// gradients of L w.r.t. (grad_coeffs, grad_x).

template <typename T>
std::tuple<T, T> chebyshev_polynomial_v_evaluate_backward_backward(
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

    // Compute V_k(x) and V'_k(x) basis
    T V_km2 = T(1);           // V_0
    T V_km1 = T(2) * x - T(1); // V_1
    T Vp_km2 = T(0);          // V'_0
    T Vp_km1 = T(2);          // V'_1

    // Contribution from grad_coeffs[k] = grad_output * V_k(x)
    // dL/d(grad_output) += sum_k gg_coeffs[k] * V_k(x)
    // dL/d(x) += grad_output * sum_k gg_coeffs[k] * V'_k(x)

    grad_grad_output += gg_coeffs[0] * V_km2;  // k=0: V_0

    if (N >= 2) {
        grad_grad_output += gg_coeffs[1] * V_km1;  // k=1: V_1
        grad_x_out += grad_output * gg_coeffs[1] * Vp_km1;
    }

    for (int64_t k = 2; k < N; ++k) {
        T V_k = T(2) * x * V_km1 - V_km2;
        T Vp_k = T(2) * V_km1 + T(2) * x * Vp_km1 - Vp_km2;

        grad_grad_output += gg_coeffs[k] * V_k;
        grad_x_out += grad_output * gg_coeffs[k] * Vp_k;

        V_km2 = V_km1;
        V_km1 = V_k;
        Vp_km2 = Vp_km1;
        Vp_km1 = Vp_k;
    }

    // Contribution from grad_x = grad_output * p'(x)
    if (N >= 2) {
        T deriv = T(0);

        // Reset values
        V_km2 = T(1);
        V_km1 = T(2) * x - T(1);
        Vp_km2 = T(0);
        Vp_km1 = T(2);

        // k=1: V'_1 = 2
        deriv += coeffs[1] * Vp_km1;
        grad_coeffs_out[1] += gg_x * grad_output * Vp_km1;

        for (int64_t k = 2; k < N; ++k) {
            T V_k = T(2) * x * V_km1 - V_km2;
            T Vp_k = T(2) * V_km1 + T(2) * x * Vp_km1 - Vp_km2;

            deriv += coeffs[k] * Vp_k;
            grad_coeffs_out[k] += gg_x * grad_output * Vp_k;

            V_km2 = V_km1;
            V_km1 = V_k;
            Vp_km2 = Vp_km1;
            Vp_km1 = Vp_k;
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
                T Vkm2 = T(1);
                T Vkm1 = T(2) * xv - T(1);
                T Vpkm2 = T(0);
                T Vpkm1 = T(2);

                d += coeffs[1] * Vpkm1;

                for (int64_t k = 2; k < N; ++k) {
                    T Vk = T(2) * xv * Vkm1 - Vkm2;
                    T Vpk = T(2) * Vkm1 + T(2) * xv * Vpkm1 - Vpkm2;

                    d += coeffs[k] * Vpk;

                    Vkm2 = Vkm1;
                    Vkm1 = Vk;
                    Vpkm2 = Vpkm1;
                    Vpkm1 = Vpk;
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
std::tuple<c10::complex<T>, c10::complex<T>> chebyshev_polynomial_v_evaluate_backward_backward(
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

    C V_km2(T(1), T(0));
    C V_km1 = C(T(2), T(0)) * x - C(T(1), T(0));
    C Vp_km2(T(0), T(0));
    C Vp_km1(T(2), T(0));

    grad_grad_output += gg_coeffs[0] * V_km2;

    if (N >= 2) {
        grad_grad_output += gg_coeffs[1] * V_km1;
        grad_x_out += grad_output * gg_coeffs[1] * Vp_km1;
    }

    for (int64_t k = 2; k < N; ++k) {
        C V_k = C(T(2), T(0)) * x * V_km1 - V_km2;
        C Vp_k = C(T(2), T(0)) * V_km1 + C(T(2), T(0)) * x * Vp_km1 - Vp_km2;

        grad_grad_output += gg_coeffs[k] * V_k;
        grad_x_out += grad_output * gg_coeffs[k] * Vp_k;

        V_km2 = V_km1;
        V_km1 = V_k;
        Vp_km2 = Vp_km1;
        Vp_km1 = Vp_k;
    }

    if (N >= 2) {
        C deriv(T(0), T(0));

        V_km2 = C(T(1), T(0));
        V_km1 = C(T(2), T(0)) * x - C(T(1), T(0));
        Vp_km2 = C(T(0), T(0));
        Vp_km1 = C(T(2), T(0));

        deriv += coeffs[1] * Vp_km1;
        grad_coeffs_out[1] += gg_x * grad_output * Vp_km1;

        for (int64_t k = 2; k < N; ++k) {
            C V_k = C(T(2), T(0)) * x * V_km1 - V_km2;
            C Vp_k = C(T(2), T(0)) * V_km1 + C(T(2), T(0)) * x * Vp_km1 - Vp_km2;

            deriv += coeffs[k] * Vp_k;
            grad_coeffs_out[k] += gg_x * grad_output * Vp_k;

            V_km2 = V_km1;
            V_km1 = V_k;
            Vp_km2 = Vp_km1;
            Vp_km1 = Vp_k;
        }

        grad_grad_output += gg_x * deriv;
    }

    return {grad_grad_output, grad_x_out};
}

} // namespace torchscience::kernel::polynomial
