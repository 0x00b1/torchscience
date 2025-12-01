#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

namespace torchscience::kernel::polynomial {

// Second-order backward for Chebyshev U polynomial evaluation
// Similar structure to Chebyshev T backward_backward
//
// From backward:
//   grad_coeffs[k] = grad_output * U_k(x)
//   grad_x = grad_output * p'(x)
//
// This computes gradients of L w.r.t. (grad_output, coeffs, x) given
// gradients of L w.r.t. (grad_coeffs, grad_x).

template <typename T>
std::tuple<T, T> chebyshev_polynomial_u_evaluate_backward_backward(
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

    // Compute U_k(x) basis
    T U_km2 = T(1);
    T U_km1 = T(2) * x;

    // Contribution from grad_coeffs[k] = grad_output * U_k(x)
    // dL/d(grad_output) += sum_k gg_coeffs[k] * U_k(x)
    // dL/d(x) += grad_output * sum_k gg_coeffs[k] * U'_k(x)

    grad_grad_output += gg_coeffs[0] * U_km2;  // k=0

    if (N >= 2) {
        grad_grad_output += gg_coeffs[1] * U_km1;  // k=1
        // U'_1(x) = 2
        grad_x_out += grad_output * gg_coeffs[1] * T(2);
    }

    T T_km2 = T(1);
    T T_km1 = x;
    T T_k = T(2) * x * T_km1 - T_km2;
    T x2m1 = x * x - T(1);

    if (N >= 3) {
        T_km2 = T_km1;
        T_km1 = T_k;

        for (int64_t k = 2; k < N; ++k) {
            T U_k = T(2) * x * U_km1 - U_km2;
            T_k = T(2) * x * T_km1 - T_km2;

            grad_grad_output += gg_coeffs[k] * U_k;

            if (std::abs(x2m1) > T(1e-10)) {
                T U_deriv = (T(k + 1) * T_k - x * U_k) / x2m1;
                grad_x_out += grad_output * gg_coeffs[k] * U_deriv;
            }

            U_km2 = U_km1;
            U_km1 = U_k;
            T_km2 = T_km1;
            T_km1 = T_k;
        }
    }

    // Contribution from grad_x = grad_output * p'(x)
    if (N >= 2) {
        // Compute p'(x) for dL/d(grad_output) contribution
        T deriv = T(0);

        U_km2 = T(1);
        U_km1 = T(2) * x;
        T_km2 = T(1);
        T_km1 = x;
        T_k = T(2) * x * T_km1 - T_km2;

        // k=1: U'_1 = 2
        deriv += coeffs[1] * T(2);
        grad_coeffs_out[1] += gg_x * grad_output * T(2);

        if (N >= 3) {
            T_km2 = T_km1;
            T_km1 = T_k;

            for (int64_t k = 2; k < N; ++k) {
                T U_k = T(2) * x * U_km1 - U_km2;
                T_k = T(2) * x * T_km1 - T_km2;

                if (std::abs(x2m1) > T(1e-10)) {
                    T U_deriv = (T(k + 1) * T_k - x * U_k) / x2m1;
                    deriv += coeffs[k] * U_deriv;
                    grad_coeffs_out[k] += gg_x * grad_output * U_deriv;
                }

                U_km2 = U_km1;
                U_km1 = U_k;
                T_km2 = T_km1;
                T_km1 = T_k;
            }
        }

        grad_grad_output += gg_x * deriv;

        // dL/d(x) += gg_x * grad_output * p''(x)
        // For simplicity, compute p'' numerically for second derivative
        if (N >= 3 && std::abs(x2m1) > T(1e-10)) {
            T eps = T(1e-6);
            T x_plus = x + eps;
            T x_minus = x - eps;

            // Compute p'(x+eps) and p'(x-eps) to get p''
            auto compute_deriv = [&](T xv) -> T {
                T d = T(0);
                T Ukm2 = T(1), Ukm1 = T(2) * xv;
                T Tkm2 = T(1), Tkm1 = xv;
                T Tk = T(2) * xv * Tkm1 - Tkm2;
                T xv2m1 = xv * xv - T(1);

                d += coeffs[1] * T(2);

                Tkm2 = Tkm1;
                Tkm1 = Tk;

                for (int64_t k = 2; k < N; ++k) {
                    T Uk = T(2) * xv * Ukm1 - Ukm2;
                    Tk = T(2) * xv * Tkm1 - Tkm2;

                    if (std::abs(xv2m1) > T(1e-10)) {
                        T Ud = (T(k + 1) * Tk - xv * Uk) / xv2m1;
                        d += coeffs[k] * Ud;
                    }

                    Ukm2 = Ukm1;
                    Ukm1 = Uk;
                    Tkm2 = Tkm1;
                    Tkm1 = Tk;
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
std::tuple<c10::complex<T>, c10::complex<T>> chebyshev_polynomial_u_evaluate_backward_backward(
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

    C U_km2(T(1), T(0));
    C U_km1 = C(T(2), T(0)) * x;

    grad_grad_output += gg_coeffs[0] * U_km2;

    if (N >= 2) {
        grad_grad_output += gg_coeffs[1] * U_km1;
        grad_x_out += grad_output * gg_coeffs[1] * C(T(2), T(0));
    }

    C T_km2(T(1), T(0));
    C T_km1 = x;
    C T_k = C(T(2), T(0)) * x * T_km1 - T_km2;
    C x2m1 = x * x - C(T(1), T(0));

    if (N >= 3) {
        T_km2 = T_km1;
        T_km1 = T_k;

        for (int64_t k = 2; k < N; ++k) {
            C U_k = C(T(2), T(0)) * x * U_km1 - U_km2;
            T_k = C(T(2), T(0)) * x * T_km1 - T_km2;

            grad_grad_output += gg_coeffs[k] * U_k;

            if (std::abs(x2m1) > T(1e-10)) {
                C U_deriv = (C(T(k + 1), T(0)) * T_k - x * U_k) / x2m1;
                grad_x_out += grad_output * gg_coeffs[k] * U_deriv;
            }

            U_km2 = U_km1;
            U_km1 = U_k;
            T_km2 = T_km1;
            T_km1 = T_k;
        }
    }

    if (N >= 2) {
        C deriv(T(0), T(0));

        U_km2 = C(T(1), T(0));
        U_km1 = C(T(2), T(0)) * x;
        T_km2 = C(T(1), T(0));
        T_km1 = x;
        T_k = C(T(2), T(0)) * x * T_km1 - T_km2;

        deriv += coeffs[1] * C(T(2), T(0));
        grad_coeffs_out[1] += gg_x * grad_output * C(T(2), T(0));

        if (N >= 3) {
            T_km2 = T_km1;
            T_km1 = T_k;

            for (int64_t k = 2; k < N; ++k) {
                C U_k = C(T(2), T(0)) * x * U_km1 - U_km2;
                T_k = C(T(2), T(0)) * x * T_km1 - T_km2;

                if (std::abs(x2m1) > T(1e-10)) {
                    C U_deriv = (C(T(k + 1), T(0)) * T_k - x * U_k) / x2m1;
                    deriv += coeffs[k] * U_deriv;
                    grad_coeffs_out[k] += gg_x * grad_output * U_deriv;
                }

                U_km2 = U_km1;
                U_km1 = U_k;
                T_km2 = T_km1;
                T_km1 = T_k;
            }
        }

        grad_grad_output += gg_x * deriv;
    }

    return {grad_grad_output, grad_x_out};
}

} // namespace torchscience::kernel::polynomial
