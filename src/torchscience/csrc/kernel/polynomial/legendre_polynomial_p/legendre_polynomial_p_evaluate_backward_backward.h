#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

namespace torchscience::kernel::polynomial {

// Second-order backward for Legendre P polynomial evaluation

template <typename T>
std::tuple<T, T> legendre_polynomial_p_evaluate_backward_backward(
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

    // Compute P_k(x) and P'_k(x)
    T P_km2 = T(1);  // P_0
    T P_km1 = x;     // P_1
    T Pp_km2 = T(0); // P'_0
    T Pp_km1 = T(1); // P'_1

    // k=0: grad_coeffs[0] = grad_output * P_0 = grad_output
    grad_grad_output += gg_coeffs[0] * P_km2;

    if (N >= 2) {
        grad_grad_output += gg_coeffs[1] * P_km1;
        grad_x_out += grad_output * gg_coeffs[1] * Pp_km1;
    }

    for (int64_t k = 2; k < N; ++k) {
        T P_k = (T(2 * k - 1) * x * P_km1 - T(k - 1) * P_km2) / T(k);
        T Pp_k = (T(2 * k - 1) / T(k)) * (P_km1 + x * Pp_km1) - (T(k - 1) / T(k)) * Pp_km2;

        grad_grad_output += gg_coeffs[k] * P_k;
        grad_x_out += grad_output * gg_coeffs[k] * Pp_k;

        P_km2 = P_km1;
        P_km1 = P_k;
        Pp_km2 = Pp_km1;
        Pp_km1 = Pp_k;
    }

    // Contribution from grad_x = grad_output * p'(x)
    if (N >= 2) {
        T deriv = T(0);

        P_km2 = T(1);
        P_km1 = x;
        Pp_km2 = T(0);
        Pp_km1 = T(1);

        deriv += coeffs[1] * Pp_km1;
        grad_coeffs_out[1] += gg_x * grad_output * Pp_km1;

        for (int64_t k = 2; k < N; ++k) {
            T P_k = (T(2 * k - 1) * x * P_km1 - T(k - 1) * P_km2) / T(k);
            T Pp_k = (T(2 * k - 1) / T(k)) * (P_km1 + x * Pp_km1) - (T(k - 1) / T(k)) * Pp_km2;

            deriv += coeffs[k] * Pp_k;
            grad_coeffs_out[k] += gg_x * grad_output * Pp_k;

            P_km2 = P_km1;
            P_km1 = P_k;
            Pp_km2 = Pp_km1;
            Pp_km1 = Pp_k;
        }

        grad_grad_output += gg_x * deriv;

        // Compute second derivative numerically
        if (N >= 3) {
            T eps = T(1e-6);

            auto compute_deriv = [&](T xv) -> T {
                T d = T(0);
                T Pkm2 = T(1), Pkm1 = xv;
                T Ppkm2 = T(0), Ppkm1 = T(1);

                d += coeffs[1] * Ppkm1;

                for (int64_t k = 2; k < N; ++k) {
                    T Pk = (T(2 * k - 1) * xv * Pkm1 - T(k - 1) * Pkm2) / T(k);
                    T Ppk = (T(2 * k - 1) / T(k)) * (Pkm1 + xv * Ppkm1) - (T(k - 1) / T(k)) * Ppkm2;

                    d += coeffs[k] * Ppk;

                    Pkm2 = Pkm1;
                    Pkm1 = Pk;
                    Ppkm2 = Ppkm1;
                    Ppkm1 = Ppk;
                }
                return d;
            };

            T deriv_plus = compute_deriv(x + eps);
            T deriv_minus = compute_deriv(x - eps);
            T second_deriv = (deriv_plus - deriv_minus) / (T(2) * eps);

            grad_x_out += gg_x * grad_output * second_deriv;
        }
    }

    return {grad_grad_output, grad_x_out};
}

// Complex specialization
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> legendre_polynomial_p_evaluate_backward_backward(
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

    C P_km2(T(1), T(0));
    C P_km1 = x;
    C Pp_km2(T(0), T(0));
    C Pp_km1(T(1), T(0));

    grad_grad_output += gg_coeffs[0] * P_km2;

    if (N >= 2) {
        grad_grad_output += gg_coeffs[1] * P_km1;
        grad_x_out += grad_output * gg_coeffs[1] * Pp_km1;
    }

    for (int64_t k = 2; k < N; ++k) {
        C P_k = (C(T(2 * k - 1), T(0)) * x * P_km1 - C(T(k - 1), T(0)) * P_km2) / C(T(k), T(0));
        C Pp_k = (C(T(2 * k - 1), T(0)) / C(T(k), T(0))) * (P_km1 + x * Pp_km1) -
                 (C(T(k - 1), T(0)) / C(T(k), T(0))) * Pp_km2;

        grad_grad_output += gg_coeffs[k] * P_k;
        grad_x_out += grad_output * gg_coeffs[k] * Pp_k;

        P_km2 = P_km1;
        P_km1 = P_k;
        Pp_km2 = Pp_km1;
        Pp_km1 = Pp_k;
    }

    if (N >= 2) {
        C deriv(T(0), T(0));

        P_km2 = C(T(1), T(0));
        P_km1 = x;
        Pp_km2 = C(T(0), T(0));
        Pp_km1 = C(T(1), T(0));

        deriv += coeffs[1] * Pp_km1;
        grad_coeffs_out[1] += gg_x * grad_output * Pp_km1;

        for (int64_t k = 2; k < N; ++k) {
            C P_k = (C(T(2 * k - 1), T(0)) * x * P_km1 - C(T(k - 1), T(0)) * P_km2) / C(T(k), T(0));
            C Pp_k = (C(T(2 * k - 1), T(0)) / C(T(k), T(0))) * (P_km1 + x * Pp_km1) -
                     (C(T(k - 1), T(0)) / C(T(k), T(0))) * Pp_km2;

            deriv += coeffs[k] * Pp_k;
            grad_coeffs_out[k] += gg_x * grad_output * Pp_k;

            P_km2 = P_km1;
            P_km1 = P_k;
            Pp_km2 = Pp_km1;
            Pp_km1 = Pp_k;
        }

        grad_grad_output += gg_x * deriv;
    }

    return {grad_grad_output, grad_x_out};
}

} // namespace torchscience::kernel::polynomial
