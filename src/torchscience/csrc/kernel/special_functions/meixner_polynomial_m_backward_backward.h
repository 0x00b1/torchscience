#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "meixner_polynomial_m.h"

namespace torchscience::kernel::special_functions {

// Second-order backward for Meixner polynomial M_n(x; beta, c)
//
// All derivatives computed via finite differences.
template <typename T>
std::tuple<T, T, T, T, T> meixner_polynomial_m_backward_backward(
    T gradient_gradient_n,
    T gradient_gradient_x,
    T gradient_gradient_beta,
    T gradient_gradient_c,
    T gradient,
    T n,
    T x,
    T beta,
    T c
) {
    T eps = T(1e-7);
    T two_eps = T(2) * eps;

    // First derivatives dM/d{n,x,beta,c} for gradient_gradient_output
    T M_plus_n = meixner_polynomial_m(n + eps, x, beta, c);
    T M_minus_n = meixner_polynomial_m(n - eps, x, beta, c);
    T dM_dn = (M_plus_n - M_minus_n) / two_eps;

    T M_plus_x = meixner_polynomial_m(n, x + eps, beta, c);
    T M_minus_x = meixner_polynomial_m(n, x - eps, beta, c);
    T dM_dx = (M_plus_x - M_minus_x) / two_eps;

    T M_plus_beta = meixner_polynomial_m(n, x, beta + eps, c);
    T M_minus_beta = meixner_polynomial_m(n, x, beta - eps, c);
    T dM_dbeta = (M_plus_beta - M_minus_beta) / two_eps;

    T M_plus_c = meixner_polynomial_m(n, x, beta, c + eps);
    T M_minus_c = meixner_polynomial_m(n, x, beta, c - eps);
    T dM_dc = (M_plus_c - M_minus_c) / two_eps;

    // gradient_gradient_output = sum of gg_* * dM/d*
    T gradient_gradient_output = gradient_gradient_n * dM_dn +
                                  gradient_gradient_x * dM_dx +
                                  gradient_gradient_beta * dM_dbeta +
                                  gradient_gradient_c * dM_dc;

    // Second derivatives via finite differences for new gradients
    // d^2M/dx^2
    T M_center = meixner_polynomial_m(n, x, beta, c);
    T d2M_dx2 = (M_plus_x - T(2) * M_center + M_minus_x) / (eps * eps);

    // d^2M/dbeta^2
    T d2M_dbeta2 = (M_plus_beta - T(2) * M_center + M_minus_beta) / (eps * eps);

    // d^2M/dc^2
    T d2M_dc2 = (M_plus_c - T(2) * M_center + M_minus_c) / (eps * eps);

    // d^2M/dn^2
    T d2M_dn2 = (M_plus_n - T(2) * M_center + M_minus_n) / (eps * eps);

    // Compute new gradients (simplified: only diagonal second derivatives)
    T new_gradient_n = gradient_gradient_n * gradient * d2M_dn2;
    T new_gradient_x = gradient_gradient_x * gradient * d2M_dx2;
    T new_gradient_beta = gradient_gradient_beta * gradient * d2M_dbeta2;
    T new_gradient_c = gradient_gradient_c * gradient * d2M_dc2;

    return {gradient_gradient_output, new_gradient_n, new_gradient_x, new_gradient_beta, new_gradient_c};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
meixner_polynomial_m_backward_backward(
    c10::complex<T> gradient_gradient_n,
    c10::complex<T> gradient_gradient_x,
    c10::complex<T> gradient_gradient_beta,
    c10::complex<T> gradient_gradient_c,
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> x,
    c10::complex<T> beta,
    c10::complex<T> c
) {
    c10::complex<T> eps(T(1e-7), T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> two_eps = two * eps;

    // First derivatives dM/d{n,x,beta,c} for gradient_gradient_output
    c10::complex<T> M_plus_n = meixner_polynomial_m(n + eps, x, beta, c);
    c10::complex<T> M_minus_n = meixner_polynomial_m(n - eps, x, beta, c);
    c10::complex<T> dM_dn = (M_plus_n - M_minus_n) / two_eps;

    c10::complex<T> M_plus_x = meixner_polynomial_m(n, x + eps, beta, c);
    c10::complex<T> M_minus_x = meixner_polynomial_m(n, x - eps, beta, c);
    c10::complex<T> dM_dx = (M_plus_x - M_minus_x) / two_eps;

    c10::complex<T> M_plus_beta = meixner_polynomial_m(n, x, beta + eps, c);
    c10::complex<T> M_minus_beta = meixner_polynomial_m(n, x, beta - eps, c);
    c10::complex<T> dM_dbeta = (M_plus_beta - M_minus_beta) / two_eps;

    c10::complex<T> M_plus_c = meixner_polynomial_m(n, x, beta, c + eps);
    c10::complex<T> M_minus_c = meixner_polynomial_m(n, x, beta, c - eps);
    c10::complex<T> dM_dc = (M_plus_c - M_minus_c) / two_eps;

    // gradient_gradient_output = sum of gg_* * dM/d*
    c10::complex<T> gradient_gradient_output = gradient_gradient_n * dM_dn +
                                                gradient_gradient_x * dM_dx +
                                                gradient_gradient_beta * dM_dbeta +
                                                gradient_gradient_c * dM_dc;

    // Second derivatives via finite differences
    c10::complex<T> M_center = meixner_polynomial_m(n, x, beta, c);
    c10::complex<T> d2M_dx2 = (M_plus_x - two * M_center + M_minus_x) / (eps * eps);
    c10::complex<T> d2M_dbeta2 = (M_plus_beta - two * M_center + M_minus_beta) / (eps * eps);
    c10::complex<T> d2M_dc2 = (M_plus_c - two * M_center + M_minus_c) / (eps * eps);
    c10::complex<T> d2M_dn2 = (M_plus_n - two * M_center + M_minus_n) / (eps * eps);

    // Compute new gradients
    c10::complex<T> new_gradient_n = gradient_gradient_n * gradient * d2M_dn2;
    c10::complex<T> new_gradient_x = gradient_gradient_x * gradient * d2M_dx2;
    c10::complex<T> new_gradient_beta = gradient_gradient_beta * gradient * d2M_dbeta2;
    c10::complex<T> new_gradient_c = gradient_gradient_c * gradient * d2M_dc2;

    return {gradient_gradient_output, new_gradient_n, new_gradient_x, new_gradient_beta, new_gradient_c};
}

} // namespace torchscience::kernel::special_functions
