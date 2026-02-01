#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "hahn_polynomial_q.h"

namespace torchscience::kernel::special_functions {

// Second-order backward for Hahn polynomial Q_n(x; alpha, beta, N)
//
// All derivatives computed via finite differences.
template <typename T>
std::tuple<T, T, T, T, T, T> hahn_polynomial_q_backward_backward(
    T gradient_gradient_n,
    T gradient_gradient_x,
    T gradient_gradient_alpha,
    T gradient_gradient_beta,
    T gradient_gradient_N,
    T gradient,
    T n,
    T x,
    T alpha,
    T beta,
    T N
) {
    T eps = T(1e-7);
    T two_eps = T(2) * eps;

    // First derivatives dQ/d{n,x,alpha,beta,N} for gradient_gradient_output
    T Q_plus_n = hahn_polynomial_q(n + eps, x, alpha, beta, N);
    T Q_minus_n = hahn_polynomial_q(n - eps, x, alpha, beta, N);
    T dQ_dn = (Q_plus_n - Q_minus_n) / two_eps;

    T Q_plus_x = hahn_polynomial_q(n, x + eps, alpha, beta, N);
    T Q_minus_x = hahn_polynomial_q(n, x - eps, alpha, beta, N);
    T dQ_dx = (Q_plus_x - Q_minus_x) / two_eps;

    T Q_plus_alpha = hahn_polynomial_q(n, x, alpha + eps, beta, N);
    T Q_minus_alpha = hahn_polynomial_q(n, x, alpha - eps, beta, N);
    T dQ_dalpha = (Q_plus_alpha - Q_minus_alpha) / two_eps;

    T Q_plus_beta = hahn_polynomial_q(n, x, alpha, beta + eps, N);
    T Q_minus_beta = hahn_polynomial_q(n, x, alpha, beta - eps, N);
    T dQ_dbeta = (Q_plus_beta - Q_minus_beta) / two_eps;

    T Q_plus_N = hahn_polynomial_q(n, x, alpha, beta, N + eps);
    T Q_minus_N = hahn_polynomial_q(n, x, alpha, beta, N - eps);
    T dQ_dN = (Q_plus_N - Q_minus_N) / two_eps;

    // gradient_gradient_output = sum of gg_* * dQ/d*
    T gradient_gradient_output = gradient_gradient_n * dQ_dn +
                                  gradient_gradient_x * dQ_dx +
                                  gradient_gradient_alpha * dQ_dalpha +
                                  gradient_gradient_beta * dQ_dbeta +
                                  gradient_gradient_N * dQ_dN;

    // Second derivatives via finite differences for new gradients
    T Q_center = hahn_polynomial_q(n, x, alpha, beta, N);

    // d^2Q/dn^2
    T d2Q_dn2 = (Q_plus_n - T(2) * Q_center + Q_minus_n) / (eps * eps);

    // d^2Q/dx^2
    T d2Q_dx2 = (Q_plus_x - T(2) * Q_center + Q_minus_x) / (eps * eps);

    // d^2Q/dalpha^2
    T d2Q_dalpha2 = (Q_plus_alpha - T(2) * Q_center + Q_minus_alpha) / (eps * eps);

    // d^2Q/dbeta^2
    T d2Q_dbeta2 = (Q_plus_beta - T(2) * Q_center + Q_minus_beta) / (eps * eps);

    // d^2Q/dN^2
    T d2Q_dN2 = (Q_plus_N - T(2) * Q_center + Q_minus_N) / (eps * eps);

    // Compute new gradients (simplified: only diagonal second derivatives)
    T new_gradient_n = gradient_gradient_n * gradient * d2Q_dn2;
    T new_gradient_x = gradient_gradient_x * gradient * d2Q_dx2;
    T new_gradient_alpha = gradient_gradient_alpha * gradient * d2Q_dalpha2;
    T new_gradient_beta = gradient_gradient_beta * gradient * d2Q_dbeta2;
    T new_gradient_N = gradient_gradient_N * gradient * d2Q_dN2;

    return {gradient_gradient_output, new_gradient_n, new_gradient_x,
            new_gradient_alpha, new_gradient_beta, new_gradient_N};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>,
           c10::complex<T>, c10::complex<T>, c10::complex<T>>
hahn_polynomial_q_backward_backward(
    c10::complex<T> gradient_gradient_n,
    c10::complex<T> gradient_gradient_x,
    c10::complex<T> gradient_gradient_alpha,
    c10::complex<T> gradient_gradient_beta,
    c10::complex<T> gradient_gradient_N,
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> x,
    c10::complex<T> alpha,
    c10::complex<T> beta,
    c10::complex<T> N
) {
    c10::complex<T> eps(T(1e-7), T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> two_eps = two * eps;

    // First derivatives dQ/d{n,x,alpha,beta,N} for gradient_gradient_output
    c10::complex<T> Q_plus_n = hahn_polynomial_q(n + eps, x, alpha, beta, N);
    c10::complex<T> Q_minus_n = hahn_polynomial_q(n - eps, x, alpha, beta, N);
    c10::complex<T> dQ_dn = (Q_plus_n - Q_minus_n) / two_eps;

    c10::complex<T> Q_plus_x = hahn_polynomial_q(n, x + eps, alpha, beta, N);
    c10::complex<T> Q_minus_x = hahn_polynomial_q(n, x - eps, alpha, beta, N);
    c10::complex<T> dQ_dx = (Q_plus_x - Q_minus_x) / two_eps;

    c10::complex<T> Q_plus_alpha = hahn_polynomial_q(n, x, alpha + eps, beta, N);
    c10::complex<T> Q_minus_alpha = hahn_polynomial_q(n, x, alpha - eps, beta, N);
    c10::complex<T> dQ_dalpha = (Q_plus_alpha - Q_minus_alpha) / two_eps;

    c10::complex<T> Q_plus_beta = hahn_polynomial_q(n, x, alpha, beta + eps, N);
    c10::complex<T> Q_minus_beta = hahn_polynomial_q(n, x, alpha, beta - eps, N);
    c10::complex<T> dQ_dbeta = (Q_plus_beta - Q_minus_beta) / two_eps;

    c10::complex<T> Q_plus_N = hahn_polynomial_q(n, x, alpha, beta, N + eps);
    c10::complex<T> Q_minus_N = hahn_polynomial_q(n, x, alpha, beta, N - eps);
    c10::complex<T> dQ_dN = (Q_plus_N - Q_minus_N) / two_eps;

    // gradient_gradient_output = sum of gg_* * dQ/d*
    c10::complex<T> gradient_gradient_output = gradient_gradient_n * dQ_dn +
                                                gradient_gradient_x * dQ_dx +
                                                gradient_gradient_alpha * dQ_dalpha +
                                                gradient_gradient_beta * dQ_dbeta +
                                                gradient_gradient_N * dQ_dN;

    // Second derivatives via finite differences
    c10::complex<T> Q_center = hahn_polynomial_q(n, x, alpha, beta, N);
    c10::complex<T> d2Q_dn2 = (Q_plus_n - two * Q_center + Q_minus_n) / (eps * eps);
    c10::complex<T> d2Q_dx2 = (Q_plus_x - two * Q_center + Q_minus_x) / (eps * eps);
    c10::complex<T> d2Q_dalpha2 = (Q_plus_alpha - two * Q_center + Q_minus_alpha) / (eps * eps);
    c10::complex<T> d2Q_dbeta2 = (Q_plus_beta - two * Q_center + Q_minus_beta) / (eps * eps);
    c10::complex<T> d2Q_dN2 = (Q_plus_N - two * Q_center + Q_minus_N) / (eps * eps);

    // Compute new gradients
    c10::complex<T> new_gradient_n = gradient_gradient_n * gradient * d2Q_dn2;
    c10::complex<T> new_gradient_x = gradient_gradient_x * gradient * d2Q_dx2;
    c10::complex<T> new_gradient_alpha = gradient_gradient_alpha * gradient * d2Q_dalpha2;
    c10::complex<T> new_gradient_beta = gradient_gradient_beta * gradient * d2Q_dbeta2;
    c10::complex<T> new_gradient_N = gradient_gradient_N * gradient * d2Q_dN2;

    return {gradient_gradient_output, new_gradient_n, new_gradient_x,
            new_gradient_alpha, new_gradient_beta, new_gradient_N};
}

} // namespace torchscience::kernel::special_functions
