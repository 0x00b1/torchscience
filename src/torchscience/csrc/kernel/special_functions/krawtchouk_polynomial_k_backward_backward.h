#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "krawtchouk_polynomial_k.h"

namespace torchscience::kernel::special_functions {

// Second-order backward for Krawtchouk polynomial K_n(x; p, N)
//
// All derivatives computed via finite differences.
template <typename T>
std::tuple<T, T, T, T, T> krawtchouk_polynomial_k_backward_backward(
    T gradient_gradient_n,
    T gradient_gradient_x,
    T gradient_gradient_p,
    T gradient_gradient_N,
    T gradient,
    T n,
    T x,
    T p,
    T N
) {
    T eps = T(1e-7);
    T two_eps = T(2) * eps;

    // First derivatives dK/d{n,x,p,N} for gradient_gradient_output
    T K_plus_n = krawtchouk_polynomial_k(n + eps, x, p, N);
    T K_minus_n = krawtchouk_polynomial_k(n - eps, x, p, N);
    T dK_dn = (K_plus_n - K_minus_n) / two_eps;

    T K_plus_x = krawtchouk_polynomial_k(n, x + eps, p, N);
    T K_minus_x = krawtchouk_polynomial_k(n, x - eps, p, N);
    T dK_dx = (K_plus_x - K_minus_x) / two_eps;

    T K_plus_p = krawtchouk_polynomial_k(n, x, p + eps, N);
    T K_minus_p = krawtchouk_polynomial_k(n, x, p - eps, N);
    T dK_dp = (K_plus_p - K_minus_p) / two_eps;

    T K_plus_N = krawtchouk_polynomial_k(n, x, p, N + eps);
    T K_minus_N = krawtchouk_polynomial_k(n, x, p, N - eps);
    T dK_dN = (K_plus_N - K_minus_N) / two_eps;

    // gradient_gradient_output = sum of gg_* * dK/d*
    T gradient_gradient_output = gradient_gradient_n * dK_dn +
                                  gradient_gradient_x * dK_dx +
                                  gradient_gradient_p * dK_dp +
                                  gradient_gradient_N * dK_dN;

    // Second derivatives via finite differences for new gradients
    // d^2K/dx^2
    T K_center = krawtchouk_polynomial_k(n, x, p, N);
    T d2K_dx2 = (K_plus_x - T(2) * K_center + K_minus_x) / (eps * eps);

    // d^2K/dp^2
    T d2K_dp2 = (K_plus_p - T(2) * K_center + K_minus_p) / (eps * eps);

    // d^2K/dN^2
    T d2K_dN2 = (K_plus_N - T(2) * K_center + K_minus_N) / (eps * eps);

    // d^2K/dn^2
    T d2K_dn2 = (K_plus_n - T(2) * K_center + K_minus_n) / (eps * eps);

    // Compute new gradients (simplified: only diagonal second derivatives)
    T new_gradient_n = gradient_gradient_n * gradient * d2K_dn2;
    T new_gradient_x = gradient_gradient_x * gradient * d2K_dx2;
    T new_gradient_p = gradient_gradient_p * gradient * d2K_dp2;
    T new_gradient_N = gradient_gradient_N * gradient * d2K_dN2;

    return {gradient_gradient_output, new_gradient_n, new_gradient_x, new_gradient_p, new_gradient_N};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
krawtchouk_polynomial_k_backward_backward(
    c10::complex<T> gradient_gradient_n,
    c10::complex<T> gradient_gradient_x,
    c10::complex<T> gradient_gradient_p,
    c10::complex<T> gradient_gradient_N,
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> x,
    c10::complex<T> p,
    c10::complex<T> N
) {
    c10::complex<T> eps(T(1e-7), T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> two_eps = two * eps;

    // First derivatives dK/d{n,x,p,N} for gradient_gradient_output
    c10::complex<T> K_plus_n = krawtchouk_polynomial_k(n + eps, x, p, N);
    c10::complex<T> K_minus_n = krawtchouk_polynomial_k(n - eps, x, p, N);
    c10::complex<T> dK_dn = (K_plus_n - K_minus_n) / two_eps;

    c10::complex<T> K_plus_x = krawtchouk_polynomial_k(n, x + eps, p, N);
    c10::complex<T> K_minus_x = krawtchouk_polynomial_k(n, x - eps, p, N);
    c10::complex<T> dK_dx = (K_plus_x - K_minus_x) / two_eps;

    c10::complex<T> K_plus_p = krawtchouk_polynomial_k(n, x, p + eps, N);
    c10::complex<T> K_minus_p = krawtchouk_polynomial_k(n, x, p - eps, N);
    c10::complex<T> dK_dp = (K_plus_p - K_minus_p) / two_eps;

    c10::complex<T> K_plus_N = krawtchouk_polynomial_k(n, x, p, N + eps);
    c10::complex<T> K_minus_N = krawtchouk_polynomial_k(n, x, p, N - eps);
    c10::complex<T> dK_dN = (K_plus_N - K_minus_N) / two_eps;

    // gradient_gradient_output = sum of gg_* * dK/d*
    c10::complex<T> gradient_gradient_output = gradient_gradient_n * dK_dn +
                                                gradient_gradient_x * dK_dx +
                                                gradient_gradient_p * dK_dp +
                                                gradient_gradient_N * dK_dN;

    // Second derivatives via finite differences
    c10::complex<T> K_center = krawtchouk_polynomial_k(n, x, p, N);
    c10::complex<T> d2K_dx2 = (K_plus_x - two * K_center + K_minus_x) / (eps * eps);
    c10::complex<T> d2K_dp2 = (K_plus_p - two * K_center + K_minus_p) / (eps * eps);
    c10::complex<T> d2K_dN2 = (K_plus_N - two * K_center + K_minus_N) / (eps * eps);
    c10::complex<T> d2K_dn2 = (K_plus_n - two * K_center + K_minus_n) / (eps * eps);

    // Compute new gradients
    c10::complex<T> new_gradient_n = gradient_gradient_n * gradient * d2K_dn2;
    c10::complex<T> new_gradient_x = gradient_gradient_x * gradient * d2K_dx2;
    c10::complex<T> new_gradient_p = gradient_gradient_p * gradient * d2K_dp2;
    c10::complex<T> new_gradient_N = gradient_gradient_N * gradient * d2K_dN2;

    return {gradient_gradient_output, new_gradient_n, new_gradient_x, new_gradient_p, new_gradient_N};
}

} // namespace torchscience::kernel::special_functions
