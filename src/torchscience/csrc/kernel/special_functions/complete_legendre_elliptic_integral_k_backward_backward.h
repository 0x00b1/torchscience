#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "carlson_elliptic_integral_r_f.h"
#include "carlson_elliptic_integral_r_g.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute dK/dm = [E(m) - (1-m)K(m)] / [2m(1-m)]
template <typename T>
T compute_dK_dm(T m) {
    const T eps = std::numeric_limits<T>::epsilon();

    // Handle the m = 0 case with the known limit
    if (std::abs(m) < eps) {
        return static_cast<T>(M_PI) / T(4);
    }

    // Handle m = 1 case (singularity)
    if (std::abs(m - T(1)) < eps) {
        return std::numeric_limits<T>::infinity();
    }

    T one_minus_m = T(1) - m;

    // K(m) = R_F(0, 1-m, 1)
    T K = carlson_elliptic_integral_r_f(T(0), one_minus_m, T(1));

    // E(m) = 2 * R_G(0, 1-m, 1)
    T E = T(2) * carlson_elliptic_integral_r_g(T(0), one_minus_m, T(1));

    // dK/dm = [E(m) - (1-m)K(m)] / [2m(1-m)]
    T numerator = E - one_minus_m * K;
    T denominator = T(2) * m * one_minus_m;

    return numerator / denominator;
}

// Compute second derivative of K (d2K/dm2) using numerical differentiation
// This is the first derivative of dK/dm
template <typename T>
T compute_d2K_dm2(T m) {
    // Use central difference with appropriate step size
    // Step size ~ eps^(1/2) for optimal first derivative accuracy
    const T h = std::sqrt(std::numeric_limits<T>::epsilon());

    // For m near 0 or 1, use one-sided differences
    const T boundary_eps = h * T(10);

    if (m < boundary_eps) {
        // Near m = 0, use forward difference
        T f0 = compute_dK_dm(m);
        T f1 = compute_dK_dm(m + h);
        return (f1 - f0) / h;
    } else if (m > T(1) - boundary_eps) {
        // Near m = 1, use backward difference
        T f0 = compute_dK_dm(m - h);
        T f1 = compute_dK_dm(m);
        return (f1 - f0) / h;
    } else {
        // Central difference: d(dK/dm)/dm = (dK/dm(m+h) - dK/dm(m-h)) / (2h)
        T f_plus = compute_dK_dm(m + h);
        T f_minus = compute_dK_dm(m - h);
        return (f_plus - f_minus) / (T(2) * h);
    }
}

template <typename T>
c10::complex<T> compute_dK_dm(c10::complex<T> m) {
    const T eps = std::numeric_limits<T>::epsilon();
    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> four(T(4), T(0));
    c10::complex<T> pi_val(static_cast<T>(M_PI), T(0));

    // Handle the m ~ 0 case
    if (std::abs(m) < eps) {
        return pi_val / four;
    }

    // Handle m ~ 1 case
    if (std::abs(m - one) < eps) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    c10::complex<T> one_minus_m = one - m;

    // K(m) = R_F(0, 1-m, 1)
    c10::complex<T> K = carlson_elliptic_integral_r_f(zero, one_minus_m, one);

    // E(m) = 2 * R_G(0, 1-m, 1)
    c10::complex<T> E = two * carlson_elliptic_integral_r_g(zero, one_minus_m, one);

    // dK/dm = [E(m) - (1-m)K(m)] / [2m(1-m)]
    c10::complex<T> numerator = E - one_minus_m * K;
    c10::complex<T> denominator = two * m * one_minus_m;

    return numerator / denominator;
}

template <typename T>
c10::complex<T> compute_d2K_dm2(c10::complex<T> m) {
    // Use central difference in the complex plane
    // Step size ~ eps^(1/2) for optimal first derivative accuracy
    const T h = std::sqrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> ch(h, T(0));

    // Central difference: d(dK/dm)/dm = (dK/dm(m+h) - dK/dm(m-h)) / (2h)
    c10::complex<T> f_plus = compute_dK_dm(m + ch);
    c10::complex<T> f_minus = compute_dK_dm(m - ch);

    return (f_plus - f_minus) / (c10::complex<T>(T(2), T(0)) * ch);
}

} // namespace detail

// Real backward_backward
// Returns gradients for (grad_output, m)
template <typename T>
std::tuple<T, T> complete_legendre_elliptic_integral_k_backward_backward(
    T gg_m,
    T grad_output,
    T m
) {
    // dK/dm
    T dK_dm = detail::compute_dK_dm(m);

    // Gradient w.r.t. grad_output: gg_m * dK/dm
    T grad_grad_output = gg_m * dK_dm;

    // d²K/dm²
    T d2K_dm2 = detail::compute_d2K_dm2(m);

    // Gradient w.r.t. m: gg_m * grad_output * d²K/dm²
    T grad_m = gg_m * grad_output * d2K_dm2;

    return {grad_grad_output, grad_m};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> complete_legendre_elliptic_integral_k_backward_backward(
    c10::complex<T> gg_m,
    c10::complex<T> grad_output,
    c10::complex<T> m
) {
    // dK/dm
    c10::complex<T> dK_dm = detail::compute_dK_dm(m);

    // Gradient w.r.t. grad_output with conjugation for Wirtinger derivatives
    c10::complex<T> grad_grad_output = gg_m * std::conj(dK_dm);

    // d²K/dm²
    c10::complex<T> d2K_dm2 = detail::compute_d2K_dm2(m);

    // Gradient w.r.t. m with conjugation
    c10::complex<T> grad_m = gg_m * grad_output * std::conj(d2K_dm2);

    return {grad_grad_output, grad_m};
}

} // namespace torchscience::kernel::special_functions
