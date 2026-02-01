#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "weierstrass_detail.h"
#include "theta_1.h"
#include "theta_2.h"
#include "theta_3.h"

namespace torchscience::kernel::special_functions {

// Weierstrass elliptic function P(z; g2, g3)
//
// Mathematical definition:
// The Weierstrass P-function satisfies the differential equation:
// P'(z)^2 = 4*P(z)^3 - g2*P(z) - g3
//
// where g2 and g3 are the Weierstrass invariants.
//
// Domain:
// - z: complex (the argument)
// - g2, g3: complex (the invariants)
//
// Special values:
// - P(0; g2, g3) = infinity (double pole at the origin)
// - P(omega1; g2, g3) = e1 (half-period value)
// - P(omega2; g2, g3) = e2
// - P(omega3; g2, g3) = e3
//
// Laurent expansion around z = 0:
// P(z) = 1/z^2 + (g2/20)*z^2 + (g3/28)*z^4 + O(z^6)
//
// Algorithm:
// Uses the theta function representation:
// P(z) = (pi/(2*omega1))^2 * (theta2^4(0,q) + theta3^4(0,q))/3
//      - (pi/(2*omega1))^2 * theta1''(v,q) / theta1(v,q)
// where v = pi*z/(2*omega1) and q is the nome.
//
// For numerical stability, we use finite differences to compute theta1''.

namespace detail {

template <typename T>
inline T weierstrass_p_tolerance() {
    return T(1e-10);
}

template <>
inline float weierstrass_p_tolerance<float>() { return 1e-5f; }

template <>
inline double weierstrass_p_tolerance<double>() { return 1e-14; }

template <>
inline c10::Half weierstrass_p_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 weierstrass_p_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

// Compute theta_1''(v, q) using finite differences
// theta_1''(v) = (theta_1(v+h) - 2*theta_1(v) + theta_1(v-h)) / h^2
template <typename T>
c10::complex<T> theta_1_second_derivative(c10::complex<T> v, c10::complex<T> q) {
    // Choose step size based on precision
    const T h = T(1e-5);
    const c10::complex<T> h_c(h, T(0));

    c10::complex<T> theta_plus = theta_1(v + h_c, q);
    c10::complex<T> theta_center = theta_1(v, q);
    c10::complex<T> theta_minus = theta_1(v - h_c, q);

    return (theta_plus - c10::complex<T>(T(2), T(0)) * theta_center + theta_minus) /
           c10::complex<T>(h * h, T(0));
}

template <typename T>
T theta_1_second_derivative(T v, T q) {
    const T h = T(1e-5);

    T theta_plus = theta_1(v + h, q);
    T theta_center = theta_1(v, q);
    T theta_minus = theta_1(v - h, q);

    return (theta_plus - T(2) * theta_center + theta_minus) / (h * h);
}

} // namespace detail

template <typename T>
T weierstrass_p(T z, T g2, T g3) {
    const T tol = detail::weierstrass_p_tolerance<T>();
    const T pi = T(3.14159265358979323846);

    // Handle z = 0: double pole, return infinity
    if (std::abs(z) < tol) {
        return std::numeric_limits<T>::infinity();
    }

    // For small z, use the Laurent series expansion for better accuracy
    // P(z) = 1/z^2 + (g2/20)*z^2 + (g3/28)*z^4 + O(z^6)
    if (std::abs(z) < T(0.1)) {
        T z2 = z * z;
        T z4 = z2 * z2;
        return T(1) / z2 + (g2 / T(20)) * z2 + (g3 / T(28)) * z4;
    }

    // Convert invariants to lattice parameters
    auto params = weierstrass_detail::invariants_to_lattice_params(g2, g3);

    // Extract omega1 and nome q
    c10::complex<T> omega1 = params.omega1;
    c10::complex<T> q = params.q;

    // Compute v = pi * z / (2 * omega1)
    c10::complex<T> z_c(z, T(0));
    c10::complex<T> pi_c(pi, T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> v = pi_c * z_c / (two * omega1);

    // Compute the constant term: (pi/(2*omega1))^2 * (theta2^4(0) + theta3^4(0))/3
    c10::complex<T> pi_over_2omega1 = pi_c / (two * omega1);
    c10::complex<T> pi_over_2omega1_sq = pi_over_2omega1 * pi_over_2omega1;

    c10::complex<T> zero_c(T(0), T(0));
    c10::complex<T> theta2_0 = theta_2(zero_c, q);
    c10::complex<T> theta3_0 = theta_3(zero_c, q);

    c10::complex<T> theta2_4 = theta2_0 * theta2_0 * theta2_0 * theta2_0;
    c10::complex<T> theta3_4 = theta3_0 * theta3_0 * theta3_0 * theta3_0;

    c10::complex<T> three(T(3), T(0));
    c10::complex<T> constant_term = pi_over_2omega1_sq * (theta2_4 + theta3_4) / three;

    // Compute the theta_1 term: -(pi/(2*omega1))^2 * theta1''(v) / theta1(v)
    c10::complex<T> theta1_v = theta_1(v, q);

    // Check if we're at a lattice point (theta1 = 0)
    if (std::abs(theta1_v) < tol) {
        return std::numeric_limits<T>::infinity();
    }

    c10::complex<T> theta1_pp_v = detail::theta_1_second_derivative(v, q);
    c10::complex<T> theta_term = pi_over_2omega1_sq * theta1_pp_v / theta1_v;

    // P(z) = constant_term - theta_term
    c10::complex<T> result = constant_term - theta_term;

    // For real inputs, return the real part
    return result.real();
}

template <typename T>
c10::complex<T> weierstrass_p(c10::complex<T> z, c10::complex<T> g2, c10::complex<T> g3) {
    const T tol = detail::weierstrass_p_tolerance<T>();
    const T pi = T(3.14159265358979323846);

    // Handle z = 0: double pole, return infinity
    if (std::abs(z) < tol) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    // For small z, use the Laurent series expansion for better accuracy
    // P(z) = 1/z^2 + (g2/20)*z^2 + (g3/28)*z^4 + O(z^6)
    if (std::abs(z) < T(0.1)) {
        c10::complex<T> z2 = z * z;
        c10::complex<T> z4 = z2 * z2;
        c10::complex<T> one(T(1), T(0));
        c10::complex<T> twenty(T(20), T(0));
        c10::complex<T> twentyeight(T(28), T(0));
        return one / z2 + (g2 / twenty) * z2 + (g3 / twentyeight) * z4;
    }

    // Convert invariants to lattice parameters
    auto params = weierstrass_detail::invariants_to_lattice_params(g2, g3);

    // Extract omega1 and nome q
    c10::complex<T> omega1 = params.omega1;
    c10::complex<T> q = params.q;

    // Compute v = pi * z / (2 * omega1)
    c10::complex<T> pi_c(pi, T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> v = pi_c * z / (two * omega1);

    // Compute the constant term: (pi/(2*omega1))^2 * (theta2^4(0) + theta3^4(0))/3
    c10::complex<T> pi_over_2omega1 = pi_c / (two * omega1);
    c10::complex<T> pi_over_2omega1_sq = pi_over_2omega1 * pi_over_2omega1;

    c10::complex<T> zero_c(T(0), T(0));
    c10::complex<T> theta2_0 = theta_2(zero_c, q);
    c10::complex<T> theta3_0 = theta_3(zero_c, q);

    c10::complex<T> theta2_4 = theta2_0 * theta2_0 * theta2_0 * theta2_0;
    c10::complex<T> theta3_4 = theta3_0 * theta3_0 * theta3_0 * theta3_0;

    c10::complex<T> three(T(3), T(0));
    c10::complex<T> constant_term = pi_over_2omega1_sq * (theta2_4 + theta3_4) / three;

    // Compute the theta_1 term: -(pi/(2*omega1))^2 * theta1''(v) / theta1(v)
    c10::complex<T> theta1_v = theta_1(v, q);

    // Check if we're at a lattice point (theta1 = 0)
    if (std::abs(theta1_v) < tol) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    c10::complex<T> theta1_pp_v = detail::theta_1_second_derivative(v, q);
    c10::complex<T> theta_term = pi_over_2omega1_sq * theta1_pp_v / theta1_v;

    // P(z) = constant_term - theta_term
    return constant_term - theta_term;
}

} // namespace torchscience::kernel::special_functions
