#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "weierstrass_detail.h"
#include "theta_1.h"

namespace torchscience::kernel::special_functions {

// Weierstrass eta quasi-period eta1 = zeta(omega1)
//
// Mathematical definition:
// The Weierstrass eta function (quasi-period) is defined as:
//   eta1 = zeta(omega1)
// where omega1 is the first half-period and zeta is the Weierstrass zeta function.
//
// The quasi-periods satisfy the quasi-periodicity relation:
//   zeta(z + 2*omega_i) = zeta(z) + 2*eta_i
//
// They also satisfy the Legendre relation:
//   eta1*omega3 - eta3*omega1 = pi*i/2
//
// Formula via theta functions:
//   eta1 = -pi^2 * theta1'''(0,q) / (12 * omega1 * theta1'(0,q))
//
// where theta1'(0,q) is the first derivative of theta1 at v=0.
//
// Domain:
// - g2, g3: complex (the Weierstrass invariants)
//
// Note: This returns eta1, the quasi-period associated with omega1.

namespace detail {

template <typename T>
inline T weierstrass_eta_tolerance() {
    return T(1e-10);
}

template <>
inline float weierstrass_eta_tolerance<float>() { return 1e-5f; }

template <>
inline double weierstrass_eta_tolerance<double>() { return 1e-14; }

template <>
inline c10::Half weierstrass_eta_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 weierstrass_eta_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
inline T weierstrass_eta_finite_diff_step() {
    return std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
}

template <>
inline float weierstrass_eta_finite_diff_step<float>() {
    return 1e-4f;
}

template <>
inline double weierstrass_eta_finite_diff_step<double>() {
    return 1e-7;
}

// Compute theta_1'(0, q) using finite differences
// theta_1'(v) = (theta_1(v+h) - theta_1(v-h)) / (2h)
template <typename T>
c10::complex<T> theta_1_first_derivative_at_zero(c10::complex<T> q) {
    const T h = weierstrass_eta_finite_diff_step<T>();
    const c10::complex<T> h_c(h, T(0));

    c10::complex<T> theta_plus = theta_1(h_c, q);
    c10::complex<T> theta_minus = theta_1(-h_c, q);

    return (theta_plus - theta_minus) / (T(2) * h_c);
}

// Compute theta_1'''(0, q) using finite differences
// theta_1'''(v) = (theta_1(v+2h) - 2*theta_1(v+h) + 2*theta_1(v-h) - theta_1(v-2h)) / (2h^3)
template <typename T>
c10::complex<T> theta_1_third_derivative_at_zero(c10::complex<T> q) {
    const T h = weierstrass_eta_finite_diff_step<T>();
    const c10::complex<T> h_c(h, T(0));
    const c10::complex<T> two_h_c(T(2) * h, T(0));
    const c10::complex<T> two(T(2), T(0));

    c10::complex<T> theta_2h = theta_1(two_h_c, q);
    c10::complex<T> theta_h = theta_1(h_c, q);
    c10::complex<T> theta_mh = theta_1(-h_c, q);
    c10::complex<T> theta_m2h = theta_1(-two_h_c, q);

    // Central finite difference for third derivative
    // f'''(0) = (f(2h) - 2*f(h) + 2*f(-h) - f(-2h)) / (2h^3)
    c10::complex<T> numer = theta_2h - two * theta_h + two * theta_mh - theta_m2h;
    c10::complex<T> denom(T(2) * h * h * h, T(0));

    return numer / denom;
}

} // namespace detail

template <typename T>
T weierstrass_eta(T g2, T g3) {
    const T pi = T(3.14159265358979323846);

    // Convert invariants to lattice parameters
    auto params = weierstrass_detail::invariants_to_lattice_params(g2, g3);

    c10::complex<T> omega1 = params.omega1;
    c10::complex<T> q = params.q;

    // Compute theta1'(0, q) and theta1'''(0, q)
    c10::complex<T> theta1_prime = detail::theta_1_first_derivative_at_zero<T>(q);
    c10::complex<T> theta1_triple_prime = detail::theta_1_third_derivative_at_zero<T>(q);

    // eta1 = -pi^2 * theta1'''(0,q) / (12 * omega1 * theta1'(0,q))
    c10::complex<T> pi_c(pi, T(0));
    c10::complex<T> twelve(T(12), T(0));

    c10::complex<T> eta1 = -pi_c * pi_c * theta1_triple_prime / (twelve * omega1 * theta1_prime);

    // For real inputs, return the real part
    return eta1.real();
}

template <typename T>
c10::complex<T> weierstrass_eta(c10::complex<T> g2, c10::complex<T> g3) {
    const T pi = T(3.14159265358979323846);

    // Convert invariants to lattice parameters
    auto params = weierstrass_detail::invariants_to_lattice_params(g2, g3);

    c10::complex<T> omega1 = params.omega1;
    c10::complex<T> q = params.q;

    // Compute theta1'(0, q) and theta1'''(0, q)
    c10::complex<T> theta1_prime = detail::theta_1_first_derivative_at_zero<T>(q);
    c10::complex<T> theta1_triple_prime = detail::theta_1_third_derivative_at_zero<T>(q);

    // eta1 = -pi^2 * theta1'''(0,q) / (12 * omega1 * theta1'(0,q))
    c10::complex<T> pi_c(pi, T(0));
    c10::complex<T> twelve(T(12), T(0));

    c10::complex<T> eta1 = -pi_c * pi_c * theta1_triple_prime / (twelve * omega1 * theta1_prime);

    return eta1;
}

} // namespace torchscience::kernel::special_functions
