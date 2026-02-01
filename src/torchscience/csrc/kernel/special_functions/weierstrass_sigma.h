#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "weierstrass_detail.h"
#include "theta_1.h"

namespace torchscience::kernel::special_functions {

// Weierstrass sigma function σ(z; g2, g3)
//
// Mathematical definition:
// The Weierstrass sigma function is an entire (no poles) odd function that
// satisfies:
// - σ'(z)/σ(z) = ζ(z) (Weierstrass zeta function)
// - σ(0) = 0
//
// Taylor expansion around z = 0:
// σ(z) = z - (g2/240)*z^5 - (g3/840)*z^7 + O(z^9)
//
// Theta function representation:
// σ(z) = (2ω₁/π) * exp(η₁z²/(2ω₁)) * θ₁(πz/(2ω₁), q) / θ₁'(0, q)
//
// where:
// - ω₁ is the first half-period
// - η₁ = ζ(ω₁) is the quasi-period
// - q is the nome
// - θ₁ is the Jacobi theta function
//
// Domain:
// - z: complex (the argument)
// - g2, g3: complex (the Weierstrass invariants)
//
// Special values:
// - σ(0; g2, g3) = 0
// - σ(-z) = -σ(z) (odd function)
//
// Algorithm:
// 1. Handle z = 0: return 0
// 2. For small |z|, use Taylor series for accuracy
// 3. Otherwise, use the theta function formula

namespace detail {

template <typename T>
inline T weierstrass_sigma_tolerance() {
    return T(1e-10);
}

template <>
inline float weierstrass_sigma_tolerance<float>() { return 1e-5f; }

template <>
inline double weierstrass_sigma_tolerance<double>() { return 1e-14; }

template <>
inline c10::Half weierstrass_sigma_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 weierstrass_sigma_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

// Compute θ₁'(0, q) using finite difference
// θ₁'(v, q) = (θ₁(v+h, q) - θ₁(v-h, q)) / (2h)
// At v = 0: θ₁'(0, q)
template <typename T>
c10::complex<T> theta_1_derivative_at_zero(c10::complex<T> q) {
    const T h = T(1e-6);
    const c10::complex<T> h_c(h, T(0));
    const c10::complex<T> zero_c(T(0), T(0));

    c10::complex<T> theta_plus = theta_1(h_c, q);
    c10::complex<T> theta_minus = theta_1(-h_c, q);

    return (theta_plus - theta_minus) / c10::complex<T>(T(2) * h, T(0));
}

template <typename T>
T theta_1_derivative_at_zero(T q) {
    const T h = T(1e-6);

    T theta_plus = theta_1(h, q);
    T theta_minus = theta_1(-h, q);

    return (theta_plus - theta_minus) / (T(2) * h);
}

// Compute η₁ = ζ(ω₁), the quasi-period
// For the Legendre relation: η₁ω₃ - η₃ω₁ = πi/2
// We use the formula: η₁ = π²/(12ω₁) * (θ₂⁴(0,q) + 2θ₃⁴(0,q))/(θ₂²(0,q)θ₃²(0,q))
// But for simplicity, we use finite differences to compute ζ(ω₁)
// from the relation ζ(z) = -∫ P(z) dz + constant
// Actually, use: η₁ = -(π²/12ω₁) * (1 - 24 * sum_{n=1}^∞ n*q^{2n}/(1-q^{2n}))
// For simplicity, we'll use a simpler approximation based on the q-expansion
template <typename T>
c10::complex<T> compute_eta1(c10::complex<T> omega1, c10::complex<T> q) {
    const T pi = T(3.14159265358979323846);
    const T tol = weierstrass_sigma_tolerance<T>();
    constexpr int max_terms = 100;

    // Compute the Eisenstein series sum: E2 = 1 - 24 * sum_{n>=1} n*q^{2n}/(1-q^{2n})
    c10::complex<T> q2 = q * q;
    c10::complex<T> sum(T(0), T(0));
    c10::complex<T> q2n = q2;

    for (int n = 1; n < max_terms; ++n) {
        c10::complex<T> one(T(1), T(0));
        c10::complex<T> term = c10::complex<T>(T(n), T(0)) * q2n / (one - q2n);
        sum += term;

        if (std::abs(term) < tol) break;

        q2n = q2n * q2;
    }

    c10::complex<T> E2 = c10::complex<T>(T(1), T(0)) - c10::complex<T>(T(24), T(0)) * sum;

    // η₁ = (π²/(12ω₁)) * E2
    c10::complex<T> pi_sq(pi * pi, T(0));
    c10::complex<T> twelve(T(12), T(0));

    return (pi_sq / (twelve * omega1)) * E2;
}

template <typename T>
T compute_eta1_real(c10::complex<T> omega1, c10::complex<T> q) {
    return compute_eta1(omega1, q).real();
}

} // namespace detail

template <typename T>
T weierstrass_sigma(T z, T g2, T g3) {
    const T tol = detail::weierstrass_sigma_tolerance<T>();
    const T pi = T(3.14159265358979323846);

    // Handle z = 0: σ(0) = 0
    if (std::abs(z) < tol) {
        return T(0);
    }

    // For small z, use the Taylor series expansion for better accuracy
    // σ(z) = z - (g2/240)*z^5 - (g3/840)*z^7 + O(z^9)
    if (std::abs(z) < T(0.1)) {
        T z2 = z * z;
        T z4 = z2 * z2;
        T z5 = z4 * z;
        T z7 = z5 * z2;
        return z - (g2 / T(240)) * z5 - (g3 / T(840)) * z7;
    }

    // Convert invariants to lattice parameters
    auto params = weierstrass_detail::invariants_to_lattice_params(g2, g3);

    // Extract omega1 and nome q
    c10::complex<T> omega1 = params.omega1;
    c10::complex<T> q = params.q;

    // Compute v = π * z / (2 * ω₁)
    c10::complex<T> z_c(z, T(0));
    c10::complex<T> pi_c(pi, T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> v = pi_c * z_c / (two * omega1);

    // Compute θ₁(v, q)
    c10::complex<T> theta1_v = theta_1(v, q);

    // Compute θ₁'(0, q)
    c10::complex<T> theta1_prime_0 = detail::theta_1_derivative_at_zero(q);

    // Check for degenerate case
    if (std::abs(theta1_prime_0) < tol) {
        // Fallback to Taylor series
        T z2 = z * z;
        T z4 = z2 * z2;
        T z5 = z4 * z;
        T z7 = z5 * z2;
        return z - (g2 / T(240)) * z5 - (g3 / T(840)) * z7;
    }

    // Compute η₁ (quasi-period)
    c10::complex<T> eta1 = detail::compute_eta1(omega1, q);

    // Compute the exponential factor: exp(η₁z²/(2ω₁))
    c10::complex<T> exp_arg = eta1 * z_c * z_c / (two * omega1);
    c10::complex<T> exp_factor = std::exp(exp_arg);

    // Compute σ(z) = (2ω₁/π) * exp(η₁z²/(2ω₁)) * θ₁(v,q) / θ₁'(0,q)
    c10::complex<T> prefactor = two * omega1 / pi_c;
    c10::complex<T> result = prefactor * exp_factor * theta1_v / theta1_prime_0;

    // For real inputs, return the real part
    return result.real();
}

template <typename T>
c10::complex<T> weierstrass_sigma(c10::complex<T> z, c10::complex<T> g2, c10::complex<T> g3) {
    const T tol = detail::weierstrass_sigma_tolerance<T>();
    const T pi = T(3.14159265358979323846);

    // Handle z = 0: σ(0) = 0
    if (std::abs(z) < tol) {
        return c10::complex<T>(T(0), T(0));
    }

    // For small z, use the Taylor series expansion for better accuracy
    // σ(z) = z - (g2/240)*z^5 - (g3/840)*z^7 + O(z^9)
    if (std::abs(z) < T(0.1)) {
        c10::complex<T> z2 = z * z;
        c10::complex<T> z4 = z2 * z2;
        c10::complex<T> z5 = z4 * z;
        c10::complex<T> z7 = z5 * z2;
        c10::complex<T> c240(T(240), T(0));
        c10::complex<T> c840(T(840), T(0));
        return z - (g2 / c240) * z5 - (g3 / c840) * z7;
    }

    // Convert invariants to lattice parameters
    auto params = weierstrass_detail::invariants_to_lattice_params(g2, g3);

    // Extract omega1 and nome q
    c10::complex<T> omega1 = params.omega1;
    c10::complex<T> q = params.q;

    // Compute v = π * z / (2 * ω₁)
    c10::complex<T> pi_c(pi, T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> v = pi_c * z / (two * omega1);

    // Compute θ₁(v, q)
    c10::complex<T> theta1_v = theta_1(v, q);

    // Compute θ₁'(0, q)
    c10::complex<T> theta1_prime_0 = detail::theta_1_derivative_at_zero(q);

    // Check for degenerate case
    if (std::abs(theta1_prime_0) < tol) {
        // Fallback to Taylor series
        c10::complex<T> z2 = z * z;
        c10::complex<T> z4 = z2 * z2;
        c10::complex<T> z5 = z4 * z;
        c10::complex<T> z7 = z5 * z2;
        c10::complex<T> c240(T(240), T(0));
        c10::complex<T> c840(T(840), T(0));
        return z - (g2 / c240) * z5 - (g3 / c840) * z7;
    }

    // Compute η₁ (quasi-period)
    c10::complex<T> eta1 = detail::compute_eta1(omega1, q);

    // Compute the exponential factor: exp(η₁z²/(2ω₁))
    c10::complex<T> exp_arg = eta1 * z * z / (two * omega1);
    c10::complex<T> exp_factor = std::exp(exp_arg);

    // Compute σ(z) = (2ω₁/π) * exp(η₁z²/(2ω₁)) * θ₁(v,q) / θ₁'(0,q)
    c10::complex<T> prefactor = two * omega1 / pi_c;

    return prefactor * exp_factor * theta1_v / theta1_prime_0;
}

} // namespace torchscience::kernel::special_functions
