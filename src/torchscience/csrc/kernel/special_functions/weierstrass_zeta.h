#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "weierstrass_sigma.h"

namespace torchscience::kernel::special_functions {

// Weierstrass zeta function zeta(z; g2, g3)
//
// Mathematical definition:
// The Weierstrass zeta function is defined as the logarithmic derivative
// of the sigma function:
//
//   zeta(z) = sigma'(z) / sigma(z)
//
// Equivalently, zeta'(z) = -P(z) (negative of Weierstrass P function)
//
// Laurent expansion around z = 0:
// zeta(z) = 1/z - (g2/60)*z^3 - (g3/140)*z^5 + O(z^7)
//
// Special properties:
// - zeta(0) = infinity (simple pole at the origin)
// - zeta(-z) = -zeta(z) (odd function)
// - zeta is NOT periodic (quasi-periodic with additive constants)
// - zeta(z + 2*omega_i) = zeta(z) + 2*eta_i (quasi-periodicity)
//
// Domain:
// - z: complex (the argument)
// - g2, g3: complex (the Weierstrass invariants)
//
// Algorithm:
// 1. Handle z = 0: return infinity (simple pole)
// 2. For small |z|, use Laurent series for accuracy
// 3. Otherwise, compute zeta(z) = sigma'(z)/sigma(z) via numerical differentiation

namespace detail {

template <typename T>
inline T weierstrass_zeta_tolerance() {
    return T(1e-10);
}

template <>
inline float weierstrass_zeta_tolerance<float>() { return 1e-5f; }

template <>
inline double weierstrass_zeta_tolerance<double>() { return 1e-14; }

template <>
inline c10::Half weierstrass_zeta_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 weierstrass_zeta_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
inline T weierstrass_zeta_finite_diff_step() {
    return std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
}

template <>
inline float weierstrass_zeta_finite_diff_step<float>() {
    return 1e-4f;
}

template <>
inline double weierstrass_zeta_finite_diff_step<double>() {
    return 1e-7;
}

} // namespace detail

template <typename T>
T weierstrass_zeta(T z, T g2, T g3) {
    const T tol = detail::weierstrass_zeta_tolerance<T>();

    // Handle z = 0: simple pole, return infinity
    if (std::abs(z) < tol) {
        return std::numeric_limits<T>::infinity();
    }

    // For small z, use the Laurent series expansion for better accuracy
    // zeta(z) = 1/z - (g2/60)*z^3 - (g3/140)*z^5 + O(z^7)
    if (std::abs(z) < T(0.1)) {
        T z2 = z * z;
        T z3 = z2 * z;
        T z5 = z3 * z2;
        return T(1) / z - (g2 / T(60)) * z3 - (g3 / T(140)) * z5;
    }

    // Compute zeta(z) = sigma'(z) / sigma(z) via numerical differentiation
    // sigma'(z) = (sigma(z+h) - sigma(z-h)) / (2h)
    T h = detail::weierstrass_zeta_finite_diff_step<T>();

    T sigma_center = weierstrass_sigma(z, g2, g3);

    // If sigma is very small, we're near a zero (lattice point)
    // In this case, zeta has a simple pole
    if (std::abs(sigma_center) < tol) {
        return std::numeric_limits<T>::infinity();
    }

    T sigma_plus = weierstrass_sigma(z + h, g2, g3);
    T sigma_minus = weierstrass_sigma(z - h, g2, g3);

    T sigma_prime = (sigma_plus - sigma_minus) / (T(2) * h);

    return sigma_prime / sigma_center;
}

template <typename T>
c10::complex<T> weierstrass_zeta(c10::complex<T> z, c10::complex<T> g2, c10::complex<T> g3) {
    const T tol = detail::weierstrass_zeta_tolerance<T>();

    // Handle z = 0: simple pole, return infinity
    if (std::abs(z) < tol) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    // For small z, use the Laurent series expansion for better accuracy
    // zeta(z) = 1/z - (g2/60)*z^3 - (g3/140)*z^5 + O(z^7)
    if (std::abs(z) < T(0.1)) {
        c10::complex<T> one(T(1), T(0));
        c10::complex<T> sixty(T(60), T(0));
        c10::complex<T> one_forty(T(140), T(0));
        c10::complex<T> z2 = z * z;
        c10::complex<T> z3 = z2 * z;
        c10::complex<T> z5 = z3 * z2;
        return one / z - (g2 / sixty) * z3 - (g3 / one_forty) * z5;
    }

    // Compute zeta(z) = sigma'(z) / sigma(z) via numerical differentiation
    T h = detail::weierstrass_zeta_finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    c10::complex<T> sigma_center = weierstrass_sigma(z, g2, g3);

    // If sigma is very small, we're near a zero (lattice point)
    if (std::abs(sigma_center) < tol) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    c10::complex<T> sigma_plus = weierstrass_sigma(z + h_complex, g2, g3);
    c10::complex<T> sigma_minus = weierstrass_sigma(z - h_complex, g2, g3);

    c10::complex<T> sigma_prime = (sigma_plus - sigma_minus) / (T(2) * h_complex);

    return sigma_prime / sigma_center;
}

} // namespace torchscience::kernel::special_functions
