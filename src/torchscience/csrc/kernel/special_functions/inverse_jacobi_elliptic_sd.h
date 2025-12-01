#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "jacobi_elliptic_sd.h"
#include "jacobi_elliptic_sn.h"
#include "jacobi_elliptic_cn.h"
#include "jacobi_elliptic_dn.h"

namespace torchscience::kernel::special_functions {

// Inverse Jacobi elliptic function arcsd(x, m)
//
// Mathematical definition:
// arcsd(x, m) = u such that sd(u, m) = x
//
// where:
// - sd(u, m) = sn(u, m) / dn(u, m) is the Jacobi elliptic sd function
//
// Domain:
// - x: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - arcsd(0, m) = 0 for all m (since sd(0, m) = 0)
// - arcsd(x, 0) = arcsin(x) (circular limit)
// - arcsd(x, 1) = arcsinh(x) (hyperbolic limit)
//
// Algorithm:
// Uses Newton's method to solve sd(u, m) = x for u.
// The derivative d(sd)/du = cn(u,m) / dn(u,m)^2 is used for Newton iteration.
//
// Initial guess:
// - For small x: u ≈ x (since sd(u,m) ≈ u for small u)
// - For m ≈ 0: u ≈ arcsin(x)
// - For m ≈ 1: u ≈ arcsinh(x)

namespace detail {

template <typename T>
constexpr int inverse_jacobi_sd_max_iterations() { return 50; }

template <typename T>
constexpr T inverse_jacobi_sd_tolerance();

template <>
constexpr float inverse_jacobi_sd_tolerance<float>() { return 1e-6f; }

template <>
constexpr double inverse_jacobi_sd_tolerance<double>() { return 1e-14; }

template <>
inline c10::Half inverse_jacobi_sd_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 inverse_jacobi_sd_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

template <typename T>
T inverse_jacobi_elliptic_sd(T x, T m) {
    const T tol = detail::inverse_jacobi_sd_tolerance<T>();
    const int max_iter = detail::inverse_jacobi_sd_max_iterations<T>();

    // Special case: x = 0 => arcsd(0, m) = 0
    if (std::abs(x) < tol) {
        return T(0);
    }

    // Special case: m = 0 => sd(u, 0) = sin(u), so arcsd(x, 0) = arcsin(x)
    if (std::abs(m) < tol) {
        // For x outside [-1, 1], arcsin is not defined for real numbers
        // but we return the principal value clamped
        if (x > T(1)) {
            return std::asin(T(1));
        } else if (x < T(-1)) {
            return std::asin(T(-1));
        }
        return std::asin(x);
    }

    // Special case: m = 1 => sd(u, 1) = sinh(u), so arcsd(x, 1) = arcsinh(x)
    if (std::abs(m - T(1)) < tol) {
        return std::asinh(x);
    }

    // Initial guess based on limiting cases and small-argument approximation
    T u;
    if (std::abs(x) < T(0.5)) {
        // For small x, sd(u, m) ≈ u, so u ≈ x
        u = x;
    } else if (m < T(0.5)) {
        // Closer to circular: use arcsin-like guess
        T x_clamped = std::min(std::max(x, T(-1)), T(1));
        u = std::asin(x_clamped);
    } else {
        // Closer to hyperbolic: use arcsinh-like guess
        u = std::asinh(x);
    }

    // Newton's method: solve sd(u, m) - x = 0
    // f(u) = sd(u, m) - x
    // f'(u) = d(sd)/du = cn(u, m) / dn(u, m)^2
    for (int iter = 0; iter < max_iter; ++iter) {
        T sd_u = jacobi_elliptic_sd(u, m);
        T cn_u = jacobi_elliptic_cn(u, m);
        T dn_u = jacobi_elliptic_dn(u, m);

        T f = sd_u - x;

        // Check convergence
        if (std::abs(f) < tol) {
            break;
        }

        // Derivative: d(sd)/du = d(sn/dn)/du = (cn*dn^2 + sn*m*sn*cn/dn) / dn^2
        // Simplified: d(sd)/du = cn / dn^2 (using dn^2 = 1 - m*sn^2 and chain rule)
        // Actually: d(sn/dn)/du = (cn*dn - sn*(-m*sn*cn/dn)) / dn^2
        //                       = (cn*dn + m*sn^2*cn/dn) / dn^2
        //                       = cn*(dn^2 + m*sn^2) / dn^3
        //                       = cn*(1 - m*sn^2 + m*sn^2) / dn^3
        //                       = cn / dn^3
        // Wait, let me recalculate:
        // sd = sn/dn
        // d(sd)/du = (d(sn)/du * dn - sn * d(dn)/du) / dn^2
        // d(sn)/du = cn*dn
        // d(dn)/du = -m*sn*cn
        // d(sd)/du = (cn*dn*dn - sn*(-m*sn*cn)) / dn^2
        //          = (cn*dn^2 + m*sn^2*cn) / dn^2
        //          = cn*(dn^2 + m*sn^2) / dn^2
        //          = cn*(1 - m*sn^2 + m*sn^2) / dn^2  [using dn^2 = 1 - m*sn^2]
        //          = cn / dn^2

        T dn_sq = dn_u * dn_u;
        if (std::abs(dn_sq) < tol * tol) {
            // Near a pole, use smaller step
            u = u - f * tol;
            continue;
        }

        T f_prime = cn_u / dn_sq;

        // Avoid division by near-zero derivative
        if (std::abs(f_prime) < tol) {
            // Use a damped step
            u = u - f * T(0.1);
            continue;
        }

        T delta = f / f_prime;

        // Damping for large steps to improve convergence
        if (std::abs(delta) > T(1)) {
            delta = delta / std::abs(delta);  // Sign of delta
        }

        u = u - delta;
    }

    return u;
}

template <typename T>
c10::complex<T> inverse_jacobi_elliptic_sd(c10::complex<T> x, c10::complex<T> m) {
    const T tol = detail::inverse_jacobi_sd_tolerance<T>();
    const int max_iter = detail::inverse_jacobi_sd_max_iterations<T>();

    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> one(T(1), T(0));

    // Special case: x = 0 => arcsd(0, m) = 0
    if (std::abs(x) < tol) {
        return zero;
    }

    // Special case: m = 0 => arcsd(x, 0) = arcsin(x)
    if (std::abs(m) < tol) {
        return std::asin(x);
    }

    // Special case: m = 1 => arcsd(x, 1) = arcsinh(x)
    if (std::abs(m - one) < tol) {
        return std::asinh(x);
    }

    // Initial guess
    c10::complex<T> u;
    if (std::abs(x) < T(0.5)) {
        u = x;
    } else {
        u = std::asinh(x);
    }

    // Newton's method
    for (int iter = 0; iter < max_iter; ++iter) {
        c10::complex<T> sd_u = jacobi_elliptic_sd(u, m);
        c10::complex<T> cn_u = jacobi_elliptic_cn(u, m);
        c10::complex<T> dn_u = jacobi_elliptic_dn(u, m);

        c10::complex<T> f = sd_u - x;

        if (std::abs(f) < tol) {
            break;
        }

        c10::complex<T> dn_sq = dn_u * dn_u;
        if (std::abs(dn_sq) < tol * tol) {
            u = u - f * c10::complex<T>(tol, T(0));
            continue;
        }

        c10::complex<T> f_prime = cn_u / dn_sq;

        if (std::abs(f_prime) < tol) {
            u = u - f * c10::complex<T>(T(0.1), T(0));
            continue;
        }

        c10::complex<T> delta = f / f_prime;

        // Damping for large steps
        if (std::abs(delta) > T(1)) {
            delta = delta / std::abs(delta);
        }

        u = u - delta;
    }

    return u;
}

} // namespace torchscience::kernel::special_functions
