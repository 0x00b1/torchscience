#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "jacobi_elliptic_sc.h"
#include "jacobi_elliptic_sn.h"
#include "jacobi_elliptic_cn.h"
#include "jacobi_elliptic_dn.h"

namespace torchscience::kernel::special_functions {

// Inverse Jacobi elliptic function arcsc(x, m)
//
// Mathematical definition:
// arcsc(x, m) = u such that sc(u, m) = x
//
// where:
// - sc(u, m) = sn(u, m) / cn(u, m) is the Jacobi elliptic sc function
//
// Domain:
// - x: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - arcsc(0, m) = 0 for all m (since sc(0, m) = 0)
// - arcsc(x, 0) = arctan(x) (circular limit, since sc(u,0) = tan(u))
// - arcsc(x, 1) = arcsinh(x) (hyperbolic limit, since sc(u,1) = sinh(u))
//
// Algorithm:
// Uses Newton's method to solve sc(u, m) = x for u.
// The derivative d(sc)/du is used for Newton iteration.
//
// Note on the derivative:
// sc = sn/cn
// d(sc)/du = (d(sn)/du * cn - sn * d(cn)/du) / cn^2
// d(sn)/du = cn*dn
// d(cn)/du = -sn*dn
// d(sc)/du = (cn*dn*cn - sn*(-sn*dn)) / cn^2
//          = (cn^2*dn + sn^2*dn) / cn^2
//          = dn*(cn^2 + sn^2) / cn^2
//          = dn / cn^2  [using sn^2 + cn^2 = 1]

namespace detail {

template <typename T>
constexpr int inverse_jacobi_sc_max_iterations() { return 50; }

template <typename T>
constexpr T inverse_jacobi_sc_tolerance();

template <>
constexpr float inverse_jacobi_sc_tolerance<float>() { return 1e-6f; }

template <>
constexpr double inverse_jacobi_sc_tolerance<double>() { return 1e-14; }

template <>
inline c10::Half inverse_jacobi_sc_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 inverse_jacobi_sc_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

template <typename T>
T inverse_jacobi_elliptic_sc(T x, T m) {
    const T tol = detail::inverse_jacobi_sc_tolerance<T>();
    const int max_iter = detail::inverse_jacobi_sc_max_iterations<T>();

    // Special case: x = 0 => arcsc(0, m) = 0
    if (std::abs(x) < tol) {
        return T(0);
    }

    // Special case: m = 0 => sc(u, 0) = tan(u), so arcsc(x, 0) = arctan(x)
    if (std::abs(m) < tol) {
        return std::atan(x);
    }

    // Special case: m = 1 => sc(u, 1) = sinh(u), so arcsc(x, 1) = arcsinh(x)
    if (std::abs(m - T(1)) < tol) {
        return std::asinh(x);
    }

    // Initial guess based on limiting cases and small-argument approximation
    T u;
    if (std::abs(x) < T(0.5)) {
        // For small x, sc(u, m) ≈ u, so u ≈ x
        u = x;
    } else if (m < T(0.5)) {
        // Closer to circular: use arctan-like guess
        u = std::atan(x);
    } else {
        // Closer to hyperbolic: use arcsinh-like guess
        u = std::asinh(x);
    }

    // Newton's method: solve sc(u, m) - x = 0
    // f(u) = sc(u, m) - x
    // f'(u) = d(sc)/du = dn(u, m) / cn(u, m)^2
    for (int iter = 0; iter < max_iter; ++iter) {
        T sc_u = jacobi_elliptic_sc(u, m);
        T cn_u = jacobi_elliptic_cn(u, m);
        T dn_u = jacobi_elliptic_dn(u, m);

        T f = sc_u - x;

        // Check convergence
        if (std::abs(f) < tol) {
            break;
        }

        T cn_sq = cn_u * cn_u;
        if (std::abs(cn_sq) < tol * tol) {
            // Near a pole (cn = 0), use smaller step
            u = u - f * tol;
            continue;
        }

        // f'(u) = dn / cn^2
        T f_prime = dn_u / cn_sq;

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
c10::complex<T> inverse_jacobi_elliptic_sc(c10::complex<T> x, c10::complex<T> m) {
    const T tol = detail::inverse_jacobi_sc_tolerance<T>();
    const int max_iter = detail::inverse_jacobi_sc_max_iterations<T>();

    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> one(T(1), T(0));

    // Special case: x = 0 => arcsc(0, m) = 0
    if (std::abs(x) < tol) {
        return zero;
    }

    // Special case: m = 0 => arcsc(x, 0) = arctan(x)
    if (std::abs(m) < tol) {
        return std::atan(x);
    }

    // Special case: m = 1 => arcsc(x, 1) = arcsinh(x)
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
        c10::complex<T> sc_u = jacobi_elliptic_sc(u, m);
        c10::complex<T> cn_u = jacobi_elliptic_cn(u, m);
        c10::complex<T> dn_u = jacobi_elliptic_dn(u, m);

        c10::complex<T> f = sc_u - x;

        if (std::abs(f) < tol) {
            break;
        }

        c10::complex<T> cn_sq = cn_u * cn_u;
        if (std::abs(cn_sq) < tol * tol) {
            u = u - f * c10::complex<T>(tol, T(0));
            continue;
        }

        c10::complex<T> f_prime = dn_u / cn_sq;

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
