#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "jacobi_elliptic_cd.h"
#include "jacobi_elliptic_sn.h"
#include "jacobi_elliptic_cn.h"
#include "jacobi_elliptic_dn.h"

namespace torchscience::kernel::special_functions {

// Inverse Jacobi elliptic function arccd(x, m)
//
// Mathematical definition:
// arccd(x, m) = u such that cd(u, m) = x
//
// where:
// - cd(u, m) = cn(u, m) / dn(u, m) is the Jacobi elliptic cd function
//
// Domain:
// - x: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - arccd(1, m) = 0 for all m (since cd(0, m) = 1)
// - arccd(x, 0) = arccos(x) (circular limit)
// - arccd(x, 1) = 0 for all x (since cd(u, 1) = 1 for all u when cn and dn are both sech)
//
// Algorithm:
// Uses Newton's method to solve cd(u, m) = x for u.
// The derivative d(cd)/du = -sn(u,m) / dn(u,m)^2 is used for Newton iteration.
//
// Note on the derivative:
// cd = cn/dn
// d(cd)/du = (d(cn)/du * dn - cn * d(dn)/du) / dn^2
// d(cn)/du = -sn*dn
// d(dn)/du = -m*sn*cn
// d(cd)/du = (-sn*dn*dn - cn*(-m*sn*cn)) / dn^2
//          = (-sn*dn^2 + m*sn*cn^2) / dn^2
//          = -sn*(dn^2 - m*cn^2) / dn^2
//          = -sn*(1 - m*sn^2 - m*cn^2) / dn^2
//          = -sn*(1 - m*(sn^2 + cn^2)) / dn^2
//          = -sn*(1 - m) / dn^2  [using sn^2 + cn^2 = 1]

namespace detail {

template <typename T>
constexpr int inverse_jacobi_cd_max_iterations() { return 50; }

template <typename T>
constexpr T inverse_jacobi_cd_tolerance();

template <>
constexpr float inverse_jacobi_cd_tolerance<float>() { return 1e-6f; }

template <>
constexpr double inverse_jacobi_cd_tolerance<double>() { return 1e-14; }

template <>
inline c10::Half inverse_jacobi_cd_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 inverse_jacobi_cd_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

template <typename T>
T inverse_jacobi_elliptic_cd(T x, T m) {
    const T tol = detail::inverse_jacobi_cd_tolerance<T>();
    const int max_iter = detail::inverse_jacobi_cd_max_iterations<T>();

    // Special case: x = 1 => arccd(1, m) = 0 (since cd(0, m) = 1)
    if (std::abs(x - T(1)) < tol) {
        return T(0);
    }

    // Special case: m = 0 => cd(u, 0) = cos(u), so arccd(x, 0) = arccos(x)
    if (std::abs(m) < tol) {
        // For x outside [-1, 1], arccos is not defined for real numbers
        if (x > T(1)) {
            return T(0);  // arccos(1) = 0
        } else if (x < T(-1)) {
            return std::acos(T(-1));  // pi
        }
        return std::acos(x);
    }

    // Special case: m = 1 => cd(u, 1) = cn(u,1)/dn(u,1) = sech(u)/sech(u) = 1
    // So arccd(x, 1) only exists if x = 1 (handled above)
    // For x != 1, return NaN or a reasonable approximation
    if (std::abs(m - T(1)) < tol) {
        if (std::abs(x - T(1)) < T(0.1)) {
            // Near x = 1, return a small value
            return T(0);
        }
        // No solution exists for x != 1 when m = 1
        return std::numeric_limits<T>::quiet_NaN();
    }

    // Initial guess based on limiting cases
    T u;
    if (std::abs(x - T(1)) < T(0.1)) {
        // Near x = 1 (cd(0, m) = 1), use small u
        // cd(u, m) ≈ 1 - u^2/2 * (derivative at 0)
        // d(cd)/du|_{u=0} = -sn(0)*(...) = 0
        // So we need second derivative or just start with sqrt(2*(1-x))
        u = std::sqrt(T(2) * (T(1) - x));
    } else if (m < T(0.5)) {
        // Closer to circular: use arccos-like guess
        T x_clamped = std::min(std::max(x, T(-1)), T(1));
        u = std::acos(x_clamped);
    } else {
        // General case: use arccos as starting point scaled
        T x_clamped = std::min(std::max(x, T(-1)), T(1));
        u = std::acos(x_clamped);
    }

    // Newton's method: solve cd(u, m) - x = 0
    // f(u) = cd(u, m) - x
    // f'(u) = d(cd)/du = -sn(u, m) * (1 - m) / dn(u, m)^2
    for (int iter = 0; iter < max_iter; ++iter) {
        T cd_u = jacobi_elliptic_cd(u, m);
        T sn_u = jacobi_elliptic_sn(u, m);
        T dn_u = jacobi_elliptic_dn(u, m);

        T f = cd_u - x;

        // Check convergence
        if (std::abs(f) < tol) {
            break;
        }

        T dn_sq = dn_u * dn_u;
        if (std::abs(dn_sq) < tol * tol) {
            // Near a pole, use smaller step
            u = u - f * tol;
            continue;
        }

        // f'(u) = -sn * (1 - m) / dn^2
        T f_prime = -sn_u * (T(1) - m) / dn_sq;

        // Avoid division by near-zero derivative
        if (std::abs(f_prime) < tol) {
            // When derivative is small, use a fixed step
            // This happens when sn(u) ≈ 0, i.e., u ≈ 0 or u ≈ 2K(m)
            u = u - f * T(0.1);
            continue;
        }

        T delta = f / f_prime;

        // Damping for large steps to improve convergence
        if (std::abs(delta) > T(1)) {
            delta = delta / std::abs(delta);  // Sign of delta
        }

        u = u - delta;

        // Ensure u stays in reasonable range
        if (u < T(0)) {
            u = tol;
        }
    }

    return u;
}

template <typename T>
c10::complex<T> inverse_jacobi_elliptic_cd(c10::complex<T> x, c10::complex<T> m) {
    const T tol = detail::inverse_jacobi_cd_tolerance<T>();
    const int max_iter = detail::inverse_jacobi_cd_max_iterations<T>();

    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> one(T(1), T(0));

    // Special case: x = 1 => arccd(1, m) = 0
    if (std::abs(x - one) < tol) {
        return zero;
    }

    // Special case: m = 0 => arccd(x, 0) = arccos(x)
    if (std::abs(m) < tol) {
        return std::acos(x);
    }

    // Special case: m = 1 => cd(u, 1) = 1, so arccd(x, 1) only defined for x = 1
    if (std::abs(m - one) < tol) {
        if (std::abs(x - one) < T(0.1)) {
            return zero;
        }
        return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), T(0));
    }

    // Initial guess
    c10::complex<T> u;
    if (std::abs(x - one) < T(0.5)) {
        u = std::sqrt(c10::complex<T>(T(2), T(0)) * (one - x));
    } else {
        u = std::acos(x);
    }

    // Newton's method
    for (int iter = 0; iter < max_iter; ++iter) {
        c10::complex<T> cd_u = jacobi_elliptic_cd(u, m);
        c10::complex<T> sn_u = jacobi_elliptic_sn(u, m);
        c10::complex<T> dn_u = jacobi_elliptic_dn(u, m);

        c10::complex<T> f = cd_u - x;

        if (std::abs(f) < tol) {
            break;
        }

        c10::complex<T> dn_sq = dn_u * dn_u;
        if (std::abs(dn_sq) < tol * tol) {
            u = u - f * c10::complex<T>(tol, T(0));
            continue;
        }

        c10::complex<T> f_prime = -sn_u * (one - m) / dn_sq;

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
