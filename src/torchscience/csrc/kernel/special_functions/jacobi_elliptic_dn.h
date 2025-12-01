#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function dn(u, m)
//
// Mathematical definition:
// dn(u, m) = sqrt(1 - m * sn^2(u, m)) = sqrt(1 - m * sin^2(am(u, m)))
//
// where:
// - am(u, m) is the Jacobi amplitude function
// - sn(u, m) = sin(am(u, m)) is the Jacobi elliptic sine
//
// Domain:
// - u: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - dn(0, m) = 1 for all m
// - dn(u, 0) = 1 for all u
// - dn(u, 1) = sech(u)
// - dn(K(m), m) = sqrt(1-m) where K(m) is the complete elliptic integral
//
// The function is periodic with period 2K(m) where K is the complete
// elliptic integral of the first kind.

namespace detail {

// Compute Jacobi amplitude am(u, m) using the arithmetic-geometric mean (AGM)
// This is the inverse of the incomplete elliptic integral of the first kind
template <typename T>
T jacobi_amplitude_am(T u, T m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    const int max_iter = 30;

    // Handle special cases
    if (std::abs(m) < eps) {
        // m = 0: am(u, 0) = u
        return u;
    }

    if (std::abs(m - T(1)) < eps) {
        // m = 1: am(u, 1) = gd(u) = 2*atan(exp(u)) - pi/2 = atan(sinh(u))
        return std::atan(std::sinh(u));
    }

    // AGM algorithm for computing the Jacobi amplitude
    // Store the sequence of scaling factors
    T a[max_iter + 1];
    T c[max_iter + 1];

    a[0] = T(1);
    T b = std::sqrt(T(1) - m);
    c[0] = std::sqrt(m);

    int n = 0;
    for (n = 0; n < max_iter; ++n) {
        if (std::abs(c[n]) < eps) {
            break;
        }
        T a_new = (a[n] + b) / T(2);
        T b_new = std::sqrt(a[n] * b);
        c[n + 1] = (a[n] - b) / T(2);
        a[n + 1] = a_new;
        b = b_new;
    }

    // Compute phi_n = 2^n * a_n * u
    T phi = std::ldexp(a[n] * u, n);

    // Backward recurrence to find phi_0 = am(u, m)
    for (int k = n; k > 0; --k) {
        phi = (phi + std::asin(c[k] * std::sin(phi) / a[k])) / T(2);
    }

    return phi;
}

template <typename T>
c10::complex<T> jacobi_amplitude_am(c10::complex<T> u, c10::complex<T> m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    const int max_iter = 30;

    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));

    // Handle special cases
    if (std::abs(m) < eps) {
        // m = 0: am(u, 0) = u
        return u;
    }

    if (std::abs(m - one) < eps) {
        // m = 1: am(u, 1) = atan(sinh(u))
        return std::atan(std::sinh(u));
    }

    // AGM algorithm for complex numbers
    c10::complex<T> a[max_iter + 1];
    c10::complex<T> c_arr[max_iter + 1];

    a[0] = one;
    c10::complex<T> b = std::sqrt(one - m);
    c_arr[0] = std::sqrt(m);

    int n = 0;
    for (n = 0; n < max_iter; ++n) {
        if (std::abs(c_arr[n]) < eps) {
            break;
        }
        c10::complex<T> a_new = (a[n] + b) / two;
        c10::complex<T> b_new = std::sqrt(a[n] * b);
        c_arr[n + 1] = (a[n] - b) / two;
        a[n + 1] = a_new;
        b = b_new;
    }

    // Compute phi_n = 2^n * a_n * u
    c10::complex<T> phi = a[n] * u * c10::complex<T>(std::ldexp(T(1), n), T(0));

    // Backward recurrence
    for (int k = n; k > 0; --k) {
        phi = (phi + std::asin(c_arr[k] * std::sin(phi) / a[k])) / two;
    }

    return phi;
}

} // namespace detail

template <typename T>
T jacobi_elliptic_dn(T u, T m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Special case: m = 0
    // dn(u, 0) = 1 for all u
    if (std::abs(m) < eps) {
        return T(1);
    }

    // Special case: m = 1
    // dn(u, 1) = sech(u) = 1/cosh(u)
    if (std::abs(m - T(1)) < eps) {
        return T(1) / std::cosh(u);
    }

    // General case: dn(u, m) = sqrt(1 - m * sin^2(am(u, m)))
    T am = detail::jacobi_amplitude_am(u, m);
    T sin_am = std::sin(am);
    T dn_squared = T(1) - m * sin_am * sin_am;

    // Ensure non-negative for numerical stability
    if (dn_squared < T(0)) {
        dn_squared = T(0);
    }

    return std::sqrt(dn_squared);
}

template <typename T>
c10::complex<T> jacobi_elliptic_dn(c10::complex<T> u, c10::complex<T> m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    c10::complex<T> one(T(1), T(0));

    // Special case: m = 0
    if (std::abs(m) < eps) {
        return one;
    }

    // Special case: m = 1
    if (std::abs(m - one) < eps) {
        return one / std::cosh(u);
    }

    // General case: dn(u, m) = sqrt(1 - m * sin^2(am(u, m)))
    c10::complex<T> am = detail::jacobi_amplitude_am(u, m);
    c10::complex<T> sin_am = std::sin(am);
    c10::complex<T> dn_squared = one - m * sin_am * sin_am;

    return std::sqrt(dn_squared);
}

} // namespace torchscience::kernel::special_functions
