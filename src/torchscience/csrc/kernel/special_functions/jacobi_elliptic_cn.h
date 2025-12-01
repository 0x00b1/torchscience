#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <vector>

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function cn(u, m)
//
// Mathematical definition:
// cn(u, m) = cos(am(u, m))
//
// where am(u, m) is the Jacobi amplitude function, defined as the inverse
// of the incomplete elliptic integral of the first kind:
// u = F(am, m) = integral from 0 to am of d(theta) / sqrt(1 - m * sin^2(theta))
//
// The parameter m is the "parameter convention" where m = k^2 (k is the modulus).
//
// Domain:
// - u: real or complex
// - m: 0 <= m <= 1 for real, complex plane otherwise
//
// Special values:
// - cn(0, m) = 1 for all m
// - cn(K(m), m) = 0 where K(m) is the complete elliptic integral of first kind
// - cn(u, 0) = cos(u)
// - cn(u, 1) = sech(u) = 1/cosh(u)
//
// Periodicity:
// - cn(u + 4K(m), m) = cn(u, m) where K(m) is the complete elliptic integral
//
// Algorithm:
// Uses the arithmetic-geometric mean (AGM) descending Landen transformation
// to compute cn(u, m) efficiently and accurately.

namespace detail {

template <typename T>
constexpr T jacobi_cn_eps();

template <>
constexpr float jacobi_cn_eps<float>() { return 1e-7f; }

template <>
constexpr double jacobi_cn_eps<double>() { return 1e-15; }

template <>
inline c10::Half jacobi_cn_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 jacobi_cn_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int jacobi_cn_max_iter() { return 50; }

} // namespace detail

template <typename T>
T jacobi_elliptic_cn(T u, T m) {
    const T eps = detail::jacobi_cn_eps<T>();

    // Special case: m = 0, cn(u, 0) = cos(u)
    if (std::abs(m) < eps) {
        return std::cos(u);
    }

    // Special case: m = 1, cn(u, 1) = sech(u)
    if (std::abs(m - T(1)) < eps) {
        return T(1) / std::cosh(u);
    }

    // Use the AGM (Arithmetic-Geometric Mean) method
    // Store the sequence of a_n and c_n values
    const int max_iter = detail::jacobi_cn_max_iter<T>();
    std::vector<T> a_vals;
    std::vector<T> c_vals;
    a_vals.reserve(max_iter);
    c_vals.reserve(max_iter);

    T a = T(1);
    T b = std::sqrt(T(1) - m);
    T c = std::sqrt(m);

    a_vals.push_back(a);
    c_vals.push_back(c);

    int n = 0;
    while (std::abs(c) > eps && n < max_iter) {
        T a_new = (a + b) / T(2);
        T b_new = std::sqrt(a * b);
        c = (a - b) / T(2);

        a = a_new;
        b = b_new;

        a_vals.push_back(a);
        c_vals.push_back(c);
        ++n;
    }

    // Compute phi_n = 2^n * a_n * u
    T phi = std::ldexp(a * u, n);  // phi = 2^n * a * u

    // Backward recurrence to compute phi_0 = am(u, m)
    // phi_{i-1} = (phi_i + asin(c_i/a_i * sin(phi_i))) / 2
    for (int i = n; i > 0; --i) {
        T sin_phi = std::sin(phi);
        T c_i = c_vals[i];
        T a_i = a_vals[i];
        T adjustment = std::asin(c_i / a_i * sin_phi);
        phi = (phi + adjustment) / T(2);
    }

    // cn(u, m) = cos(am(u, m)) = cos(phi)
    return std::cos(phi);
}

template <typename T>
c10::complex<T> jacobi_elliptic_cn(c10::complex<T> u, c10::complex<T> m) {
    const T eps = detail::jacobi_cn_eps<T>();
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> two(T(2), T(0));

    // Special case: m ~ 0, cn(u, 0) = cos(u)
    if (std::abs(m) < eps) {
        return std::cos(u);
    }

    // Special case: m ~ 1, cn(u, 1) = sech(u)
    if (std::abs(m - one) < eps) {
        return one / std::cosh(u);
    }

    // Use the AGM method for complex arguments
    const int max_iter = detail::jacobi_cn_max_iter<T>();
    std::vector<c10::complex<T>> a_vals;
    std::vector<c10::complex<T>> c_vals;
    a_vals.reserve(max_iter);
    c_vals.reserve(max_iter);

    c10::complex<T> a = one;
    c10::complex<T> b = std::sqrt(one - m);
    c10::complex<T> c = std::sqrt(m);

    a_vals.push_back(a);
    c_vals.push_back(c);

    int n = 0;
    while (std::abs(c) > eps && n < max_iter) {
        c10::complex<T> a_new = (a + b) / two;
        c10::complex<T> b_new = std::sqrt(a * b);
        c = (a - b) / two;

        a = a_new;
        b = b_new;

        a_vals.push_back(a);
        c_vals.push_back(c);
        ++n;
    }

    // Compute phi_n = 2^n * a_n * u
    c10::complex<T> scale(std::ldexp(T(1), n), T(0));
    c10::complex<T> phi = scale * a * u;

    // Backward recurrence
    for (int i = n; i > 0; --i) {
        c10::complex<T> sin_phi = std::sin(phi);
        c10::complex<T> c_i = c_vals[i];
        c10::complex<T> a_i = a_vals[i];
        c10::complex<T> adjustment = std::asin(c_i / a_i * sin_phi);
        phi = (phi + adjustment) / two;
    }

    // cn(u, m) = cos(am(u, m)) = cos(phi)
    return std::cos(phi);
}

} // namespace torchscience::kernel::special_functions
