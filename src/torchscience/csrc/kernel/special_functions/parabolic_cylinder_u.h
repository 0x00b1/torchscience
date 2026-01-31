#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "gamma.h"
#include "log_gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Tolerance constants for parabolic cylinder functions
template <typename T>
constexpr T pcf_eps();

template <>
constexpr float pcf_eps<float>() { return 1e-7f; }

template <>
constexpr double pcf_eps<double>() { return 1e-15; }

template <>
inline c10::Half pcf_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 pcf_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int pcf_max_iter() { return 500; }

// Taylor series for U(a, z) around z=0
// DLMF 12.4.1: U(a,z) = U1(a) * y1(a,z) + U2(a) * y2(a,z)
// where y1, y2 are even/odd power series solutions
template <typename T>
T parabolic_cylinder_u_taylor(T a, T z) {
    const T eps = pcf_eps<T>();
    const int max_iter = pcf_max_iter<T>();
    const T pi = static_cast<T>(M_PI);
    const T sqrt_pi = std::sqrt(pi);

    // Compute coefficients U1(a) and U2(a) from DLMF 12.4.3-12.4.4
    T half_a = a / T(2);
    T log_U1 = std::log(sqrt_pi) - (a + T(1)) / T(2) * std::log(T(2)) - log_gamma((T(1) - a) / T(2));
    T log_U2 = std::log(sqrt_pi) - half_a * std::log(T(2)) - log_gamma(-half_a);

    T U1 = std::exp(log_U1);
    T U2 = std::exp(log_U2);

    if (!std::isfinite(U1)) U1 = T(0);
    if (!std::isfinite(U2)) U2 = T(0);

    T z2 = z * z;
    T y1 = T(1);
    T y2 = z;

    T term_even = T(1);
    T term_odd = T(1);

    for (int n = 0; n < max_iter; ++n) {
        T coeff_even = (a + T(n) + T(0.5)) / (T(2*n + 1) * T(2*n + 2));
        term_even *= coeff_even * z2;
        y1 += term_even;

        T coeff_odd = (a + T(n) + T(1)) / (T(2*n + 2) * T(2*n + 3));
        term_odd *= coeff_odd * z2;
        y2 += term_odd * z;

        if (std::abs(term_even) < eps * std::abs(y1) &&
            std::abs(term_odd * z) < eps * std::abs(y2)) {
            break;
        }
    }

    return U1 * y1 + U2 * y2;
}

// Asymptotic expansion for U(a, z) for large |z|
template <typename T>
T parabolic_cylinder_u_asymptotic(T a, T z) {
    const T eps = pcf_eps<T>();
    const int max_terms = 50;

    T z2 = z * z;
    T log_prefix = -z2 / T(4) + (-a - T(0.5)) * std::log(std::abs(z));

    T sum = T(1);
    T term = T(1);
    T neg_2z2_inv = T(-1) / (T(2) * z2);

    for (int s = 1; s < max_terms; ++s) {
        T factor = (T(0.5) + a + T(2*s - 2)) * (T(0.5) + a + T(2*s - 1)) / T(s);
        term *= factor * neg_2z2_inv;
        sum += term;

        if (std::abs(term) < eps * std::abs(sum)) {
            break;
        }
        if (std::abs(term) > T(1e10)) {
            break;
        }
    }

    return std::exp(log_prefix) * sum;
}

// Complex Taylor series for U(a, z)
template <typename T>
c10::complex<T> parabolic_cylinder_u_taylor(c10::complex<T> a, c10::complex<T> z) {
    const T eps = pcf_eps<T>();
    const int max_iter = pcf_max_iter<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> half(T(0.5), T(0));
    const T pi = static_cast<T>(M_PI);
    const T sqrt_pi = std::sqrt(pi);
    const c10::complex<T> sqrt_pi_c(sqrt_pi, T(0));

    c10::complex<T> half_a = a / two;
    c10::complex<T> log_U1 = std::log(sqrt_pi_c) - (a + one) / two * std::log(two) - log_gamma((one - a) / two);
    c10::complex<T> log_U2 = std::log(sqrt_pi_c) - half_a * std::log(two) - log_gamma(-half_a);

    c10::complex<T> U1 = std::exp(log_U1);
    c10::complex<T> U2 = std::exp(log_U2);

    if (!std::isfinite(std::abs(U1))) U1 = c10::complex<T>(T(0), T(0));
    if (!std::isfinite(std::abs(U2))) U2 = c10::complex<T>(T(0), T(0));

    c10::complex<T> z2 = z * z;
    c10::complex<T> y1 = one;
    c10::complex<T> y2 = z;

    c10::complex<T> term_even = one;
    c10::complex<T> term_odd = one;

    for (int n = 0; n < max_iter; ++n) {
        c10::complex<T> n_c(T(n), T(0));
        c10::complex<T> coeff_even = (a + n_c + half) / (c10::complex<T>(T(2*n + 1), T(0)) * c10::complex<T>(T(2*n + 2), T(0)));
        term_even *= coeff_even * z2;
        y1 += term_even;

        c10::complex<T> coeff_odd = (a + n_c + one) / (c10::complex<T>(T(2*n + 2), T(0)) * c10::complex<T>(T(2*n + 3), T(0)));
        term_odd *= coeff_odd * z2;
        y2 += term_odd * z;

        if (std::abs(term_even) < eps * std::abs(y1) &&
            std::abs(term_odd * z) < eps * std::abs(y2)) {
            break;
        }
    }

    return U1 * y1 + U2 * y2;
}

// Complex asymptotic expansion for U(a, z)
template <typename T>
c10::complex<T> parabolic_cylinder_u_asymptotic(c10::complex<T> a, c10::complex<T> z) {
    const T eps = pcf_eps<T>();
    const int max_terms = 50;
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> half(T(0.5), T(0));
    const c10::complex<T> four(T(4), T(0));

    c10::complex<T> z2 = z * z;
    c10::complex<T> log_prefix = -z2 / four + (-a - half) * std::log(z);

    c10::complex<T> sum = one;
    c10::complex<T> term = one;
    c10::complex<T> neg_2z2_inv = -one / (two * z2);

    for (int s = 1; s < max_terms; ++s) {
        c10::complex<T> s_c(T(s), T(0));
        c10::complex<T> factor = (half + a + c10::complex<T>(T(2*s - 2), T(0))) *
                                  (half + a + c10::complex<T>(T(2*s - 1), T(0))) / s_c;
        term *= factor * neg_2z2_inv;
        sum += term;

        if (std::abs(term) < eps * std::abs(sum)) {
            break;
        }
        if (std::abs(term) > T(1e10)) {
            break;
        }
    }

    return std::exp(log_prefix) * sum;
}

} // namespace detail

// Main function: parabolic_cylinder_u(a, z)
template <typename T>
T parabolic_cylinder_u(T a, T z) {
    if (std::isnan(a) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T abs_z = std::abs(z);
    if (abs_z < T(10)) {
        return detail::parabolic_cylinder_u_taylor(a, z);
    } else {
        return detail::parabolic_cylinder_u_asymptotic(a, z);
    }
}

// Complex version
template <typename T>
c10::complex<T> parabolic_cylinder_u(c10::complex<T> a, c10::complex<T> z) {
    T abs_z = std::abs(z);
    if (abs_z < T(10)) {
        return detail::parabolic_cylinder_u_taylor(a, z);
    } else {
        return detail::parabolic_cylinder_u_asymptotic(a, z);
    }
}

} // namespace torchscience::kernel::special_functions
