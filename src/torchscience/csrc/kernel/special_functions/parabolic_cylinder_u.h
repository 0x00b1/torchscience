#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "gamma.h"

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

// Taylor series for U(a, z) using confluent hypergeometric 1F1 representation
// From mpmath documentation:
// e^{-z^2/4} U(a,z) = U(a,0) * 1F1(-a/2+1/4; 1/2; -z^2/2) + U'(a,0) * z * 1F1(-a/2+3/4; 3/2; -z^2/2)
// Therefore: U(a,z) = e^{z^2/4} * [U(a,0) * 1F1(...) + U'(a,0) * z * 1F1(...)]
//
// This representation is more numerically stable because:
// 1. The 1F1 series has argument -z^2/2 (alternating signs) rather than z^2 (all positive)
// 2. The exponential factor is e^{z^2/4} (grows with z) which compensates for the decaying 1F1
template <typename T>
T parabolic_cylinder_u_taylor(T a, T z) {
    const T eps = pcf_eps<T>();
    const int max_iter = pcf_max_iter<T>();
    const T pi = static_cast<T>(M_PI);
    const T sqrt_pi = std::sqrt(pi);

    // Compute initial values from DLMF 12.4.5-12.4.6
    // U(a,0) = sqrt(π) * 2^(-1/4 - a/2) / Γ(3/4 + a/2)
    // U'(a,0) = -sqrt(π) * 2^(1/4 - a/2) / Γ(1/4 + a/2)
    // Use gamma() directly to get correct sign for negative arguments
    T gamma_arg1 = T(0.75) + a / T(2);
    T gamma_arg2 = T(0.25) + a / T(2);

    T gamma_val1 = gamma(gamma_arg1);
    T gamma_val2 = gamma(gamma_arg2);

    T U0 = sqrt_pi * std::pow(T(2), -T(0.25) - a / T(2)) / gamma_val1;
    T U0_prime = -sqrt_pi * std::pow(T(2), T(0.25) - a / T(2)) / gamma_val2;

    // Handle gamma poles (set coefficient to 0 when gamma is infinite)
    if (!std::isfinite(U0)) U0 = T(0);
    if (!std::isfinite(U0_prime)) U0_prime = T(0);

    // Special case: z = 0
    // At z=0: 1F1(a;b;0) = 1, so U(a,0) = U0 * 1 + U0_prime * 0 * 1 = U0
    if (z == T(0)) {
        return U0;
    }

    T z2 = z * z;
    T w = -z2 / T(2);  // Argument for 1F1

    // Compute 1F1(-a/2 + 1/4; 1/2; w)
    // 1F1(alpha; beta; w) = sum_{n=0}^{inf} (alpha)_n / (beta)_n * w^n / n!
    // Term ratio: term_{n+1}/term_n = (alpha + n) * w / ((beta + n) * (n + 1))
    T alpha1 = -a / T(2) + T(0.25);
    T beta1 = T(0.5);
    T f1 = T(1);
    T term1 = T(1);

    for (int n = 0; n < max_iter; ++n) {
        T ratio = (alpha1 + T(n)) * w / ((beta1 + T(n)) * T(n + 1));
        term1 *= ratio;
        f1 += term1;
        if (std::abs(term1) < eps * std::abs(f1)) {
            break;
        }
        if (!std::isfinite(term1)) {
            break;
        }
    }

    // Compute 1F1(-a/2 + 3/4; 3/2; w)
    T alpha2 = -a / T(2) + T(0.75);
    T beta2 = T(1.5);
    T f2 = T(1);
    T term2 = T(1);

    for (int n = 0; n < max_iter; ++n) {
        T ratio = (alpha2 + T(n)) * w / ((beta2 + T(n)) * T(n + 1));
        term2 *= ratio;
        f2 += term2;
        if (std::abs(term2) < eps * std::abs(f2)) {
            break;
        }
        if (!std::isfinite(term2)) {
            break;
        }
    }

    // U(a,z) = e^{z^2/4} * [U(a,0) * f1 + U'(a,0) * z * f2]
    // Note: we DON'T multiply by exp(z^2/4) yet. The formula from mpmath is:
    // e^{-z^2/4} U(a,z) = U(a,0) * f1 + U'(a,0) * z * f2
    // So: U(a,z) = e^{z^2/4} * [U(a,0) * f1 + U'(a,0) * z * f2]
    T exp_factor = std::exp(z2 / T(4));

    return exp_factor * (U0 * f1 + U0_prime * z * f2);
}

// Asymptotic expansion for U(a, z) for large |z|
// Uses optimal truncation: stop when terms start growing (asymptotic divergence)
template <typename T>
T parabolic_cylinder_u_asymptotic(T a, T z) {
    const T eps = pcf_eps<T>();
    const int max_terms = 100;

    T z2 = z * z;
    T log_prefix = -z2 / T(4) + (-a - T(0.5)) * std::log(std::abs(z));

    T sum = T(1);
    T best_sum = sum;
    T term = T(1);
    T prev_abs_term = std::abs(term);
    T neg_2z2_inv = T(-1) / (T(2) * z2);

    for (int s = 1; s < max_terms; ++s) {
        T factor = (T(0.5) + a + T(2*s - 2)) * (T(0.5) + a + T(2*s - 1)) / T(s);
        term *= factor * neg_2z2_inv;
        sum += term;

        T abs_term = std::abs(term);

        // Optimal truncation: if terms start growing, stop and use best sum
        if (abs_term > prev_abs_term) {
            break;
        }
        best_sum = sum;
        prev_abs_term = abs_term;

        if (abs_term < eps * std::abs(sum)) {
            break;
        }
    }

    return std::exp(log_prefix) * best_sum;
}

// Complex Taylor series for U(a, z) using confluent hypergeometric 1F1 representation
// e^{-z^2/4} U(a,z) = U(a,0) * 1F1(-a/2+1/4; 1/2; -z^2/2) + U'(a,0) * z * 1F1(-a/2+3/4; 3/2; -z^2/2)
// Therefore: U(a,z) = e^{z^2/4} * [U(a,0) * 1F1(...) + U'(a,0) * z * 1F1(...)]
template <typename T>
c10::complex<T> parabolic_cylinder_u_taylor(c10::complex<T> a, c10::complex<T> z) {
    const T eps = pcf_eps<T>();
    const int max_iter = pcf_max_iter<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> four(T(4), T(0));
    const c10::complex<T> half(T(0.5), T(0));
    const c10::complex<T> three_half(T(1.5), T(0));
    const c10::complex<T> quarter(T(0.25), T(0));
    const c10::complex<T> three_quarter(T(0.75), T(0));
    const T pi = static_cast<T>(M_PI);
    const T sqrt_pi = std::sqrt(pi);
    const c10::complex<T> sqrt_pi_c(sqrt_pi, T(0));

    // Compute initial values from DLMF 12.4.5-12.4.6
    // U(a,0) = sqrt(π) * 2^(-1/4 - a/2) / Γ(3/4 + a/2)
    // U'(a,0) = -sqrt(π) * 2^(1/4 - a/2) / Γ(1/4 + a/2)
    c10::complex<T> gamma_arg1 = three_quarter + a / two;
    c10::complex<T> gamma_arg2 = quarter + a / two;

    c10::complex<T> gamma_val1 = gamma(gamma_arg1);
    c10::complex<T> gamma_val2 = gamma(gamma_arg2);

    c10::complex<T> U0 = sqrt_pi_c * std::pow(two, -quarter - a / two) / gamma_val1;
    c10::complex<T> U0_prime = -sqrt_pi_c * std::pow(two, quarter - a / two) / gamma_val2;

    // Handle gamma poles (set coefficient to 0 when gamma is infinite)
    if (!std::isfinite(std::abs(U0))) U0 = c10::complex<T>(T(0), T(0));
    if (!std::isfinite(std::abs(U0_prime))) U0_prime = c10::complex<T>(T(0), T(0));

    // Special case: z = 0
    if (std::abs(z) == T(0)) {
        return U0;
    }

    c10::complex<T> z2 = z * z;
    c10::complex<T> w = -z2 / two;  // Argument for 1F1

    // Compute 1F1(-a/2 + 1/4; 1/2; w)
    c10::complex<T> alpha1 = -a / two + quarter;
    c10::complex<T> beta1 = half;
    c10::complex<T> f1 = one;
    c10::complex<T> term1 = one;

    for (int n = 0; n < max_iter; ++n) {
        c10::complex<T> n_c(T(n), T(0));
        c10::complex<T> ratio = (alpha1 + n_c) * w / ((beta1 + n_c) * (n_c + one));
        term1 *= ratio;
        f1 += term1;
        if (std::abs(term1) < eps * std::abs(f1)) {
            break;
        }
        if (!std::isfinite(std::abs(term1))) {
            break;
        }
    }

    // Compute 1F1(-a/2 + 3/4; 3/2; w)
    c10::complex<T> alpha2 = -a / two + three_quarter;
    c10::complex<T> beta2 = three_half;
    c10::complex<T> f2 = one;
    c10::complex<T> term2 = one;

    for (int n = 0; n < max_iter; ++n) {
        c10::complex<T> n_c(T(n), T(0));
        c10::complex<T> ratio = (alpha2 + n_c) * w / ((beta2 + n_c) * (n_c + one));
        term2 *= ratio;
        f2 += term2;
        if (std::abs(term2) < eps * std::abs(f2)) {
            break;
        }
        if (!std::isfinite(std::abs(term2))) {
            break;
        }
    }

    // U(a,z) = e^{z^2/4} * [U(a,0) * f1 + U'(a,0) * z * f2]
    c10::complex<T> exp_factor = std::exp(z2 / four);

    return exp_factor * (U0 * f1 + U0_prime * z * f2);
}

// Complex asymptotic expansion for U(a, z)
// Uses optimal truncation: stop when terms start growing (asymptotic divergence)
template <typename T>
c10::complex<T> parabolic_cylinder_u_asymptotic(c10::complex<T> a, c10::complex<T> z) {
    const T eps = pcf_eps<T>();
    const int max_terms = 100;
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> half(T(0.5), T(0));
    const c10::complex<T> four(T(4), T(0));

    c10::complex<T> z2 = z * z;
    c10::complex<T> log_prefix = -z2 / four + (-a - half) * std::log(z);

    c10::complex<T> sum = one;
    c10::complex<T> best_sum = sum;
    c10::complex<T> term = one;
    T prev_abs_term = std::abs(term);
    c10::complex<T> neg_2z2_inv = -one / (two * z2);

    for (int s = 1; s < max_terms; ++s) {
        c10::complex<T> s_c(T(s), T(0));
        c10::complex<T> factor = (half + a + c10::complex<T>(T(2*s - 2), T(0))) *
                                  (half + a + c10::complex<T>(T(2*s - 1), T(0))) / s_c;
        term *= factor * neg_2z2_inv;
        sum += term;

        T abs_term = std::abs(term);

        // Optimal truncation: if terms start growing, stop and use best sum
        if (abs_term > prev_abs_term) {
            break;
        }
        best_sum = sum;
        prev_abs_term = abs_term;

        if (abs_term < eps * std::abs(sum)) {
            break;
        }
    }

    return std::exp(log_prefix) * best_sum;
}

} // namespace detail

// Main function: parabolic_cylinder_u(a, z)
template <typename T>
T parabolic_cylinder_u(T a, T z) {
    if (std::isnan(a) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T abs_z = std::abs(z);
    // Use Taylor series for small z, asymptotic expansion for large z
    // The threshold is chosen to balance accuracy between the two methods
    if (abs_z < T(5)) {
        return detail::parabolic_cylinder_u_taylor(a, z);
    } else {
        return detail::parabolic_cylinder_u_asymptotic(a, z);
    }
}

// Complex version
template <typename T>
c10::complex<T> parabolic_cylinder_u(c10::complex<T> a, c10::complex<T> z) {
    T abs_z = std::abs(z);
    // Use Taylor series for small z, asymptotic expansion for large z
    // The threshold is chosen to balance accuracy between the two methods
    if (abs_z < T(5)) {
        return detail::parabolic_cylinder_u_taylor(a, z);
    } else {
        return detail::parabolic_cylinder_u_asymptotic(a, z);
    }
}

} // namespace torchscience::kernel::special_functions
