#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "gamma.h"
#include "log_gamma.h"
#include "parabolic_cylinder_u.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Taylor series for V(a, z) around z=0
// DLMF 12.2.18: V(a,z) = Gamma(1/2+a)/pi * [sin(pi*a)*U(a,z) + U(a,-z)]
// Note: This formula has removable singularities at a = -n-1/2 for n >= 0
// where Gamma(1/2+a) has poles. We handle these by perturbation.
template <typename T>
T parabolic_cylinder_v_taylor(T a, T z) {
    const T pi = static_cast<T>(M_PI);

    T half_plus_a = T(0.5) + a;

    // Check if we're near a pole of Γ(1/2+a): occurs when 1/2+a is a non-positive integer
    T nearest_int = std::round(half_plus_a);
    bool near_pole = (half_plus_a <= T(0.5)) &&
                     (std::abs(half_plus_a - nearest_int) < T(1e-10));

    if (near_pole) {
        // Near a pole: use finite difference to avoid the singularity
        // The formula has a removable singularity, so we average nearby values
        T delta = T(1e-6);

        // Compute at a + delta
        T gamma_p = gamma(T(0.5) + a + delta);
        T sin_p = std::sin(pi * (a + delta));
        T u_p_z = parabolic_cylinder_u_taylor(a + delta, z);
        T u_p_neg_z = parabolic_cylinder_u_taylor(a + delta, -z);
        T v_plus = gamma_p / pi * (sin_p * u_p_z + u_p_neg_z);

        // Compute at a - delta
        T gamma_m = gamma(T(0.5) + a - delta);
        T sin_m = std::sin(pi * (a - delta));
        T u_m_z = parabolic_cylinder_u_taylor(a - delta, z);
        T u_m_neg_z = parabolic_cylinder_u_taylor(a - delta, -z);
        T v_minus = gamma_m / pi * (sin_m * u_m_z + u_m_neg_z);

        return (v_plus + v_minus) / T(2);
    }

    // Standard formula
    T gamma_half_plus_a = gamma(half_plus_a);
    T sin_pi_a = std::sin(pi * a);

    T u_a_z = parabolic_cylinder_u_taylor(a, z);
    T u_a_neg_z = parabolic_cylinder_u_taylor(a, -z);

    return gamma_half_plus_a / pi * (sin_pi_a * u_a_z + u_a_neg_z);
}

// Asymptotic expansion for V(a, z) for large |z|
// DLMF 12.9.2
template <typename T>
T parabolic_cylinder_v_asymptotic(T a, T z) {
    const T eps = pcf_eps<T>();
    const int max_terms = 50;
    const T pi = static_cast<T>(M_PI);

    T z2 = z * z;
    T log_prefix = z2 / T(4) + (a - T(0.5)) * std::log(std::abs(z));

    // Asymptotic series similar to U but with opposite sign in exponent
    T sum = T(1);
    T term = T(1);
    T inv_2z2 = T(1) / (T(2) * z2);

    for (int s = 1; s < max_terms; ++s) {
        T factor = (T(0.5) - a + T(2*s - 2)) * (T(0.5) - a + T(2*s - 1)) / T(s);
        term *= factor * inv_2z2;
        sum += term;

        if (std::abs(term) < eps * std::abs(sum)) {
            break;
        }
        if (std::abs(term) > T(1e10)) {
            break;
        }
    }

    return std::sqrt(T(2) / pi) * std::exp(log_prefix) * sum;
}

// Complex Taylor series for V(a, z)
// Same pole handling as real version for a near -n-1/2
template <typename T>
c10::complex<T> parabolic_cylinder_v_taylor(c10::complex<T> a, c10::complex<T> z) {
    const T pi = static_cast<T>(M_PI);
    const c10::complex<T> pi_c(pi, T(0));
    const c10::complex<T> half(T(0.5), T(0));

    c10::complex<T> half_plus_a = half + a;

    // Check if we're near a pole (real a near -n-1/2)
    bool near_pole = false;
    if (std::abs(half_plus_a.imag()) < T(1e-10)) {
        T real_part = half_plus_a.real();
        T nearest_int = std::round(real_part);
        near_pole = (real_part <= T(0.5)) &&
                    (std::abs(real_part - nearest_int) < T(1e-10));
    }

    if (near_pole) {
        c10::complex<T> delta(T(1e-6), T(0));

        c10::complex<T> gamma_p = gamma(half + a + delta);
        c10::complex<T> sin_p = std::sin(pi_c * (a + delta));
        c10::complex<T> u_p_z = parabolic_cylinder_u_taylor(a + delta, z);
        c10::complex<T> u_p_neg_z = parabolic_cylinder_u_taylor(a + delta, -z);
        c10::complex<T> v_plus = gamma_p / pi_c * (sin_p * u_p_z + u_p_neg_z);

        c10::complex<T> gamma_m = gamma(half + a - delta);
        c10::complex<T> sin_m = std::sin(pi_c * (a - delta));
        c10::complex<T> u_m_z = parabolic_cylinder_u_taylor(a - delta, z);
        c10::complex<T> u_m_neg_z = parabolic_cylinder_u_taylor(a - delta, -z);
        c10::complex<T> v_minus = gamma_m / pi_c * (sin_m * u_m_z + u_m_neg_z);

        return (v_plus + v_minus) / c10::complex<T>(T(2), T(0));
    }

    c10::complex<T> gamma_half_plus_a = gamma(half_plus_a);
    c10::complex<T> sin_pi_a = std::sin(pi_c * a);

    c10::complex<T> u_a_z = parabolic_cylinder_u_taylor(a, z);
    c10::complex<T> u_a_neg_z = parabolic_cylinder_u_taylor(a, -z);

    return gamma_half_plus_a / pi_c * (sin_pi_a * u_a_z + u_a_neg_z);
}

// Complex asymptotic expansion for V(a, z)
template <typename T>
c10::complex<T> parabolic_cylinder_v_asymptotic(c10::complex<T> a, c10::complex<T> z) {
    const T eps = pcf_eps<T>();
    const int max_terms = 50;
    const T pi = static_cast<T>(M_PI);
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> half(T(0.5), T(0));
    const c10::complex<T> four(T(4), T(0));

    c10::complex<T> z2 = z * z;
    c10::complex<T> log_prefix = z2 / four + (a - half) * std::log(z);

    c10::complex<T> sum = one;
    c10::complex<T> term = one;
    c10::complex<T> inv_2z2 = one / (two * z2);

    for (int s = 1; s < max_terms; ++s) {
        c10::complex<T> s_c(T(s), T(0));
        c10::complex<T> factor = (half - a + c10::complex<T>(T(2*s - 2), T(0))) *
                                  (half - a + c10::complex<T>(T(2*s - 1), T(0))) / s_c;
        term *= factor * inv_2z2;
        sum += term;

        if (std::abs(term) < eps * std::abs(sum)) {
            break;
        }
        if (std::abs(term) > T(1e10)) {
            break;
        }
    }

    c10::complex<T> sqrt_2_pi(std::sqrt(T(2) / pi), T(0));
    return sqrt_2_pi * std::exp(log_prefix) * sum;
}

} // namespace detail

// Main function: parabolic_cylinder_v(a, z)
template <typename T>
T parabolic_cylinder_v(T a, T z) {
    if (std::isnan(a) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T abs_z = std::abs(z);
    if (abs_z < T(10)) {
        return detail::parabolic_cylinder_v_taylor(a, z);
    } else {
        return detail::parabolic_cylinder_v_asymptotic(a, z);
    }
}

// Complex version
template <typename T>
c10::complex<T> parabolic_cylinder_v(c10::complex<T> a, c10::complex<T> z) {
    T abs_z = std::abs(z);
    if (abs_z < T(10)) {
        return detail::parabolic_cylinder_v_taylor(a, z);
    } else {
        return detail::parabolic_cylinder_v_asymptotic(a, z);
    }
}

} // namespace torchscience::kernel::special_functions
