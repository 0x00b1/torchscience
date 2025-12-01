#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T spherical_bessel_k_1_eps();

template <>
constexpr float spherical_bessel_k_1_eps<float>() { return 1e-6f; }

template <>
constexpr double spherical_bessel_k_1_eps<double>() { return 1e-12; }

template <>
inline c10::Half spherical_bessel_k_1_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 spherical_bessel_k_1_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

// Modified spherical Bessel function of the second kind, order 1
// k_1(z) = (pi/2z^2)(1+z) e^(-z)
template <typename T>
T spherical_bessel_k_1(T z) {
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T pi = T(3.14159265358979323846);

    // For small z, return infinity (pole at z=0)
    const T eps = detail::spherical_bessel_k_1_eps<T>();
    if (std::abs(z) < eps) {
        return std::numeric_limits<T>::infinity();
    }

    T z_inv = T(1) / z;
    T z2_inv = z_inv * z_inv;
    return (pi / T(2)) * z2_inv * (T(1) + z) * std::exp(-z);
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_k_1(c10::complex<T> z) {
    const T pi = T(3.14159265358979323846);
    const T eps = detail::spherical_bessel_k_1_eps<T>();

    // For small |z|, return infinity (pole at z=0)
    if (std::abs(z) < eps) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    c10::complex<T> z_inv = c10::complex<T>(T(1), T(0)) / z;
    c10::complex<T> z2_inv = z_inv * z_inv;
    c10::complex<T> pi_over_2(pi / T(2), T(0));
    return pi_over_2 * z2_inv * (c10::complex<T>(T(1), T(0)) + z) * std::exp(-z);
}

} // namespace torchscience::kernel::special_functions
