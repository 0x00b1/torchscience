#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T spherical_bessel_k_0_eps();

template <>
constexpr float spherical_bessel_k_0_eps<float>() { return 1e-6f; }

template <>
constexpr double spherical_bessel_k_0_eps<double>() { return 1e-12; }

template <>
inline c10::Half spherical_bessel_k_0_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 spherical_bessel_k_0_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

// Modified spherical Bessel function of the second kind, order 0
// k_0(z) = (pi/2z) * e^(-z) = (pi/2) * e^(-z) / z
template <typename T>
T spherical_bessel_k_0(T z) {
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T pi_over_2 = T(1.5707963267948966192313216916398);  // pi/2

    // For very small z, k_0(z) -> infinity (pole at origin)
    const T eps = detail::spherical_bessel_k_0_eps<T>();
    if (std::abs(z) < eps) {
        // k_0(z) = (pi/2z) * e^(-z) ≈ (pi/2z) * (1 - z + z^2/2 - ...) for small z
        // The leading term dominates: k_0(z) ≈ pi/(2z)
        return pi_over_2 / z;
    }

    return pi_over_2 * std::exp(-z) / z;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_k_0(c10::complex<T> z) {
    const T pi_over_2 = T(1.5707963267948966192313216916398);

    const T eps = detail::spherical_bessel_k_0_eps<T>();

    if (std::abs(z) < eps) {
        return c10::complex<T>(pi_over_2, T(0)) / z;
    }

    return c10::complex<T>(pi_over_2, T(0)) * std::exp(-z) / z;
}

} // namespace torchscience::kernel::special_functions
