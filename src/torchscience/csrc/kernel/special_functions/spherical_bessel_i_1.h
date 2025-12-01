#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T spherical_bessel_i_1_eps();

template <>
constexpr float spherical_bessel_i_1_eps<float>() { return 1e-6f; }

template <>
constexpr double spherical_bessel_i_1_eps<double>() { return 1e-12; }

template <>
inline c10::Half spherical_bessel_i_1_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 spherical_bessel_i_1_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

// Modified spherical Bessel function of the first kind, order 1
// i_1(z) = cosh(z)/z - sinh(z)/z^2 = -sinh(z)/z^2 + cosh(z)/z
template <typename T>
T spherical_bessel_i_1(T z) {
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T eps = detail::spherical_bessel_i_1_eps<T>();

    // For small z, use Taylor series: i_1(z) = z/3 + z^3/30 + z^5/840 + ...
    if (std::abs(z) < eps) {
        T z2 = z * z;
        return z / T(3) + z * z2 / T(30) + z * z2 * z2 / T(840);
    }

    T z_inv = T(1) / z;
    return std::cosh(z) * z_inv - std::sinh(z) * z_inv * z_inv;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_i_1(c10::complex<T> z) {
    const T eps = detail::spherical_bessel_i_1_eps<T>();

    if (std::abs(z) < eps) {
        c10::complex<T> z2 = z * z;
        return z / c10::complex<T>(T(3), T(0))
               + z * z2 / c10::complex<T>(T(30), T(0))
               + z * z2 * z2 / c10::complex<T>(T(840), T(0));
    }

    c10::complex<T> z_inv = c10::complex<T>(T(1), T(0)) / z;
    return std::cosh(z) * z_inv - std::sinh(z) * z_inv * z_inv;
}

} // namespace torchscience::kernel::special_functions
