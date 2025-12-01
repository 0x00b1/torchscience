#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T spherical_bessel_y_0_eps();

template <>
constexpr float spherical_bessel_y_0_eps<float>() { return 1e-6f; }

template <>
constexpr double spherical_bessel_y_0_eps<double>() { return 1e-12; }

template <>
inline c10::Half spherical_bessel_y_0_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 spherical_bessel_y_0_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

// Spherical Bessel function of the second kind, order 0
// y_0(z) = -cos(z) / z
template <typename T>
T spherical_bessel_y_0(T z) {
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // y_0 is singular at z=0, return -infinity
    if (z == T(0)) {
        return -std::numeric_limits<T>::infinity();
    }

    return -std::cos(z) / z;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_y_0(c10::complex<T> z) {
    // For complex zero, return negative infinity (real part)
    if (z == c10::complex<T>(T(0), T(0))) {
        return c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    }

    return -std::cos(z) / z;
}

} // namespace torchscience::kernel::special_functions
