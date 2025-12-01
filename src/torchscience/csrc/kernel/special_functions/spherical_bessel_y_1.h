#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T spherical_bessel_y_1_eps();

template <>
constexpr float spherical_bessel_y_1_eps<float>() { return 1e-6f; }

template <>
constexpr double spherical_bessel_y_1_eps<double>() { return 1e-12; }

template <>
inline c10::Half spherical_bessel_y_1_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 spherical_bessel_y_1_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

// Spherical Bessel function of the second kind, order 1
// y_1(z) = -cos(z)/z^2 - sin(z)/z
template <typename T>
T spherical_bessel_y_1(T z) {
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // y_1 is singular at z=0 (no Taylor series possible)
    if (z == T(0)) {
        return -std::numeric_limits<T>::infinity();
    }

    T z_inv = T(1) / z;
    return -std::cos(z) * z_inv * z_inv - std::sin(z) * z_inv;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_y_1(c10::complex<T> z) {
    // For complex z near zero, the function has a singularity
    // but we can still compute it for non-zero complex values
    c10::complex<T> z_inv = c10::complex<T>(T(1), T(0)) / z;
    return -std::cos(z) * z_inv * z_inv - std::sin(z) * z_inv;
}

} // namespace torchscience::kernel::special_functions
