#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "spherical_bessel_j.h"
#include "spherical_bessel_y.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Tolerance constants for spherical_hankel_1
template <typename T>
constexpr T spherical_hankel_1_eps();

template <>
constexpr float spherical_hankel_1_eps<float>() { return 1e-6f; }

template <>
constexpr double spherical_hankel_1_eps<double>() { return 1e-12; }

template <>
inline c10::Half spherical_hankel_1_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 spherical_hankel_1_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

// Spherical Hankel function of the first kind of general order n
// h_n^(1)(z) = j_n(z) + i * y_n(z)
// where j_n is the spherical Bessel function of the first kind
// and y_n is the spherical Bessel function of the second kind
//
// This function ALWAYS returns complex values even for real inputs,
// since it contains an imaginary component by definition.
//
// For real z > 0:
// - The real part is j_n(z)
// - The imaginary part is y_n(z)
//
// Key properties:
// - h_n^(1)(z) represents outgoing spherical waves
// - Singularity at z = 0
// - For large |z|: h_n^(1)(z) ~ (-i)^{n+1} * e^{iz} / z

// Complex n, complex z -> complex result
// This is the primary implementation
template <typename T>
c10::complex<T> spherical_hankel_1(c10::complex<T> n, c10::complex<T> z) {
    const T eps = detail::spherical_hankel_1_eps<T>();
    const c10::complex<T> i_unit(T(0), T(1));

    // Handle NaN inputs
    if (std::isnan(n.real()) || std::isnan(n.imag()) ||
        std::isnan(z.real()) || std::isnan(z.imag())) {
        return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                               std::numeric_limits<T>::quiet_NaN());
    }

    // z = 0 is a singularity
    if (std::abs(z) < eps) {
        return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                               std::numeric_limits<T>::quiet_NaN());
    }

    // h_n^(1)(z) = j_n(z) + i * y_n(z)
    c10::complex<T> j_n = spherical_bessel_j(n, z);
    c10::complex<T> y_n = spherical_bessel_y(n, z);

    return j_n + i_unit * y_n;
}

} // namespace torchscience::kernel::special_functions
