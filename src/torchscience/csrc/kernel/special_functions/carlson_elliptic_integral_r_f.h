#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T carlson_rf_tolerance() {
    return T(1e-10);
}

template <>
inline float carlson_rf_tolerance<float>() { return 1e-5f; }

template <>
inline double carlson_rf_tolerance<double>() { return 1e-14; }

} // namespace detail

template <typename T>
T carlson_elliptic_integral_r_f(T x, T y, T z) {
    constexpr int max_iterations = 100;
    const T tolerance = detail::carlson_rf_tolerance<T>();

    for (int i = 0; i < max_iterations; ++i) {
        T sqrt_x = std::sqrt(x);
        T sqrt_y = std::sqrt(y);
        T sqrt_z = std::sqrt(z);

        T lambda = sqrt_x * sqrt_y + sqrt_y * sqrt_z + sqrt_z * sqrt_x;

        x = (x + lambda) / T(4);
        y = (y + lambda) / T(4);
        z = (z + lambda) / T(4);

        T mu = (x + y + z) / T(3);
        T max_dev = std::max({static_cast<T>(std::abs(x - mu)), static_cast<T>(std::abs(y - mu)), static_cast<T>(std::abs(z - mu))}) / mu;

        if (max_dev < tolerance) {
            T X = T(1) - x / mu;
            T Y = T(1) - y / mu;
            T Z = -(X + Y);

            T E2 = X * Y - Z * Z;
            T E3 = X * Y * Z;

            return (T(1) - E2 / T(10) + E3 / T(14) + E2 * E2 / T(24)
                    - T(3) * E2 * E3 / T(44)) / std::sqrt(mu);
        }
    }

    T mu = (x + y + z) / T(3);
    return T(1) / std::sqrt(mu);
}

template <typename T>
c10::complex<T> carlson_elliptic_integral_r_f(
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    constexpr int max_iterations = 100;
    const T tolerance = detail::carlson_rf_tolerance<T>();

    for (int i = 0; i < max_iterations; ++i) {
        c10::complex<T> sqrt_x = std::sqrt(x);
        c10::complex<T> sqrt_y = std::sqrt(y);
        c10::complex<T> sqrt_z = std::sqrt(z);

        c10::complex<T> lambda = sqrt_x * sqrt_y + sqrt_y * sqrt_z + sqrt_z * sqrt_x;

        x = (x + lambda) / T(4);
        y = (y + lambda) / T(4);
        z = (z + lambda) / T(4);

        c10::complex<T> mu = (x + y + z) / T(3);
        T max_dev = std::max({static_cast<T>(std::abs(x - mu)), static_cast<T>(std::abs(y - mu)), static_cast<T>(std::abs(z - mu))}) / static_cast<T>(std::abs(mu));

        if (max_dev < tolerance) {
            c10::complex<T> X = c10::complex<T>(T(1), T(0)) - x / mu;
            c10::complex<T> Y = c10::complex<T>(T(1), T(0)) - y / mu;
            c10::complex<T> Z = -(X + Y);

            c10::complex<T> E2 = X * Y - Z * Z;
            c10::complex<T> E3 = X * Y * Z;

            return (c10::complex<T>(T(1), T(0)) - E2 / T(10) + E3 / T(14)
                    + E2 * E2 / T(24) - c10::complex<T>(T(3), T(0)) * E2 * E3 / T(44))
                   / std::sqrt(mu);
        }
    }

    c10::complex<T> mu = (x + y + z) / T(3);
    return c10::complex<T>(T(1), T(0)) / std::sqrt(mu);
}

} // namespace torchscience::kernel::special_functions
