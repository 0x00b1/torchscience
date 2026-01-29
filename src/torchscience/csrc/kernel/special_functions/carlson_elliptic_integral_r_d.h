#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T carlson_rd_tolerance() {
    return T(1e-10);
}

template <>
inline float carlson_rd_tolerance<float>() { return 1e-5f; }

template <>
inline double carlson_rd_tolerance<double>() { return 1e-14; }

} // namespace detail

template <typename T>
T carlson_elliptic_integral_r_d(T x, T y, T z) {
    constexpr int max_iterations = 100;
    const T tolerance = detail::carlson_rd_tolerance<T>();

    T sum = T(0);
    T power4 = T(1);

    for (int i = 0; i < max_iterations; ++i) {
        T sqrt_x = std::sqrt(x);
        T sqrt_y = std::sqrt(y);
        T sqrt_z = std::sqrt(z);

        T lambda = sqrt_x * sqrt_y + sqrt_y * sqrt_z + sqrt_z * sqrt_x;

        sum += power4 / (sqrt_z * (z + lambda));
        power4 /= T(4);

        x = (x + lambda) / T(4);
        y = (y + lambda) / T(4);
        z = (z + lambda) / T(4);

        T mu = (x + y + T(3) * z) / T(5);
        T max_dev = std::max({static_cast<T>(std::abs(x - mu)), static_cast<T>(std::abs(y - mu)), static_cast<T>(std::abs(z - mu))}) / mu;

        if (max_dev < tolerance) {
            T X = (mu - x) / mu;
            T Y = (mu - y) / mu;
            T Z = -(X + Y) / T(3);

            T E2 = X * Y - T(6) * Z * Z;
            T E3 = (T(3) * X * Y - T(8) * Z * Z) * Z;
            T E4 = T(3) * (X * Y - Z * Z) * Z * Z;
            T E5 = X * Y * Z * Z * Z;

            T result = (T(1) - T(3) * E2 / T(14) + E3 / T(6) + T(9) * E2 * E2 / T(88)
                       - T(3) * E4 / T(22) - T(9) * E2 * E3 / T(52) + T(3) * E5 / T(26))
                      / (mu * std::sqrt(mu));

            return T(3) * sum + power4 * result;
        }
    }

    T mu = (x + y + T(3) * z) / T(5);
    return T(3) * sum + power4 / (mu * std::sqrt(mu));
}

template <typename T>
c10::complex<T> carlson_elliptic_integral_r_d(
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    constexpr int max_iterations = 100;
    const T tolerance = detail::carlson_rd_tolerance<T>();

    c10::complex<T> sum(T(0), T(0));
    c10::complex<T> power4(T(1), T(0));

    for (int i = 0; i < max_iterations; ++i) {
        c10::complex<T> sqrt_x = std::sqrt(x);
        c10::complex<T> sqrt_y = std::sqrt(y);
        c10::complex<T> sqrt_z = std::sqrt(z);

        c10::complex<T> lambda = sqrt_x * sqrt_y + sqrt_y * sqrt_z + sqrt_z * sqrt_x;

        sum += power4 / (sqrt_z * (z + lambda));
        power4 /= T(4);

        x = (x + lambda) / T(4);
        y = (y + lambda) / T(4);
        z = (z + lambda) / T(4);

        c10::complex<T> mu = (x + y + c10::complex<T>(T(3), T(0)) * z) / T(5);
        T max_dev = std::max({static_cast<T>(std::abs(x - mu)), static_cast<T>(std::abs(y - mu)), static_cast<T>(std::abs(z - mu))}) / static_cast<T>(std::abs(mu));

        if (max_dev < tolerance) {
            c10::complex<T> X = (mu - x) / mu;
            c10::complex<T> Y = (mu - y) / mu;
            c10::complex<T> Z = -(X + Y) / T(3);

            c10::complex<T> E2 = X * Y - c10::complex<T>(T(6), T(0)) * Z * Z;
            c10::complex<T> E3 = (c10::complex<T>(T(3), T(0)) * X * Y
                                  - c10::complex<T>(T(8), T(0)) * Z * Z) * Z;
            c10::complex<T> E4 = c10::complex<T>(T(3), T(0)) * (X * Y - Z * Z) * Z * Z;
            c10::complex<T> E5 = X * Y * Z * Z * Z;

            c10::complex<T> result = (c10::complex<T>(T(1), T(0))
                                      - c10::complex<T>(T(3), T(0)) * E2 / T(14)
                                      + E3 / T(6)
                                      + c10::complex<T>(T(9), T(0)) * E2 * E2 / T(88)
                                      - c10::complex<T>(T(3), T(0)) * E4 / T(22)
                                      - c10::complex<T>(T(9), T(0)) * E2 * E3 / T(52)
                                      + c10::complex<T>(T(3), T(0)) * E5 / T(26))
                                     / (mu * std::sqrt(mu));

            return c10::complex<T>(T(3), T(0)) * sum + power4 * result;
        }
    }

    c10::complex<T> mu = (x + y + c10::complex<T>(T(3), T(0)) * z) / T(5);
    return c10::complex<T>(T(3), T(0)) * sum + power4 / (mu * std::sqrt(mu));
}

} // namespace torchscience::kernel::special_functions
