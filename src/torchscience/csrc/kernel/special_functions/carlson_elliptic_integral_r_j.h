#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "carlson_elliptic_integral_r_c.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T carlson_rj_tolerance() {
    return T(1e-10);
}

template <>
inline float carlson_rj_tolerance<float>() { return 1e-5f; }

template <>
inline double carlson_rj_tolerance<double>() { return 1e-14; }

} // namespace detail

template <typename T>
T carlson_elliptic_integral_r_j(T x, T y, T z, T p) {
    // Carlson's elliptic integral R_J using duplication algorithm
    // R_J(x,y,z,p) = (3/2) * integral from 0 to infinity of
    //               dt / [(t+p) * sqrt((t+x)(t+y)(t+z))]
    //
    // Algorithm from Carlson (1995) "Numerical computation of real or complex
    // elliptic integrals", Numerical Algorithms 10, 13-26

    constexpr int max_iterations = 100;
    const T tolerance = detail::carlson_rj_tolerance<T>();

    // Store original p for the delta computation
    T p0 = p;
    T delta = (p - x) * (p - y) * (p - z);

    T sum = T(0);
    T power4 = T(1);

    for (int i = 0; i < max_iterations; ++i) {
        T sqrt_x = std::sqrt(x);
        T sqrt_y = std::sqrt(y);
        T sqrt_z = std::sqrt(z);
        T sqrt_p = std::sqrt(p);

        T lambda = sqrt_x * sqrt_y + sqrt_y * sqrt_z + sqrt_z * sqrt_x;

        // Compute d_n and e_n for the R_C correction
        // d_n = (sqrt_p + sqrt_x)(sqrt_p + sqrt_y)(sqrt_p + sqrt_z)
        // e_n = 4^(-3n) * delta / d_n^2
        T d_n = (sqrt_p + sqrt_x) * (sqrt_p + sqrt_y) * (sqrt_p + sqrt_z);
        T e_n = delta / (power4 * power4 * power4 * d_n * d_n);

        // Add R_C correction term
        sum += power4 * carlson_elliptic_integral_r_c(T(1), T(1) + e_n) / d_n;

        power4 /= T(4);

        x = (x + lambda) / T(4);
        y = (y + lambda) / T(4);
        z = (z + lambda) / T(4);
        p = (p + lambda) / T(4);

        // Compute mean for convergence test
        T mu = (x + y + z + T(2) * p) / T(5);

        // Check convergence
        T max_dev = std::max({static_cast<T>(std::abs(x - mu)), static_cast<T>(std::abs(y - mu)),
                             static_cast<T>(std::abs(z - mu)), static_cast<T>(std::abs(p - mu))}) / static_cast<T>(std::abs(mu));

        if (max_dev < tolerance) {
            // Compute the series expansion near convergence
            T X = (mu - x) / mu;
            T Y = (mu - y) / mu;
            T Z = (mu - z) / mu;
            T P = -(X + Y + Z) / T(2);

            T E2 = X * Y + X * Z + Y * Z - T(3) * P * P;
            T E3 = X * Y * Z + T(2) * E2 * P + T(4) * P * P * P;
            T E4 = (T(2) * X * Y * Z + E2 * P + T(3) * P * P * P) * P;
            T E5 = X * Y * Z * P * P;

            T result = (T(1) - T(3) * E2 / T(14) + E3 / T(6)
                       + T(9) * E2 * E2 / T(88) - T(3) * E4 / T(22)
                       - T(9) * E2 * E3 / T(52) + T(3) * E5 / T(26))
                      / (mu * std::sqrt(mu));

            return T(6) * sum + power4 * result;
        }
    }

    // Fallback for non-convergence
    T mu = (x + y + z + T(2) * p) / T(5);
    return T(6) * sum + power4 / (mu * std::sqrt(mu));
}

template <typename T>
c10::complex<T> carlson_elliptic_integral_r_j(
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z,
    c10::complex<T> p
) {
    constexpr int max_iterations = 100;
    const T tolerance = detail::carlson_rj_tolerance<T>();

    // Store original delta
    c10::complex<T> delta = (p - x) * (p - y) * (p - z);

    c10::complex<T> sum(T(0), T(0));
    c10::complex<T> power4(T(1), T(0));
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> three(T(3), T(0));
    c10::complex<T> four(T(4), T(0));
    c10::complex<T> six(T(6), T(0));
    c10::complex<T> nine(T(9), T(0));

    for (int i = 0; i < max_iterations; ++i) {
        c10::complex<T> sqrt_x = std::sqrt(x);
        c10::complex<T> sqrt_y = std::sqrt(y);
        c10::complex<T> sqrt_z = std::sqrt(z);
        c10::complex<T> sqrt_p = std::sqrt(p);

        c10::complex<T> lambda = sqrt_x * sqrt_y + sqrt_y * sqrt_z + sqrt_z * sqrt_x;

        // Compute d_n and e_n for the R_C correction
        c10::complex<T> d_n = (sqrt_p + sqrt_x) * (sqrt_p + sqrt_y) * (sqrt_p + sqrt_z);
        c10::complex<T> e_n = delta / (power4 * power4 * power4 * d_n * d_n);

        // Add R_C correction term
        sum += power4 * carlson_elliptic_integral_r_c(one, one + e_n) / d_n;

        power4 /= T(4);

        x = (x + lambda) / T(4);
        y = (y + lambda) / T(4);
        z = (z + lambda) / T(4);
        p = (p + lambda) / T(4);

        // Compute mean
        c10::complex<T> mu = (x + y + z + two * p) / T(5);

        // Check convergence
        T max_dev = std::max({static_cast<T>(std::abs(x - mu)), static_cast<T>(std::abs(y - mu)),
                             static_cast<T>(std::abs(z - mu)), static_cast<T>(std::abs(p - mu))}) / static_cast<T>(std::abs(mu));

        if (max_dev < tolerance) {
            // Compute the series expansion near convergence
            c10::complex<T> X = (mu - x) / mu;
            c10::complex<T> Y = (mu - y) / mu;
            c10::complex<T> Z = (mu - z) / mu;
            c10::complex<T> P = -(X + Y + Z) / T(2);

            c10::complex<T> E2 = X * Y + X * Z + Y * Z - three * P * P;
            c10::complex<T> E3 = X * Y * Z + two * E2 * P + four * P * P * P;
            c10::complex<T> E4 = (two * X * Y * Z + E2 * P + three * P * P * P) * P;
            c10::complex<T> E5 = X * Y * Z * P * P;

            c10::complex<T> result = (one
                                      - three * E2 / T(14)
                                      + E3 / T(6)
                                      + nine * E2 * E2 / T(88)
                                      - three * E4 / T(22)
                                      - nine * E2 * E3 / T(52)
                                      + three * E5 / T(26))
                                     / (mu * std::sqrt(mu));

            return six * sum + power4 * result;
        }
    }

    // Fallback for non-convergence
    c10::complex<T> mu = (x + y + z + two * p) / T(5);
    return six * sum + power4 / (mu * std::sqrt(mu));
}

} // namespace torchscience::kernel::special_functions
