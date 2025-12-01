#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>
#include "rational_polynomial_evaluate.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Cephes coefficients for J₀(z) rational approximation (|z| <= 5)
// Source: Cephes Math Library (Stephen L. Moshier)
// J₀(z) ≈ (z² - DR1)(z² - DR2) * RP(z²)/RQ(z²)
// where DR1 and DR2 are squares of the first two zeros of J₀

constexpr double j0_RP[] = {
    -4.79443220978201773821E9,
     1.95617491946556577543E12,
    -2.49248344360967716204E14,
     9.70862251047306323952E15,
};

constexpr double j0_RQ[] = {
    // 1.0 (implicit leading coefficient)
     4.99563147152651017219E2,
     1.73785401676374683123E5,
     4.84409658339962045305E7,
     1.11855537045356834862E10,
     2.11277520115489217587E12,
     3.10518229857422583814E14,
     3.18121955943204943306E16,
     1.71086294081043136091E18,
};

// Zeros of J₀ squared (for factoring)
constexpr double J0_DR1 = 5.78318596294678452118E0;    // first zero squared: (2.4048...)²
constexpr double J0_DR2 = 3.04712623436620863991E1;   // second zero squared: (5.5201...)²

// Cephes coefficients for asymptotic expansion (|z| > 5)
// J₀(z) ≈ sqrt(2/(π*z)) * [P(z)*cos(θ) - Q(z)*sin(θ)]
// where θ = z - π/4

constexpr double j0_PP[] = {
     7.96936729297347051624E-4,
     8.28352392107440799803E-2,
     1.23953371646414299388E0,
     5.44725003058768775090E0,
     8.74716500199817011941E0,
     5.30324038235394892183E0,
     9.99999999999999997821E-1,
};

constexpr double j0_PQ[] = {
     9.24408810558863637013E-4,
     8.56288474354474431428E-2,
     1.25352743901058953537E0,
     5.47097740330417105182E0,
     8.76190883237069594232E0,
     5.30605288235394617618E0,
     1.00000000000000000218E0,
};

constexpr double j0_QP[] = {
    -1.13663838898469149931E-2,
    -1.28252718670509318512E0,
    -1.95539544257735972385E1,
    -9.32060152123768231369E1,
    -1.77681167980488050595E2,
    -1.47077505154951170175E2,
    -5.14105326766599330220E1,
    -6.05014350600728481186E0,
};

constexpr double j0_QQ[] = {
    // 1.0 (implicit leading coefficient)
     6.43178256118178023184E1,
     8.56430025976980587198E2,
     3.88240183605401609683E3,
     7.24046774195652478189E3,
     5.93072701187316984827E3,
     2.06209331660327847417E3,
     2.42005740240291393179E2,
};

constexpr double J0_SQ2OPI = 0.79788456080286535587989;  // sqrt(2/pi)
constexpr double J0_PIO4 = 0.78539816339744830961566;    // pi/4

} // namespace detail

template <typename T>
T bessel_j_0(T z) {
    // Handle special values
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (std::isinf(z)) {
        // J₀(±∞) = 0
        return T(0);
    }

    // J₀ is even: J₀(-z) = J₀(z)
    T x = std::abs(z);

    if (x <= T(5.0)) {
        // Small argument: rational polynomial approximation
        // J₀(z) = (z² - DR1)(z² - DR2) * RP(z²)/RQ(z²)
        T z2 = x * x;
        T p = (z2 - T(detail::J0_DR1)) * (z2 - T(detail::J0_DR2));
        p = p * detail::polevl(z2, detail::j0_RP, 3) / detail::p1evl(z2, detail::j0_RQ, 8);
        return p;
    } else {
        // Large argument: asymptotic expansion
        // J₀(z) ≈ sqrt(2/(π*z)) * [P(z)*cos(θ) - Q(z)*sin(θ)]
        // where θ = z - π/4
        T w = T(5.0) / x;
        T z2 = T(25.0) / (x * x);
        T p = detail::polevl(z2, detail::j0_PP, 6) / detail::polevl(z2, detail::j0_PQ, 6);
        T q = detail::polevl(z2, detail::j0_QP, 7) / detail::p1evl(z2, detail::j0_QQ, 7);

        T xn = x - T(detail::J0_PIO4);
        T cosxn = std::cos(xn);
        T sinxn = std::sin(xn);

        p = p * cosxn - w * q * sinxn;

        return p * T(detail::J0_SQ2OPI) / std::sqrt(x);
    }
}

// Complex version
// Note: The asymptotic expansion is primarily validated near the real axis.
// For complex z far from the real axis, accuracy should be verified empirically.
// J₀ satisfies the even function property: J₀(-z) = J₀(z)
template <typename T>
c10::complex<T> bessel_j_0(c10::complex<T> z) {
    T mag = std::abs(z);

    if (mag <= T(5.0)) {
        // Small argument: rational polynomial approximation
        // J₀(z) = (z² - DR1)(z² - DR2) * RP(z²)/RQ(z²)
        c10::complex<T> z2 = z * z;
        c10::complex<T> p = (z2 - c10::complex<T>(T(detail::J0_DR1), T(0))) *
                            (z2 - c10::complex<T>(T(detail::J0_DR2), T(0)));
        p = p * detail::polevl(z2, detail::j0_RP, 3) / detail::p1evl(z2, detail::j0_RQ, 8);
        return p;
    } else {
        // Large argument: asymptotic expansion
        c10::complex<T> inv_z = c10::complex<T>(T(5.0), T(0)) / z;
        c10::complex<T> z2 = c10::complex<T>(T(25.0), T(0)) / (z * z);
        c10::complex<T> p = detail::polevl(z2, detail::j0_PP, 6) / detail::polevl(z2, detail::j0_PQ, 6);
        c10::complex<T> q = detail::polevl(z2, detail::j0_QP, 7) / detail::p1evl(z2, detail::j0_QQ, 7);

        c10::complex<T> xn = z - c10::complex<T>(T(detail::J0_PIO4), T(0));
        c10::complex<T> cosxn = std::cos(xn);
        c10::complex<T> sinxn = std::sin(xn);

        p = p * cosxn - inv_z * q * sinxn;

        return p * c10::complex<T>(T(detail::J0_SQ2OPI), T(0)) / std::sqrt(z);
    }
}

} // namespace torchscience::kernel::special_functions
