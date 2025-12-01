#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>
#include "rational_polynomial_evaluate.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Cephes coefficients for J₁(z) rational approximation (|z| <= 5)
// Source: Cephes Math Library (Stephen L. Moshier)
// J₁(z) ≈ z * (1/2 + (z/5)² * RP(z²)/RQ(z²))

constexpr double j1_RP[] = {
    -8.99971225705559398224E8,
     4.52228297998194034323E11,
    -7.27494245221818276015E13,
     3.68295732863852883286E15,
};

constexpr double j1_RQ[] = {
    // 1.0 (implicit leading coefficient)
     6.20836478118054335476E2,
     2.56987256757748830383E5,
     8.35146791431949253037E7,
     2.21511595479792499675E10,
     4.74914122079991414898E12,
     7.84369607876235854894E14,
     8.95222336184627338078E16,
     5.32278620332680085395E18,
};

// Cephes coefficients for asymptotic expansion (|z| > 5)
// J₁(z) ≈ sqrt(2/(π*z)) * [P(z)*cos(θ) - Q(z)*sin(θ)]
// where θ = z - 3π/4

constexpr double j1_PP[] = {
     7.62125616208173112003E-4,
     7.31397056940917570436E-2,
     1.12719608129684925192E0,
     5.11207951146807644818E0,
     8.42404590141772420927E0,
     5.21451598682361504063E0,
     1.00000000000000000254E0,
};

constexpr double j1_PQ[] = {
     5.71323128072548699714E-4,
     6.88455908754495404082E-2,
     1.10514232634061696926E0,
     5.07386386128601488557E0,
     8.39985554327604159757E0,
     5.20982848682361821619E0,
     9.99999999999999997461E-1,
};

constexpr double j1_QP[] = {
     5.10862594750176621635E-2,
     4.98213872951233449420E0,
     7.58238284132545283818E1,
     3.66779609360150777800E2,
     7.10856304998926107277E2,
     5.97489612400613639965E2,
     2.11688757100572135698E2,
     2.52070205858023719784E1,
};

constexpr double j1_QQ[] = {
    // 1.0 (implicit leading coefficient)
     7.42373277035675149943E1,
     1.05644886038262816351E3,
     4.98641058337653607651E3,
     9.56231892404756170795E3,
     7.99704160447350683650E3,
     2.82619278517639096600E3,
     3.36093607810698293419E2,
};

// Squares of first two zeros of J₁ (for factoring in small argument approximation)
constexpr double J1_Z1 = 1.46819706421238932572E1;    // (3.83170597...)²
constexpr double J1_Z2 = 4.92184563216946036703E1;    // (7.01558667...)²

constexpr double J1_SQ2OPI = 0.79788456080286535587989;  // sqrt(2/pi)
constexpr double J1_THPIO4 = 2.35619449019234492885;     // 3*pi/4

} // namespace detail

template <typename T>
T bessel_j_1(T z) {
    // Handle special values
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (std::isinf(z)) {
        // J₁(±∞) = 0
        return T(0);
    }

    T x = z;
    T sign = T(1);

    // J₁ is odd: J₁(-z) = -J₁(z)
    if (z < T(0)) {
        x = -z;
        sign = T(-1);
    }

    if (x <= T(5.0)) {
        // Small argument: rational polynomial approximation
        // J₁(z) = x * (z² - Z1) * (z² - Z2) * RP(z²) / RQ(z²)
        // where Z1, Z2 are squares of the first two zeros of J₁
        T z2 = x * x;
        T w = (z2 - T(detail::J1_Z1)) * (z2 - T(detail::J1_Z2));
        w = w * detail::polevl(z2, detail::j1_RP, 3) / detail::p1evl(z2, detail::j1_RQ, 8);
        return sign * x * w;
    } else {
        // Large argument: asymptotic expansion
        // J₁(z) ≈ sqrt(2/(π*z)) * [P(z)*cos(θ) - Q(z)*sin(θ)]
        // where θ = z - 3π/4
        T w = T(5.0) / x;
        T z2 = w * w;
        T p = detail::polevl(z2, detail::j1_PP, 6) / detail::polevl(z2, detail::j1_PQ, 6);
        T q = detail::polevl(z2, detail::j1_QP, 7) / detail::p1evl(z2, detail::j1_QQ, 7);

        T xn = x - T(detail::J1_THPIO4);
        p = p * std::cos(xn) - w * q * std::sin(xn);
        return sign * p * T(detail::J1_SQ2OPI) / std::sqrt(x);
    }
}

// Complex version
// Note: The asymptotic expansion is primarily validated near the real axis.
// For complex z far from the real axis, accuracy should be verified empirically.
// J₁ satisfies the odd function property: J₁(-z) = -J₁(z)
template <typename T>
c10::complex<T> bessel_j_1(c10::complex<T> z) {
    T mag = std::abs(z);

    if (mag <= T(5.0)) {
        // Small argument: multiplication by z preserves odd function property
        // J₁(z) = z * (z² - Z1) * (z² - Z2) * RP(z²) / RQ(z²)
        c10::complex<T> z2 = z * z;
        c10::complex<T> w = (z2 - c10::complex<T>(T(detail::J1_Z1), T(0))) *
                            (z2 - c10::complex<T>(T(detail::J1_Z2), T(0)));
        w = w * detail::polevl(z2, detail::j1_RP, 3) / detail::p1evl(z2, detail::j1_RQ, 8);
        return z * w;
    } else {
        // Large argument: asymptotic expansion
        // J₁(z) ≈ sqrt(2/(π*z)) * [P(z)*cos(θ) - Q(z)*sin(θ)]
        // where θ = z - 3π/4
        c10::complex<T> inv_z = c10::complex<T>(T(5.0), T(0)) / z;
        c10::complex<T> z2 = inv_z * inv_z;
        c10::complex<T> p = detail::polevl(z2, detail::j1_PP, 6) / detail::polevl(z2, detail::j1_PQ, 6);
        c10::complex<T> q = detail::polevl(z2, detail::j1_QP, 7) / detail::p1evl(z2, detail::j1_QQ, 7);

        c10::complex<T> xn = z - c10::complex<T>(T(detail::J1_THPIO4), T(0));
        p = p * std::cos(xn) - inv_z * q * std::sin(xn);
        return p * c10::complex<T>(T(detail::J1_SQ2OPI), T(0)) / std::sqrt(z);
    }
}

} // namespace torchscience::kernel::special_functions
