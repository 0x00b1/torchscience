#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>
#include "rational_polynomial_evaluate.h"
#include "bessel_j_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Cephes coefficients for Y₁(z) rational approximation (|z| <= 5)
// Source: Cephes Math Library (Stephen L. Moshier)
// Y₁(z) ≈ z * YP(z²)/YQ(z²) + (2/π)[J₁(z)ln(z/2) - 1/z]

constexpr double y1_YP[] = {
     1.26320474790178026440E9,
    -6.47355876379160291031E11,
     1.14509511541823727583E14,
    -8.12770255501325109621E15,
     2.02439475713594898196E17,
    -7.78877196265950026825E17,
};

constexpr double y1_YQ[] = {
    // 1.0 (implicit leading coefficient)
     5.94301592346128195359E2,
     2.35564092943068577943E5,
     7.34811944459721705660E7,
     1.87601316108706159478E10,
     3.88231277496238566008E12,
     6.20557727146953693363E14,
     6.87141087355300489866E16,
     3.97270608116560655612E18,
};

// Cephes coefficients for asymptotic expansion (|z| > 5)
// Y₁(z) ≈ sqrt(2/(π*z)) * [P(z)*sin(θ) + Q(z)*cos(θ)]
// where θ = z - 3π/4
// Note: PP, PQ, QP, QQ are same as for J₁

constexpr double y1_PP[] = {
     7.62125616208173112003E-4,
     7.31397056940917570436E-2,
     1.12719608129684925192E0,
     5.11207951146807644818E0,
     8.42404590141772420927E0,
     5.21451598682361504063E0,
     1.00000000000000000254E0,
};

constexpr double y1_PQ[] = {
     5.71323128072548699714E-4,
     6.88455908754495404082E-2,
     1.10514232634061696926E0,
     5.07386386128601488557E0,
     8.39985554327604159757E0,
     5.20982848682361821619E0,
     9.99999999999999997461E-1,
};

constexpr double y1_QP[] = {
     5.10862594750176621635E-2,
     4.98213872951233449420E0,
     7.58238284132545283818E1,
     3.66779609360150777800E2,
     7.10856304998926107277E2,
     5.97489612400613639965E2,
     2.11688757100572135698E2,
     2.52070205858023719784E1,
};

constexpr double y1_QQ[] = {
    // 1.0 (implicit leading coefficient)
     7.42373277035675149943E1,
     1.05644886038262816351E3,
     4.98641058337653607651E3,
     9.56231892404756170795E3,
     7.99704160447350683650E3,
     2.82619278517639096600E3,
     3.36093607810698293419E2,
};

constexpr double Y1_TWOOPI = 0.63661977236758134307554;  // 2/pi
constexpr double Y1_SQ2OPI = 0.79788456080286535587989;  // sqrt(2/pi)
constexpr double Y1_THPIO4 = 2.35619449019234492885;     // 3*pi/4

} // namespace detail

template <typename T>
T bessel_y_1(T z) {
    // Handle special values
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (z <= T(0)) {
        // Y₁ is undefined for z <= 0 (branch cut along negative real axis)
        if (z == T(0)) {
            return -std::numeric_limits<T>::infinity();
        }
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (std::isinf(z)) {
        // Y₁(+∞) = 0 (oscillatory decay)
        return T(0);
    }

    if (z <= T(5.0)) {
        // Small argument: Chebyshev/rational polynomial approximation
        // Y₁(z) = z * YP(z²)/YQ(z²) + (2/π)[J₁(z)ln(z) - 1/z]
        // Note: Cephes formula uses log(x), not log(x/2)
        T z2 = z * z;
        T w = z * detail::polevl(z2, detail::y1_YP, 5) / detail::p1evl(z2, detail::y1_YQ, 8);
        w += T(detail::Y1_TWOOPI) * (bessel_j_1(z) * std::log(z) - T(1) / z);
        return w;
    } else {
        // Large argument: asymptotic expansion
        // Y₁(z) ≈ sqrt(2/(π*z)) * [P(z)*sin(θ) + Q(z)*cos(θ)]
        // where θ = z - 3π/4
        T w = T(5.0) / z;
        T z2 = w * w;
        T p = detail::polevl(z2, detail::y1_PP, 6) / detail::polevl(z2, detail::y1_PQ, 6);
        T q = detail::polevl(z2, detail::y1_QP, 7) / detail::p1evl(z2, detail::y1_QQ, 7);

        T xn = z - T(detail::Y1_THPIO4);
        T sinxn = std::sin(xn);
        T cosxn = std::cos(xn);

        // Y₁ uses sin for P term and cos for Q term (opposite of J₁)
        p = p * sinxn + w * q * cosxn;

        return p * T(detail::Y1_SQ2OPI) / std::sqrt(z);
    }
}

// Complex version
// Note: Y₁(z) has a branch cut along the negative real axis.
// For complex z with Re(z) < 0, the result depends on the branch.
// This implementation is primarily validated near the positive real axis.
template <typename T>
c10::complex<T> bessel_y_1(c10::complex<T> z) {
    T re = z.real();
    T im = z.imag();
    T mag = std::abs(z);

    // Handle z = 0
    if (mag == T(0)) {
        return c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    }

    if (mag <= T(5.0)) {
        // Small argument: Chebyshev/rational polynomial approximation
        // Y₁(z) = z * YP(z²)/YQ(z²) + (2/π)[J₁(z)ln(z) - 1/z]
        // Note: Cephes formula uses log(z), not log(z/2)
        c10::complex<T> z2 = z * z;
        c10::complex<T> w = z * detail::polevl(z2, detail::y1_YP, 5) / detail::p1evl(z2, detail::y1_YQ, 8);

        // For complex logarithm: ln(z)
        c10::complex<T> ln_z = std::log(z);

        w += c10::complex<T>(T(detail::Y1_TWOOPI), T(0)) *
             (bessel_j_1(z) * ln_z - c10::complex<T>(T(1), T(0)) / z);
        return w;
    } else {
        // Large argument: asymptotic expansion
        c10::complex<T> inv_z = c10::complex<T>(T(5.0), T(0)) / z;
        c10::complex<T> z2 = inv_z * inv_z;
        c10::complex<T> p = detail::polevl(z2, detail::y1_PP, 6) / detail::polevl(z2, detail::y1_PQ, 6);
        c10::complex<T> q = detail::polevl(z2, detail::y1_QP, 7) / detail::p1evl(z2, detail::y1_QQ, 7);

        c10::complex<T> xn = z - c10::complex<T>(T(detail::Y1_THPIO4), T(0));
        c10::complex<T> sinxn = std::sin(xn);
        c10::complex<T> cosxn = std::cos(xn);

        // Y₁ uses sin for P term and cos for Q term
        p = p * sinxn + inv_z * q * cosxn;

        return p * c10::complex<T>(T(detail::Y1_SQ2OPI), T(0)) / std::sqrt(z);
    }
}

} // namespace torchscience::kernel::special_functions
