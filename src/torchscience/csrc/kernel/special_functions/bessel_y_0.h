#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>
#include "rational_polynomial_evaluate.h"
#include "bessel_j_0.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Cephes coefficients for Y₀(z) rational approximation (|z| <= 5)
// Source: Cephes Math Library (Stephen L. Moshier)
// Y₀(z) ≈ YP(z²)/YQ(z²) + (2/π) * J₀(z) * ln(z)

constexpr double y0_YP[] = {
     1.55924367855235737965E4,
    -1.46639295903971606143E7,
     5.43526477051876500413E9,
    -9.82136065717911466409E11,
     8.75906394395366999549E13,
    -3.46628303384729719441E15,
     4.42733268572569800351E16,
    -1.84950800436986690637E16,
};

constexpr double y0_YQ[] = {
    // 1.0 (implicit leading coefficient)
     1.04128353664259848412E3,
     6.26107330137134956842E5,
     2.68919633393814121987E8,
     8.64002487103935000337E10,
     2.02979612750105546709E13,
     3.17157752842975028269E15,
     2.50596256172653059228E17,
};

// Cephes coefficients for asymptotic expansion (|z| > 5)
// Y₀(z) ≈ sqrt(2/(π*z)) * [P(z)*sin(θ) + Q(z)*cos(θ)]
// where θ = z - π/4
// These coefficients are the same as for J₀

constexpr double y0_PP[] = {
     7.96936729297347051624E-4,
     8.28352392107440799803E-2,
     1.23953371646414299388E0,
     5.44725003058768775090E0,
     8.74716500199817011941E0,
     5.30324038235394892183E0,
     9.99999999999999997821E-1,
};

constexpr double y0_PQ[] = {
     9.24408810558863637013E-4,
     8.56288474354474431428E-2,
     1.25352743901058953537E0,
     5.47097740330417105182E0,
     8.76190883237069594232E0,
     5.30605288235394617618E0,
     1.00000000000000000218E0,
};

constexpr double y0_QP[] = {
    -1.13663838898469149931E-2,
    -1.28252718670509318512E0,
    -1.95539544257735972385E1,
    -9.32060152123768231369E1,
    -1.77681167980488050595E2,
    -1.47077505154951170175E2,
    -5.14105326766599330220E1,
    -6.05014350600728481186E0,
};

constexpr double y0_QQ[] = {
    // 1.0 (implicit leading coefficient)
     6.43178256118178023184E1,
     8.56430025976980587198E2,
     3.88240183605401609683E3,
     7.24046774195652478189E3,
     5.93072701187316984827E3,
     2.06209331660327847417E3,
     2.42005740240291393179E2,
};

constexpr double Y0_TWOOPI = 0.63661977236758134307554;  // 2/pi
constexpr double Y0_SQ2OPI = 0.79788456080286535587989;  // sqrt(2/pi)
constexpr double Y0_PIO4 = 0.78539816339744830961566;    // pi/4

} // namespace detail

template <typename T>
T bessel_y_0(T z) {
    // Handle special values
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (z <= T(0)) {
        // Y₀ is undefined for z <= 0 (branch cut along negative real axis)
        if (z == T(0)) {
            return -std::numeric_limits<T>::infinity();
        }
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (std::isinf(z)) {
        // Y₀(+∞) = 0 (oscillatory decay)
        return T(0);
    }

    if (z <= T(5.0)) {
        // Small argument: Chebyshev/rational polynomial approximation
        // Y₀(z) = YP(z²)/YQ(z²) + (2/π) * J₀(z) * ln(z)
        T z2 = z * z;
        T w = detail::polevl(z2, detail::y0_YP, 7) / detail::p1evl(z2, detail::y0_YQ, 7);
        w += T(detail::Y0_TWOOPI) * bessel_j_0(z) * std::log(z);
        return w;
    } else {
        // Large argument: asymptotic expansion
        // Y₀(z) ≈ sqrt(2/(π*z)) * [P(z)*sin(θ) + Q(z)*cos(θ)]
        // where θ = z - π/4
        T w = T(5.0) / z;
        T z2 = T(25.0) / (z * z);
        T p = detail::polevl(z2, detail::y0_PP, 6) / detail::polevl(z2, detail::y0_PQ, 6);
        T q = detail::polevl(z2, detail::y0_QP, 7) / detail::p1evl(z2, detail::y0_QQ, 7);

        T xn = z - T(detail::Y0_PIO4);
        T sinxn = std::sin(xn);
        T cosxn = std::cos(xn);

        // Y₀ uses sin for P term and cos for Q term (like Y₁, opposite of J₀)
        p = p * sinxn + w * q * cosxn;

        return p * T(detail::Y0_SQ2OPI) / std::sqrt(z);
    }
}

// Complex version
// Note: Y₀(z) has a branch cut along the negative real axis.
// For complex z with Re(z) < 0, the result depends on the branch.
// This implementation is primarily validated near the positive real axis.
template <typename T>
c10::complex<T> bessel_y_0(c10::complex<T> z) {
    T re = z.real();
    T im = z.imag();
    T mag = std::abs(z);

    // Handle z = 0
    if (mag == T(0)) {
        return c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    }

    if (mag <= T(5.0)) {
        // Small argument: Chebyshev/rational polynomial approximation
        // Y₀(z) = YP(z²)/YQ(z²) + (2/π) * J₀(z) * ln(z)
        c10::complex<T> z2 = z * z;
        c10::complex<T> w = detail::polevl(z2, detail::y0_YP, 7) / detail::p1evl(z2, detail::y0_YQ, 7);

        // For complex logarithm: ln(z)
        c10::complex<T> ln_z = std::log(z);

        w += c10::complex<T>(T(detail::Y0_TWOOPI), T(0)) * bessel_j_0(z) * ln_z;
        return w;
    } else {
        // Large argument: asymptotic expansion
        c10::complex<T> inv_z = c10::complex<T>(T(5.0), T(0)) / z;
        c10::complex<T> z2 = c10::complex<T>(T(25.0), T(0)) / (z * z);
        c10::complex<T> p = detail::polevl(z2, detail::y0_PP, 6) / detail::polevl(z2, detail::y0_PQ, 6);
        c10::complex<T> q = detail::polevl(z2, detail::y0_QP, 7) / detail::p1evl(z2, detail::y0_QQ, 7);

        c10::complex<T> xn = z - c10::complex<T>(T(detail::Y0_PIO4), T(0));
        c10::complex<T> sinxn = std::sin(xn);
        c10::complex<T> cosxn = std::cos(xn);

        // Y₀ uses sin for P term and cos for Q term
        p = p * sinxn + inv_z * q * cosxn;

        return p * c10::complex<T>(T(detail::Y0_SQ2OPI), T(0)) / std::sqrt(z);
    }
}

} // namespace torchscience::kernel::special_functions
