#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>
#include "rational_polynomial_evaluate.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Constants from Cephes
constexpr double SQPII = 5.64189583547756286948e-1;  // 1/sqrt(pi)
constexpr double C1 = 0.35502805388781723926;        // Ai(0)
constexpr double C2 = 0.258819403792806798405;       // |Ai'(0)|  (note: positive here)

// Coefficients for large positive x (x >= 2.09)
// Ai(x) = sqpii * f / (2 * t * exp(zeta))
// where t = sqrt(sqrt(x)), zeta = 2*x*sqrt(x)/3
// f = polevl(z, AN, 7) / polevl(z, AD, 7) where z = 1/zeta
constexpr double AN[] = {
    3.46538101525629032477e-1,
    1.20075952739645805542e1,
    7.62796053615234516538e1,
    1.68089224934630576269e2,
    1.59756391350164413639e2,
    7.05360906840444183113e1,
    1.40264691163389668864e1,
    9.99999999999999995305e-1,
};

constexpr double AD[] = {
    5.67594532638770212846e-1,
    1.47562562584847203173e1,
    8.45138970141474626562e1,
    1.77318088145400459522e2,
    1.64234692871529701831e2,
    7.14778400825575695274e1,
    1.40959135607834029598e1,
    1.00000000000000000470e0,
};

// Coefficients for large negative x (x < -2.09)
// Oscillatory region
// Ai(x) = k * (sin(theta)*uf - cos(theta)*ug)
// where theta = zeta + pi/4, zeta = 2*(-x)*sqrt(-x)/3
// k = sqpii / sqrt(sqrt(-x))
// uf = 1 + zz * polevl(zz, AFN, 8) / p1evl(zz, AFD, 9)
// ug = z * polevl(zz, AGN, 10) / p1evl(zz, AGD, 10)
// z = 1/zeta, zz = z*z
constexpr double AFN[] = {
    -1.31696323418331795333e-1,
    -6.26456544431912369773e-1,
    -6.93158036036933542233e-1,
    -2.79779981545119124951e-1,
    -4.91900132609500318020e-2,
    -4.06265923594885404393e-3,
    -1.59276496239262096340e-4,
    -2.77649108155232920844e-6,
    -1.67787698489114633780e-8,
};

constexpr double AFD[] = {
    // p1evl format: leading coefficient 1.0 is implicit
    1.33560420706553243746e1,
    3.26825032795224613948e1,
    2.67367040941499554804e1,
    9.18707402907259625840e0,
    1.47529146771666414581e0,
    1.15687173795188044134e-1,
    4.40291641615211203805e-3,
    7.54720348287414296618e-5,
    4.51850092970580378464e-7,
};

constexpr double AGN[] = {
    1.97339932091685679179e-2,
    3.91103029615688277255e-1,
    1.06579897599595591108e0,
    9.39169229816650230044e-1,
    3.51465656105547619242e-1,
    6.33888919628925490927e-2,
    5.85804113048388458567e-3,
    2.82851600836737019778e-4,
    6.98793669997260967291e-6,
    8.11789239554389293311e-8,
    3.41551784765923618484e-10,
};

constexpr double AGD[] = {
    // p1evl format: leading coefficient 1.0 is implicit
    9.30892908077441974853e0,
    1.98352928718312140417e1,
    1.55646628932864612953e1,
    5.47686069422975497931e0,
    9.54293611618961883998e-1,
    8.64580826352392193095e-2,
    4.12656523824222607191e-3,
    1.01259085116509135510e-4,
    1.17166733214413521882e-6,
    4.91834570062930015649e-9,
};

constexpr double MACHEP = 1.11022302462515654042e-16;  // 2^-53

} // namespace detail

// Airy function of the first kind Ai(x)
template <typename T>
T airy_ai(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // For very large positive x, underflow to 0
    if (x > T(103.892)) {
        return T(0);
    }

    // For very large negative x, also return 0 (oscillates to 0)
    if (std::isinf(x)) {
        return T(0);
    }

    T ai;

    if (x < T(-2.09)) {
        // Large negative x: oscillatory region
        T t = std::sqrt(-x);
        T zeta = T(2.0) * (-x) * t / T(3.0);  // = 2*|x|*sqrt(|x|)/3
        t = std::sqrt(t);  // t = |x|^(1/4)
        T k = T(detail::SQPII) / t;
        T z = T(1.0) / zeta;
        T zz = z * z;

        // Compute uf = 1 + zz * P(zz) / Q(zz)
        T uf = T(1.0) + zz * detail::polevl(zz, detail::AFN, 8) / detail::p1evl(zz, detail::AFD, 9);
        // Compute ug = z * P(zz) / Q(zz)
        T ug = z * detail::polevl(zz, detail::AGN, 10) / detail::p1evl(zz, detail::AGD, 10);

        T theta = zeta + T(0.25) * T(M_PI);
        T f = std::sin(theta);
        T g = std::cos(theta);

        ai = k * (f * uf - g * ug);

    } else if (x >= T(2.09)) {
        // Large positive x: exponentially decaying region
        T t = std::sqrt(x);
        T zeta = T(2.0) * x * t / T(3.0);  // = 2*x*sqrt(x)/3
        T g = std::exp(zeta);
        t = std::sqrt(t);  // t = x^(1/4)
        T k = T(2.0) * t * g;
        T z = T(1.0) / zeta;

        T f = detail::polevl(z, detail::AN, 7) / detail::polevl(z, detail::AD, 7);
        ai = T(detail::SQPII) * f / k;

    } else {
        // Small |x|: Taylor series
        // Ai(x) = c1 * f - c2 * g
        // where f = sum_{k>=0} x^{3k} * prod_{j=1}^{k} 1/((3j)(3j-1)(3j-2))
        //       g = sum_{k>=0} x^{3k+1} * prod_{j=1}^{k} 1/((3j+1)(3j)(3j-1))

        T f = T(1.0);
        T g = x;
        T t = T(1.0);
        T uf = T(1.0);
        T ug = x;
        T k_val = T(1.0);
        T z = x * x * x;

        while (t > T(detail::MACHEP)) {
            uf *= z;
            k_val += T(1.0);  // k_val = 3k+1
            uf /= k_val;
            ug *= z;
            k_val += T(1.0);  // k_val = 3k+2
            ug /= k_val;
            uf /= k_val;
            f += uf;
            k_val += T(1.0);  // k_val = 3k+3 = 3(k+1)
            ug /= k_val;
            g += ug;
            t = std::abs(uf / f);
        }

        uf = T(detail::C1) * f;
        ug = T(detail::C2) * g;
        ai = uf - ug;
    }

    return ai;
}

// Complex version of Airy Ai function
template <typename T>
c10::complex<T> airy_ai(c10::complex<T> z) {
    T re = z.real();
    T im = z.imag();

    // For purely real input, use real version
    if (std::abs(im) < T(1e-14) * (T(1.0) + std::abs(re))) {
        return c10::complex<T>(airy_ai(re), T(0));
    }

    // For complex z, use Taylor series
    // Ai(z) = c1 * f - c2 * g
    c10::complex<T> f(T(1.0), T(0));
    c10::complex<T> g = z;
    c10::complex<T> uf(T(1.0), T(0));
    c10::complex<T> ug = z;
    T k_val = T(1.0);
    c10::complex<T> z3 = z * z * z;
    T t = T(1.0);

    for (int iter = 0; iter < 100 && t > T(detail::MACHEP); ++iter) {
        uf = uf * z3;
        k_val += T(1.0);
        uf = uf / c10::complex<T>(k_val, T(0));
        ug = ug * z3;
        k_val += T(1.0);
        ug = ug / c10::complex<T>(k_val, T(0));
        uf = uf / c10::complex<T>(k_val, T(0));
        f = f + uf;
        k_val += T(1.0);
        ug = ug / c10::complex<T>(k_val, T(0));
        g = g + ug;
        t = std::abs(uf) / std::max(std::abs(f), T(1e-30));
    }

    return c10::complex<T>(T(detail::C1), T(0)) * f -
           c10::complex<T>(T(detail::C2), T(0)) * g;
}

} // namespace torchscience::kernel::special_functions
