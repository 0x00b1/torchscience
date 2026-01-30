#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>
#include "rational_polynomial_evaluate.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Constants from Cephes for Bi(x)
constexpr double BI_SQPII = 5.64189583547756286948e-1;  // 1/sqrt(pi)
constexpr double BI_C1 = 0.35502805388781723926;        // Ai(0) used for series
constexpr double BI_C2 = 0.258819403792806798405;       // |Ai'(0)| used for series
constexpr double BI_SQRT3 = 1.732050807568877293527;    // sqrt(3) for Bi = sqrt(3)*(c1*f+c2*g)

// Coefficients for very large positive x (x > 8.3203353, zeta > 16)
// Bi(x) = sqpii * exp(zeta) * (1 + f) / t
// where t = x^(1/4), zeta = 2*x*sqrt(x)/3
// f = z * polevl(z, BN16, 4) / p1evl(z, BD16, 5) where z = 1/zeta
constexpr double BN16[] = {
    -2.53240795869364152689e-1,
     5.75285167332467384228e-1,
    -3.29907036873225371650e-1,
     6.44404068948199951727e-2,
    -3.82519546641336734394e-3,
};

constexpr double BD16[] = {
    // p1evl format: leading coefficient 1.0 is implicit
    -7.15685095054035237902e0,
     1.06039580715664694291e1,
    -5.23246636471251500874e0,
     9.57395864378383833152e-1,
    -5.50828147163549611107e-2,
};

// Coefficients for large negative x (x < -2.09)
// Oscillatory region
// Bi(x) = k * (cos(theta)*uf + sin(theta)*ug)
// Note: different combination than Ai (cos*uf + sin*ug vs sin*uf - cos*ug)
// where theta = zeta + pi/4, zeta = 2*|x|*sqrt(|x|)/3
// k = sqpii / sqrt(sqrt(-x))
// uf = 1 + zz * polevl(zz, AFN, 8) / p1evl(zz, AFD, 9)
// ug = z * polevl(zz, AGN, 10) / p1evl(zz, AGD, 10)
constexpr double BFN[] = {
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

constexpr double BFD[] = {
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

constexpr double BGN[] = {
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

constexpr double BGD[] = {
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

constexpr double BI_MACHEP = 1.11022302462515654042e-16;  // 2^-53

} // namespace detail

// Airy function of the second kind Bi(x)
template <typename T>
T airy_bi(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // For very large positive x, overflow to infinity
    if (x > T(104.0)) {
        return std::numeric_limits<T>::infinity();
    }

    // For positive infinity, return infinity
    if (std::isinf(x) && x > T(0)) {
        return std::numeric_limits<T>::infinity();
    }

    // For negative infinity, oscillates -> 0
    if (std::isinf(x) && x < T(0)) {
        return T(0);
    }

    T bi;

    if (x < T(-2.09)) {
        // Large negative x: oscillatory region
        // Bi(x) = k * (cos(theta)*uf + sin(theta)*ug)
        T t = std::sqrt(-x);
        T zeta = T(2.0) * (-x) * t / T(3.0);  // = 2*|x|*sqrt(|x|)/3
        t = std::sqrt(t);  // t = |x|^(1/4)
        T k = T(detail::BI_SQPII) / t;
        T z = T(1.0) / zeta;
        T zz = z * z;

        // Compute uf = 1 + zz * P(zz) / Q(zz)
        T uf = T(1.0) + zz * detail::polevl(zz, detail::BFN, 8) / detail::p1evl(zz, detail::BFD, 9);
        // Compute ug = z * P(zz) / Q(zz)
        T ug = z * detail::polevl(zz, detail::BGN, 10) / detail::p1evl(zz, detail::BGD, 10);

        T theta = zeta + T(0.25) * T(M_PI);
        T f = std::sin(theta);
        T g = std::cos(theta);

        // Bi(x) = k * (cos(theta)*uf + sin(theta)*ug)
        bi = k * (g * uf + f * ug);

    } else if (x > T(8.3203353)) {
        // Very large positive x (zeta > 16): asymptotic expansion
        // Bi(x) = sqpii * exp(zeta) * (1 + f) / t
        // where t = x^(1/4), zeta = 2*x*sqrt(x)/3
        // f = z * polevl(z, BN16, 4) / p1evl(z, BD16, 5)
        T t = std::sqrt(x);
        T zeta = T(2.0) * x * t / T(3.0);  // = 2*x*sqrt(x)/3
        T z = T(1.0) / zeta;

        T f = z * detail::polevl(z, detail::BN16, 4) / detail::p1evl(z, detail::BD16, 5);

        // For very large x, be careful about overflow
        if (zeta > T(700.0)) {
            // Use log-space computation
            T log_bi = zeta - T(0.25) * std::log(x) - T(0.5) * std::log(T(M_PI)) + std::log(T(1.0) + f);
            bi = std::exp(log_bi);
        } else {
            T g = std::exp(zeta);
            t = std::sqrt(t);  // t = x^(1/4)
            T k = T(detail::BI_SQPII) * g;
            bi = k * (T(1.0) + f) / t;
        }

    } else {
        // Small and moderate |x|: Taylor series
        // Bi(x) = sqrt(3) * (c1*f + c2*g)
        // where f = sum_{k>=0} x^{3k} * prod_{j=1}^{k} 1/((3j)(3j-1)(3j-2))
        //       g = sum_{k>=0} x^{3k+1} * prod_{j=1}^{k} 1/((3j+1)(3j)(3j-1))
        // and c1 = Ai(0), c2 = |Ai'(0)|

        T f = T(1.0);
        T g = x;
        T t = T(1.0);
        T uf = T(1.0);
        T ug = x;
        T k_val = T(1.0);
        T z = x * x * x;

        while (t > T(detail::BI_MACHEP)) {
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

        // Bi(x) = sqrt(3) * (c1*f + c2*g)
        uf = T(detail::BI_C1) * f;
        ug = T(detail::BI_C2) * g;
        bi = T(detail::BI_SQRT3) * (uf + ug);
    }

    return bi;
}

// Complex version of Airy Bi function
template <typename T>
c10::complex<T> airy_bi(c10::complex<T> z) {
    T re = z.real();
    T im = z.imag();

    // For purely real input, use real version
    if (std::abs(im) < T(1e-14) * (T(1.0) + std::abs(re))) {
        return c10::complex<T>(airy_bi(re), T(0));
    }

    // For complex z, use Taylor series
    // Bi(z) = c1 * f + c2 * g
    c10::complex<T> f(T(1.0), T(0));
    c10::complex<T> g = z;
    c10::complex<T> uf(T(1.0), T(0));
    c10::complex<T> ug = z;
    T k_val = T(1.0);
    c10::complex<T> z3 = z * z * z;
    T t = T(1.0);

    for (int iter = 0; iter < 100 && t > T(detail::BI_MACHEP); ++iter) {
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

    // Bi(z) = sqrt(3) * (c1*f + c2*g)
    return c10::complex<T>(T(detail::BI_SQRT3), T(0)) *
           (c10::complex<T>(T(detail::BI_C1), T(0)) * f +
            c10::complex<T>(T(detail::BI_C2), T(0)) * g);
}

} // namespace torchscience::kernel::special_functions
