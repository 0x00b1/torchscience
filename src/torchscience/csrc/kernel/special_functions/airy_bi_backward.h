#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>
#include "rational_polynomial_evaluate.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Constants from Cephes for Bi'(x)
constexpr double BIP_SQPII = 5.64189583547756286948e-1;  // 1/sqrt(pi)
constexpr double BIP_C1 = 0.35502805388781723926;        // Ai(0) used for series
constexpr double BIP_C2 = 0.258819403792806798405;       // |Ai'(0)| used for series
constexpr double BIP_SQRT3 = 1.732050807568877293527;    // sqrt(3) for Bi' = sqrt(3)*(c1*f'+c2*g')

// Coefficients for Bi'(x) for very large positive x (x > 8.3203353, zeta > 16)
// Bi'(x) = sqpii * exp(zeta) * t * (1 + f)
// where t = x^(1/4), zeta = 2*x*sqrt(x)/3
// f = z * polevl(z, BPPN, 4) / p1evl(z, BPPD, 5)
constexpr double BPPN[] = {
     4.65461162774651610328e-1,
    -1.08992173800493920734e0,
     6.38800117371827987759e-1,
    -1.26844349553102907034e-1,
     7.62487844342109852105e-3,
};

constexpr double BPPD[] = {
    // p1evl format: leading coefficient 1.0 is implicit
    -8.70622787633159124240e0,
     1.38993162704553213172e1,
    -7.14116144616431159572e0,
     1.34008595960680518666e0,
    -7.84273211323341930448e-2,
};

// Coefficients for Bi'(x) for large negative x (oscillatory region)
// Bi'(x) = k * (sin(theta)*uf - cos(theta)*ug)
// Note: different signs than Ai'(x) which has -k*(cos(theta)*uf + sin(theta)*ug)
// where theta = zeta + pi/4, k = sqpii * t, t = |x|^(1/4)
// uf = 1 + zz * polevl(zz, BPFN, 8) / p1evl(zz, BPFD, 9)
// ug = z * polevl(zz, BPGN, 10) / p1evl(zz, BPGD, 10)
constexpr double BPFN[] = {
    1.85365624022535566142e-1,
    8.86712188052584095637e-1,
    9.87391981747398547272e-1,
    4.01241082318003734092e-1,
    7.10304926289631174579e-2,
    5.90618657995661810071e-3,
    2.33051409401776799569e-4,
    4.08718778289035454598e-6,
    2.48379932900442457853e-8,
};

constexpr double BPFD[] = {
    // p1evl format: leading coefficient 1.0 is implicit
    1.47345854687502542552e1,
    3.75423933435489594466e1,
    3.14657751203046424330e1,
    1.09969125207298778536e1,
    1.78885054766999417817e0,
    1.41733275753662636873e-1,
    5.44066067017226003627e-3,
    9.39421290654511171663e-5,
    5.65978713036027009243e-7,
};

constexpr double BPGN[] = {
    -3.55615429033082288335e-2,
    -6.37311518129435504426e-1,
    -1.70856738884312371053e0,
    -1.50221872117316635393e0,
    -5.63606665822102676611e-1,
    -1.02101031120216891789e-1,
    -9.48396695961445269093e-3,
    -4.60325307486780994357e-4,
    -1.14300836484517375919e-5,
    -1.33415518685547420648e-7,
    -5.63803833958893494476e-10,
};

constexpr double BPGD[] = {
    // p1evl format: leading coefficient 1.0 is implicit
    9.85865801696130355144e0,
    2.16401867356585941885e1,
    1.73130776389749389525e1,
    6.17872175280828766327e0,
    1.08848694396321495475e0,
    9.95005543440888479402e-2,
    4.78468199683886610842e-3,
    1.18159633322838625562e-4,
    1.37480673554219441465e-6,
    5.79912514929147598821e-9,
};

constexpr double BIP_MACHEP = 1.11022302462515654042e-16;  // 2^-53

} // namespace detail

// Compute Bi'(x) - the derivative of the Airy function of the second kind
template <typename T>
T airy_bi_prime(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (std::isinf(x)) {
        if (x > T(0)) {
            return std::numeric_limits<T>::infinity();  // Bi'(+inf) = +inf
        } else {
            return T(0);  // Bi'(-inf) oscillates to 0
        }
    }
    if (x > T(104.0)) {
        return std::numeric_limits<T>::infinity();  // Overflow
    }

    T bip;

    if (x < T(-2.09)) {
        // Large negative x: oscillatory region
        // Bi'(x) = k * (sin(theta)*uf - cos(theta)*ug)
        T t = std::sqrt(-x);
        T zeta = T(2.0) * (-x) * t / T(3.0);
        t = std::sqrt(t);  // t = |x|^(1/4)
        T k = T(detail::BIP_SQPII) * t;
        T z = T(1.0) / zeta;
        T zz = z * z;

        T uf = T(1.0) + zz * detail::polevl(zz, detail::BPFN, 8) / detail::p1evl(zz, detail::BPFD, 9);
        T ug = z * detail::polevl(zz, detail::BPGN, 10) / detail::p1evl(zz, detail::BPGD, 10);

        T theta = zeta + T(0.25) * T(M_PI);
        T f = std::sin(theta);
        T g = std::cos(theta);

        // Bi'(x) = k * (sin(theta)*uf - cos(theta)*ug)
        bip = k * (f * uf - g * ug);

    } else if (x > T(8.3203353)) {
        // Very large positive x (zeta > 16): asymptotic expansion
        // Bi'(x) = sqpii * exp(zeta) * t * (1 + f)
        // where t = x^(1/4), zeta = 2*x*sqrt(x)/3
        // f = z * polevl(z, BPPN, 4) / p1evl(z, BPPD, 5)
        T t = std::sqrt(x);
        T zeta = T(2.0) * x * t / T(3.0);
        T z = T(1.0) / zeta;

        T f = z * detail::polevl(z, detail::BPPN, 4) / detail::p1evl(z, detail::BPPD, 5);

        // For very large x, be careful about overflow
        if (zeta > T(700.0)) {
            // Use log-space computation
            // Bi' = sqpii * exp(zeta) * t * (1 + f) where t = x^(1/4)
            T log_bip = zeta + T(0.25) * std::log(x) - T(0.5) * std::log(T(M_PI)) + std::log(T(1.0) + f);
            bip = std::exp(log_bip);
        } else {
            T g = std::exp(zeta);
            t = std::sqrt(t);  // t = x^(1/4)
            T k = T(detail::BIP_SQPII) * g;
            bip = k * t * (T(1.0) + f);
        }

    } else {
        // Small and moderate |x|: Taylor series for Bi'(x)
        // Bi'(x) = sqrt(3) * (c1*f' + c2*g')
        // where f'(x) = derivative of f series = x^2/2 + ...
        //       g'(x) = derivative of g series = 1 + x^3/3 + ...
        // and c1 = Ai(0), c2 = |Ai'(0)|

        T k_val = T(4.0);
        T uf = x * x / T(2.0);
        T ug = x * x * x / T(3.0);
        T f = uf;
        T g = T(1.0) + ug;
        uf /= T(3.0);
        T t = T(1.0);
        T z = x * x * x;

        while (t > T(detail::BIP_MACHEP)) {
            uf *= z;
            ug /= k_val;
            k_val += T(1.0);
            ug *= z;
            uf /= k_val;
            f += uf;
            k_val += T(1.0);
            ug /= k_val;
            uf /= k_val;
            g += ug;
            k_val += T(1.0);
            t = std::abs(ug / g);
        }

        // Bi'(x) = sqrt(3) * (c1*f' + c2*g')
        uf = T(detail::BIP_C1) * f;
        ug = T(detail::BIP_C2) * g;
        bip = T(detail::BIP_SQRT3) * (uf + ug);
    }

    return bip;
}

// Complex version of Bi'(z)
template <typename T>
c10::complex<T> airy_bi_prime(c10::complex<T> z) {
    T re = z.real();
    T im = z.imag();

    // For purely real input, use real version
    if (std::abs(im) < T(1e-14) * (T(1.0) + std::abs(re))) {
        return c10::complex<T>(airy_bi_prime(re), T(0));
    }

    // For complex z, use Taylor series for Bi'(z)
    T k_val = T(4.0);
    c10::complex<T> z2 = z * z;
    c10::complex<T> z3 = z2 * z;
    c10::complex<T> uf = z2 / c10::complex<T>(T(2.0), T(0));
    c10::complex<T> ug = z3 / c10::complex<T>(T(3.0), T(0));
    c10::complex<T> f = uf;
    c10::complex<T> g = c10::complex<T>(T(1.0), T(0)) + ug;
    uf = uf / c10::complex<T>(T(3.0), T(0));
    T t = T(1.0);

    for (int iter = 0; iter < 100 && t > T(detail::BIP_MACHEP); ++iter) {
        uf = uf * z3;
        ug = ug / c10::complex<T>(k_val, T(0));
        k_val += T(1.0);
        ug = ug * z3;
        uf = uf / c10::complex<T>(k_val, T(0));
        f = f + uf;
        k_val += T(1.0);
        ug = ug / c10::complex<T>(k_val, T(0));
        uf = uf / c10::complex<T>(k_val, T(0));
        g = g + ug;
        k_val += T(1.0);
        t = std::abs(ug) / std::max(std::abs(g), T(1e-30));
    }

    // Bi'(z) = sqrt(3) * (c1*f' + c2*g')
    return c10::complex<T>(T(detail::BIP_SQRT3), T(0)) *
           (c10::complex<T>(T(detail::BIP_C1), T(0)) * f +
            c10::complex<T>(T(detail::BIP_C2), T(0)) * g);
}

// Backward pass for airy_bi: d/dx Bi(x) = Bi'(x)
template <typename T>
T airy_bi_backward(T grad_output, T x) {
    return grad_output * airy_bi_prime(x);
}

// Complex backward
template <typename T>
c10::complex<T> airy_bi_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> bip = airy_bi_prime(z);
    return grad_output * std::conj(bip);
}

} // namespace torchscience::kernel::special_functions
