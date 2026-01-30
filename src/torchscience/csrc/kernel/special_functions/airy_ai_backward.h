#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>
#include "rational_polynomial_evaluate.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Constants from Cephes (same as in airy_ai.h)
constexpr double AIP_SQPII = 5.64189583547756286948e-1;  // 1/sqrt(pi)
constexpr double AIP_C1 = 0.35502805388781723926;        // Ai(0)
constexpr double AIP_C2 = 0.258819403792806798405;       // |Ai'(0)|

// Coefficients for Ai'(x) for large positive x (x >= 2.09)
// Ai'(x) = -0.5 * sqpii * t * f / exp(zeta)
// where t = sqrt(sqrt(x)), zeta = 2*x*sqrt(x)/3
// f = polevl(z, APN, 7) / polevl(z, APD, 7) where z = 1/zeta
constexpr double APN[] = {
    6.13759184814035759225e-1,
    1.47454670787755323881e1,
    8.20584123476060982430e1,
    1.71184781360976385540e2,
    1.59317847137141783523e2,
    6.99778599330103016170e1,
    1.39470856980481566958e1,
    1.00000000000000000550e0,
};

constexpr double APD[] = {
    3.34203677749736953049e-1,
    1.11810297306158156705e1,
    7.11727352147859965283e1,
    1.58778084372838313640e2,
    1.53206427475809220834e2,
    6.86752304592780337944e1,
    1.38498634758259442477e1,
    9.99999999999999994502e-1,
};

// Coefficients for Ai'(x) for large negative x (oscillatory region)
// Ai'(x) = -k * (cos(theta)*uf + sin(theta)*ug)
// where theta = zeta + pi/4, k = sqpii * t, t = |x|^(1/4)
// uf = 1 + zz * polevl(zz, APFN, 8) / p1evl(zz, APFD, 9)
// ug = z * polevl(zz, APGN, 10) / p1evl(zz, APGD, 10)
constexpr double APFN[] = {
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

constexpr double APFD[] = {
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

constexpr double APGN[] = {
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

constexpr double APGD[] = {
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

constexpr double AIP_MACHEP = 1.11022302462515654042e-16;  // 2^-53

} // namespace detail

// Compute Ai'(x) - the derivative of the Airy function
template <typename T>
T airy_ai_prime(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (std::isinf(x)) {
        return T(0);
    }
    if (x > T(103.892)) {
        return T(0);  // Underflow
    }

    T aip;

    if (x < T(-2.09)) {
        // Large negative x: oscillatory region
        T t = std::sqrt(-x);
        T zeta = T(2.0) * (-x) * t / T(3.0);
        t = std::sqrt(t);  // t = |x|^(1/4)
        T k = T(detail::AIP_SQPII) * t;
        T z = T(1.0) / zeta;
        T zz = z * z;

        T uf = T(1.0) + zz * detail::polevl(zz, detail::APFN, 8) / detail::p1evl(zz, detail::APFD, 9);
        T ug = z * detail::polevl(zz, detail::APGN, 10) / detail::p1evl(zz, detail::APGD, 10);

        T theta = zeta + T(0.25) * T(M_PI);
        T f = std::sin(theta);
        T g = std::cos(theta);

        aip = -k * (g * uf + f * ug);

    } else if (x >= T(2.09)) {
        // Large positive x: exponentially decaying region
        T t = std::sqrt(x);
        T zeta = T(2.0) * x * t / T(3.0);
        T g = std::exp(zeta);
        t = std::sqrt(t);  // t = x^(1/4)
        T z = T(1.0) / zeta;

        T f = detail::polevl(z, detail::APN, 7) / detail::polevl(z, detail::APD, 7);
        T k = T(-0.5) * T(detail::AIP_SQPII) * t / g;
        aip = f * k;

    } else {
        // Small |x|: Taylor series for Ai'(x)
        // Ai'(x) = c1 * f' - c2 * g'
        // where f'(x) = derivative of f series = x^2/2 + ...
        //       g'(x) = derivative of g series = 1 + x^3/3 + ...

        // From Cephes: the derivative is computed as:
        // f = x^2/2, g = 1 + x^3/3
        // then iterate

        T k_val = T(4.0);
        T uf = x * x / T(2.0);
        T ug = x * x * x / T(3.0);
        T f = uf;
        T g = T(1.0) + ug;
        uf /= T(3.0);
        T t = T(1.0);
        T z = x * x * x;

        while (t > T(detail::AIP_MACHEP)) {
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

        uf = T(detail::AIP_C1) * f;
        ug = T(detail::AIP_C2) * g;
        aip = uf - ug;
    }

    return aip;
}

// Complex version of Ai'(z)
template <typename T>
c10::complex<T> airy_ai_prime(c10::complex<T> z) {
    T re = z.real();
    T im = z.imag();

    // For purely real input, use real version
    if (std::abs(im) < T(1e-14) * (T(1.0) + std::abs(re))) {
        return c10::complex<T>(airy_ai_prime(re), T(0));
    }

    // For complex z, use Taylor series for Ai'(z)
    T k_val = T(4.0);
    c10::complex<T> z2 = z * z;
    c10::complex<T> z3 = z2 * z;
    c10::complex<T> uf = z2 / c10::complex<T>(T(2.0), T(0));
    c10::complex<T> ug = z3 / c10::complex<T>(T(3.0), T(0));
    c10::complex<T> f = uf;
    c10::complex<T> g = c10::complex<T>(T(1.0), T(0)) + ug;
    uf = uf / c10::complex<T>(T(3.0), T(0));
    T t = T(1.0);

    for (int iter = 0; iter < 100 && t > T(detail::AIP_MACHEP); ++iter) {
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

    return c10::complex<T>(T(detail::AIP_C1), T(0)) * f -
           c10::complex<T>(T(detail::AIP_C2), T(0)) * g;
}

// Backward pass for airy_ai: d/dx Ai(x) = Ai'(x)
template <typename T>
T airy_ai_backward(T grad_output, T x) {
    return grad_output * airy_ai_prime(x);
}

// Complex backward
template <typename T>
c10::complex<T> airy_ai_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> aip = airy_ai_prime(z);
    return grad_output * std::conj(aip);
}

} // namespace torchscience::kernel::special_functions
