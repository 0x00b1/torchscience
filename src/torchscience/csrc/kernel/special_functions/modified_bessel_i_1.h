#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>
#include "chebyshev_polynomial_t_series_evaluate.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Cephes coefficients for exp(-|x|) * I₁(x) / x in interval [0, 8]
// Source: Cephes Math Library (Stephen L. Moshier)
// The actual I₁(x) = x * exp(|x|) * chbevl(x/2 - 2, i1_A, 29)
constexpr double i1_A[] = {
     2.77791411276104639959E-18, -2.11142121435816608115E-17,
     1.55363195773620046921E-16, -1.10559694773538630805E-15,
     7.60068429473540693410E-15, -5.04218550472791168711E-14,
     3.22379336594557470981E-13, -1.98397439776494371520E-12,
     1.17361862988909016308E-11, -6.66348972350202774223E-11,
     3.62559028155211703701E-10, -1.88724975172282928790E-9,
     9.38153738649577178388E-9,  -4.44505912879632808065E-8,
     2.00329475355213526229E-7,  -8.56872026469545474066E-7,
     3.47025130813767847674E-6,  -1.32731636560394358279E-5,
     4.78156510755005422638E-5,  -1.61760815825896745588E-4,
     5.12285956168575772895E-4,  -1.51357245063125314899E-3,
     4.15642294431288815669E-3,  -1.05640848946261981558E-2,
     2.47264490306265168283E-2,  -5.29459812080949914269E-2,
     1.02643658689847095384E-1,  -1.76416518357834055153E-1,
     2.52587186443633654823E-1
};

// Cephes coefficients for I₁(x) large argument (|x| > 8)
constexpr double i1_B[] = {
     7.51729631084210481353E-18,  4.41434832307170791151E-18,
    -4.65030536848935832153E-17, -3.20952592199342395980E-17,
     2.96262899764595013876E-16,  3.30820231092092828324E-16,
    -1.88035477551078244854E-15, -3.81440307243700780478E-15,
     1.04202769841288027642E-14,  4.27244001671195135429E-14,
    -2.10154184277266431302E-14, -4.08355111109219731823E-13,
    -7.19855177624590851209E-13,  2.03562854414708950722E-12,
     1.41258074366137813316E-11,  3.25260358301548823856E-11,
    -1.89749581235054123450E-11, -5.58974346219658380687E-10,
    -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
    -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
     7.78576235018280120474E-1
};

} // namespace detail

template <typename T>
T modified_bessel_i_1(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (std::isinf(x)) {
        // I₁ is odd: I₁(-∞) = -∞, I₁(+∞) = +∞
        return std::copysign(std::numeric_limits<T>::infinity(), x);
    }

    T ax = std::abs(x);

    if (ax <= T(8.0)) {
        // Small argument: Chebyshev expansion
        T y = (ax / T(2.0)) - T(2.0);
        T result = ax * std::exp(ax) * chebyshev_polynomial_t_series_evaluate(y, detail::i1_A, 29);
        return (x < T(0)) ? -result : result;
    } else {
        // Large argument: asymptotic expansion
        T result = std::exp(ax) * chebyshev_polynomial_t_series_evaluate(T(32.0) / ax - T(2.0), detail::i1_B, 25)
                   / std::sqrt(ax);
        return (x < T(0)) ? -result : result;
    }
}

// Complex version
// Note: The asymptotic expansion is primarily validated near the real axis.
// For complex z far from the real axis, accuracy should be verified empirically.
// I₁ satisfies the odd function property: I₁(-z) = -I₁(z)
template <typename T>
c10::complex<T> modified_bessel_i_1(c10::complex<T> z) {
    T mag = std::abs(z);

    if (mag <= T(8.0)) {
        // Small argument: multiplication by z preserves odd function property
        c10::complex<T> y = (z / T(2.0)) - c10::complex<T>(T(2.0), T(0));
        return z * std::exp(z) * chebyshev_polynomial_t_series_evaluate(y, detail::i1_A, 29);
    } else {
        // Large argument: asymptotic expansion I₁(z) ≈ e^z / sqrt(2πz) * [1 + ...]
        // The sqrt(z) in denominator provides the necessary branch behavior
        c10::complex<T> inv_z = c10::complex<T>(T(32.0), T(0)) / z - c10::complex<T>(T(2.0), T(0));
        return std::exp(z) * chebyshev_polynomial_t_series_evaluate(inv_z, detail::i1_B, 25) / std::sqrt(z);
    }
}

} // namespace torchscience::kernel::special_functions
