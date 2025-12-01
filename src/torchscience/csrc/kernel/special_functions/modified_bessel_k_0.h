#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>
#include "chebyshev_polynomial_t_series_evaluate.h"
#include "modified_bessel_i_0.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Cephes coefficients for K0(x) + log(x/2) * I0(x) in interval [0, 2]
// Source: Cephes Math Library (Stephen L. Moshier)
constexpr double k0_A[] = {
     1.37446543561352307156E-16,
     4.25981614279661018399E-14,
     1.03496952576338420167E-11,
     1.90451637722020886025E-9,
     2.53479107902614945675E-7,
     2.28621210311945178607E-5,
     1.26461541144692592338E-3,
     3.59799365153615016266E-2,
     3.44289899924628486886E-1,
    -5.35327393233902768720E-1
};

// Cephes coefficients for exp(x) * sqrt(x) * K0(x) in interval [2, infinity)
constexpr double k0_B[] = {
     5.30043377268626276149E-18, -1.64758043015242134646E-17,
     5.21039150503902756861E-17, -1.67823109680541210385E-16,
     5.51205597852431940784E-16, -1.84859337734377901440E-15,
     6.34007647740507060557E-15, -2.22751332699166985548E-14,
     8.03289077536357521100E-14, -2.98009692317273043925E-13,
     1.14034058820847496303E-12, -4.51459788337394416547E-12,
     1.85594911495471785253E-11, -7.95748924447710747776E-11,
     3.57739728140030116597E-10, -1.69753450938905987466E-9,
     8.57403401741422608519E-9,  -4.66048989768794782956E-8,
     2.76681363944501510342E-7,  -1.83175552271911948767E-6,
     1.39498137188764993662E-5,  -1.28495495816278026384E-4,
     1.56988388573005337491E-3,  -3.14481013119645005427E-2,
     2.44030308206595545468E0
};

} // namespace detail

template <typename T>
T modified_bessel_k_0(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x <= T(0)) {
        // K0 is only defined for x > 0
        if (x == T(0)) {
            return std::numeric_limits<T>::infinity();
        }
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (std::isinf(x)) {
        // K0(+inf) = 0 (exponential decay)
        return T(0);
    }

    if (x <= T(2.0)) {
        // Small argument: K0(x) = -log(x/2)*I0(x) + P(x^2)
        // where P is the Chebyshev expansion
        T y = x * x - T(2.0);
        T result = chebyshev_polynomial_t_series_evaluate(y, detail::k0_A, 10);
        result -= std::log(x / T(2.0)) * modified_bessel_i_0(x);
        return result;
    } else {
        // Large argument: K0(x) = exp(-x) / sqrt(x) * Q(8/x - 2)
        T y = T(8.0) / x - T(2.0);
        T result = chebyshev_polynomial_t_series_evaluate(y, detail::k0_B, 25);
        return result * std::exp(-x) / std::sqrt(x);
    }
}

// Complex version
template <typename T>
c10::complex<T> modified_bessel_k_0(c10::complex<T> z) {
    T mag = std::abs(z);

    // Handle z = 0
    if (mag == T(0)) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    if (mag <= T(2.0)) {
        // Small argument: K0(z) = -log(z/2)*I0(z) + P(z^2 - 2)
        c10::complex<T> y = z * z - c10::complex<T>(T(2.0), T(0));
        c10::complex<T> result = chebyshev_polynomial_t_series_evaluate(y, detail::k0_A, 10);
        // log(z/2) for complex z
        c10::complex<T> ln_z_half = std::log(z / c10::complex<T>(T(2.0), T(0)));
        result -= ln_z_half * modified_bessel_i_0(z);
        return result;
    } else {
        // Large argument: K0(z) = exp(-z) / sqrt(z) * Q(8/z - 2)
        c10::complex<T> y = c10::complex<T>(T(8.0), T(0)) / z - c10::complex<T>(T(2.0), T(0));
        c10::complex<T> result = chebyshev_polynomial_t_series_evaluate(y, detail::k0_B, 25);
        return result * std::exp(-z) / std::sqrt(z);
    }
}

} // namespace torchscience::kernel::special_functions
