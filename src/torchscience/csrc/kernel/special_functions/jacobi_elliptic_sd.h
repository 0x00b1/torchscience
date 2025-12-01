#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "jacobi_elliptic_sn.h"
#include "jacobi_elliptic_dn.h"

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function sd(u, m)
//
// Mathematical definition:
// sd(u, m) = sn(u, m) / dn(u, m)
//
// where:
// - sn(u, m) is the Jacobi elliptic sine
// - dn(u, m) is the Jacobi elliptic delta amplitude
//
// Domain:
// - u: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - sd(0, m) = 0 for all m
// - sd(u, 0) = sin(u) (since sn(u,0) = sin(u) and dn(u,0) = 1)
// - sd(u, 1) = sinh(u) (since sn(u,1) = tanh(u) and dn(u,1) = sech(u))
// - sd(-u, m) = -sd(u, m) (odd function in u)
//
// The sd function has poles where dn(u, m) = 0, which occurs at:
// u = (2n+1)K(m) + i*K'(m) for integer n
//
// Relationship to other Jacobi functions:
// sd * cd = sn (since sd * cd = sn/dn * cn/dn = sn*cn/dn^2, but more simply sd = sn/dn)
// sd^2 * dn^2 = sn^2

template <typename T>
T jacobi_elliptic_sd(T u, T m) {
    T sn = jacobi_elliptic_sn(u, m);
    T dn = jacobi_elliptic_dn(u, m);

    // Handle case where dn is very small (near pole)
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    if (std::abs(dn) < eps) {
        // Return signed infinity based on sign of sn
        if (sn > T(0)) {
            return std::numeric_limits<T>::infinity();
        } else if (sn < T(0)) {
            return -std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::quiet_NaN();
        }
    }

    return sn / dn;
}

template <typename T>
c10::complex<T> jacobi_elliptic_sd(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> sn = jacobi_elliptic_sn(u, m);
    c10::complex<T> dn = jacobi_elliptic_dn(u, m);

    // Handle case where dn is very small (near pole)
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    if (std::abs(dn) < eps) {
        // Return complex infinity
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    return sn / dn;
}

} // namespace torchscience::kernel::special_functions
