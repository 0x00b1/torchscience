#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "jacobi_elliptic_cn.h"
#include "jacobi_elliptic_dn.h"

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function cd(u, m)
//
// Mathematical definition:
// cd(u, m) = cn(u, m) / dn(u, m)
//
// where:
// - cn(u, m) is the Jacobi elliptic cosine
// - dn(u, m) is the Jacobi elliptic delta amplitude
//
// Domain:
// - u: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - cd(0, m) = 1 for all m (since cn(0,m) = 1 and dn(0,m) = 1)
// - cd(u, 0) = cos(u) (since cn(u,0) = cos(u) and dn(u,0) = 1)
// - cd(u, 1) = 1 (since cn(u,1) = sech(u) and dn(u,1) = sech(u))
// - cd(-u, m) = cd(u, m) (even function in u)
//
// The cd function has poles where dn(u, m) = 0, which occurs at:
// u = (2n+1)K(m) + i*K'(m) for integer n
//
// Relationship to other Jacobi functions:
// cd = cn/dn
// cd^2 + sd^2 * m = 1 (follows from cn^2 + sn^2 = 1 and dn^2 + m*sn^2 = 1)

template <typename T>
T jacobi_elliptic_cd(T u, T m) {
    T cn = jacobi_elliptic_cn(u, m);
    T dn = jacobi_elliptic_dn(u, m);

    // Handle case where dn is very small (near pole)
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    if (std::abs(dn) < eps) {
        // Return signed infinity based on sign of cn
        if (cn > T(0)) {
            return std::numeric_limits<T>::infinity();
        } else if (cn < T(0)) {
            return -std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::quiet_NaN();
        }
    }

    return cn / dn;
}

template <typename T>
c10::complex<T> jacobi_elliptic_cd(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> cn = jacobi_elliptic_cn(u, m);
    c10::complex<T> dn = jacobi_elliptic_dn(u, m);

    // Handle case where dn is very small (near pole)
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    if (std::abs(dn) < eps) {
        // Return complex infinity
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    return cn / dn;
}

} // namespace torchscience::kernel::special_functions
