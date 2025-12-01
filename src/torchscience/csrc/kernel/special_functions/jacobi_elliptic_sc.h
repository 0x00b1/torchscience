#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "jacobi_elliptic_sn.h"
#include "jacobi_elliptic_cn.h"

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function sc(u, m)
//
// Mathematical definition:
// sc(u, m) = sn(u, m) / cn(u, m)
//
// where:
// - sn(u, m) is the Jacobi elliptic sine
// - cn(u, m) is the Jacobi elliptic cosine
//
// Domain:
// - u: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - sc(0, m) = 0 for all m (since sn(0,m) = 0 and cn(0,m) = 1)
// - sc(u, 0) = tan(u) (since sn(u,0) = sin(u) and cn(u,0) = cos(u))
// - sc(u, 1) = sinh(u) (since sn(u,1) = tanh(u) and cn(u,1) = sech(u))
// - sc(-u, m) = -sc(u, m) (odd function in u)
//
// The sc function has poles where cn(u, m) = 0, which occurs at:
// u = (2n+1)K(m) for integer n, where K(m) is the complete elliptic integral
//
// Relationship to other Jacobi functions:
// sc = sn/cn = tan(am(u,m)) where am is the Jacobi amplitude
// sc^2 + 1 = nc^2 (nc = 1/cn)
// sc * cs = 1 (cs = cn/sn)

template <typename T>
T jacobi_elliptic_sc(T u, T m) {
    T sn = jacobi_elliptic_sn(u, m);
    T cn = jacobi_elliptic_cn(u, m);

    // Handle case where cn is very small (near pole)
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    if (std::abs(cn) < eps) {
        // Return signed infinity based on sign of sn
        if (sn > T(0)) {
            return std::numeric_limits<T>::infinity();
        } else if (sn < T(0)) {
            return -std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::quiet_NaN();
        }
    }

    return sn / cn;
}

template <typename T>
c10::complex<T> jacobi_elliptic_sc(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> sn = jacobi_elliptic_sn(u, m);
    c10::complex<T> cn = jacobi_elliptic_cn(u, m);

    // Handle case where cn is very small (near pole)
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    if (std::abs(cn) < eps) {
        // Return complex infinity
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    return sn / cn;
}

} // namespace torchscience::kernel::special_functions
