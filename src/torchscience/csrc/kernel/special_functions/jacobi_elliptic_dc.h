#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>

#include "jacobi_elliptic_cn.h"
#include "jacobi_elliptic_dn.h"

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function dc(u, m)
//
// Mathematical definition:
// dc(u, m) = dn(u, m) / cn(u, m)
//
// where:
// - dn(u, m) = sqrt(1 - m * sn^2(u, m))
// - cn(u, m) = cos(am(u, m))
// - am(u, m) is the Jacobi amplitude function
//
// Domain:
// - u: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - dc(0, m) = 1 for all m (since dn(0,m) = 1 and cn(0,m) = 1)
// - dc(u, 0) = 1 / cos(u) = sec(u) (since dn(u,0) = 1 and cn(u,0) = cos(u))
// - dc(u, 1) = 1 (since dn(u,1) = cn(u,1) = sech(u))
//
// Poles:
// dc has poles where cn(u, m) = 0, i.e., at u = (2n+1)K(m) for integer n,
// where K(m) is the complete elliptic integral of the first kind.
//
// Periodicity:
// - dc(u + 4K(m), m) = dc(u, m)

template <typename T>
T jacobi_elliptic_dc(T u, T m) {
    T dn = jacobi_elliptic_dn(u, m);
    T cn = jacobi_elliptic_cn(u, m);
    return dn / cn;
}

template <typename T>
c10::complex<T> jacobi_elliptic_dc(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> dn = jacobi_elliptic_dn(u, m);
    c10::complex<T> cn = jacobi_elliptic_cn(u, m);
    return dn / cn;
}

} // namespace torchscience::kernel::special_functions
