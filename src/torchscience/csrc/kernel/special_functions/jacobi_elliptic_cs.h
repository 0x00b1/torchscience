#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>

#include "jacobi_elliptic_cn.h"
#include "jacobi_elliptic_sn.h"

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function cs(u, m)
//
// Mathematical definition:
// cs(u, m) = cn(u, m) / sn(u, m)
//
// where:
// - cn(u, m) = cos(am(u, m))
// - sn(u, m) = sin(am(u, m))
// - am(u, m) is the Jacobi amplitude function
//
// Domain:
// - u: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - cs(u, 0) = cos(u) / sin(u) = cot(u) (since cn(u,0) = cos(u) and sn(u,0) = sin(u))
// - cs(u, 1) = sech(u) / tanh(u) = 1/sinh(u) = csch(u) (since cn(u,1) = sech(u) and sn(u,1) = tanh(u))
//
// Poles:
// cs has poles where sn(u, m) = 0, i.e., at u = 2nK(m) for integer n,
// where K(m) is the complete elliptic integral of the first kind.
// This includes u = 0.
//
// Periodicity:
// - cs(u + 4K(m), m) = cs(u, m)
//
// Parity:
// - cs(-u, m) = -cs(u, m) (odd function in u)

template <typename T>
T jacobi_elliptic_cs(T u, T m) {
    T cn = jacobi_elliptic_cn(u, m);
    T sn = jacobi_elliptic_sn(u, m);
    return cn / sn;
}

template <typename T>
c10::complex<T> jacobi_elliptic_cs(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> cn = jacobi_elliptic_cn(u, m);
    c10::complex<T> sn = jacobi_elliptic_sn(u, m);
    return cn / sn;
}

} // namespace torchscience::kernel::special_functions
