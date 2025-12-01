#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>

#include "jacobi_elliptic_dn.h"
#include "jacobi_elliptic_sn.h"

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function ds(u, m)
//
// Mathematical definition:
// ds(u, m) = dn(u, m) / sn(u, m)
//
// where:
// - dn(u, m) = sqrt(1 - m * sn^2(u, m))
// - sn(u, m) = sin(am(u, m))
// - am(u, m) is the Jacobi amplitude function
//
// Domain:
// - u: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - ds(u, 0) = 1 / sin(u) = csc(u) (since dn(u,0) = 1 and sn(u,0) = sin(u))
// - ds(u, 1) = 1 / sinh(u) = csch(u) (since dn(u,1) = sech(u) and sn(u,1) = tanh(u))
//
// Poles:
// ds has poles where sn(u, m) = 0, i.e., at u = 2nK(m) for integer n,
// where K(m) is the complete elliptic integral of the first kind.
// This includes u = 0.
//
// Periodicity:
// - ds(u + 4K(m), m) = ds(u, m)
//
// Parity:
// - ds(-u, m) = -ds(u, m) (odd function in u)

template <typename T>
T jacobi_elliptic_ds(T u, T m) {
    T dn = jacobi_elliptic_dn(u, m);
    T sn = jacobi_elliptic_sn(u, m);
    return dn / sn;
}

template <typename T>
c10::complex<T> jacobi_elliptic_ds(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> dn = jacobi_elliptic_dn(u, m);
    c10::complex<T> sn = jacobi_elliptic_sn(u, m);
    return dn / sn;
}

} // namespace torchscience::kernel::special_functions
