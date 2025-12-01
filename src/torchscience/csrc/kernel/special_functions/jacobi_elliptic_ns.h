#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "jacobi_elliptic_sn.h"

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function ns(u, m)
//
// Mathematical definition:
// ns(u, m) = 1 / sn(u, m)
//
// where:
// - sn(u, m) is the Jacobi elliptic function sine amplitude
//
// Domain:
// - u: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - ns(0, m) has a pole (sn(0, m) = 0)
// - ns(u, 0) = csc(u) = 1/sin(u) for all u
// - ns(u, 1) = coth(u)
// - ns(K(m), m) = 1 where K(m) is the complete elliptic integral
//
// The function is periodic with period 4K(m) where K is the complete
// elliptic integral of the first kind.

template <typename T>
T jacobi_elliptic_ns(T u, T m) {
    T sn = jacobi_elliptic_sn(u, m);
    return T(1) / sn;
}

template <typename T>
c10::complex<T> jacobi_elliptic_ns(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> sn = jacobi_elliptic_sn(u, m);
    c10::complex<T> one(T(1), T(0));
    return one / sn;
}

} // namespace torchscience::kernel::special_functions
