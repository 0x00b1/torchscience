#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "jacobi_elliptic_dn.h"

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function nd(u, m)
//
// Mathematical definition:
// nd(u, m) = 1 / dn(u, m)
//
// where:
// - dn(u, m) is the Jacobi elliptic function delta amplitude
//
// Domain:
// - u: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - nd(0, m) = 1 for all m
// - nd(u, 0) = 1 for all u
// - nd(u, 1) = cosh(u)
// - nd(K(m), m) = 1/sqrt(1-m) where K(m) is the complete elliptic integral
//
// The function is periodic with period 2K(m) where K is the complete
// elliptic integral of the first kind.

template <typename T>
T jacobi_elliptic_nd(T u, T m) {
    T dn = jacobi_elliptic_dn(u, m);
    return T(1) / dn;
}

template <typename T>
c10::complex<T> jacobi_elliptic_nd(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> dn = jacobi_elliptic_dn(u, m);
    c10::complex<T> one(T(1), T(0));
    return one / dn;
}

} // namespace torchscience::kernel::special_functions
