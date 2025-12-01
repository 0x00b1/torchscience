#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "jacobi_elliptic_cn.h"

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function nc(u, m)
//
// Mathematical definition:
// nc(u, m) = 1 / cn(u, m)
//
// where:
// - cn(u, m) is the Jacobi elliptic function cosine amplitude
//
// Domain:
// - u: real or complex argument
// - m: elliptic parameter (0 <= m <= 1 for standard real case)
//
// Special values:
// - nc(0, m) = 1 for all m
// - nc(u, 0) = sec(u) = 1/cos(u) for all u
// - nc(u, 1) = cosh(u)
// - nc has poles at zeros of cn(u, m)
//
// The function is periodic with period 4K(m) where K is the complete
// elliptic integral of the first kind.

template <typename T>
T jacobi_elliptic_nc(T u, T m) {
    T cn = jacobi_elliptic_cn(u, m);
    return T(1) / cn;
}

template <typename T>
c10::complex<T> jacobi_elliptic_nc(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> cn = jacobi_elliptic_cn(u, m);
    c10::complex<T> one(T(1), T(0));
    return one / cn;
}

} // namespace torchscience::kernel::special_functions
