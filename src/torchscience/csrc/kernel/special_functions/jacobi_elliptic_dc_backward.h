#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "jacobi_elliptic_dc.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Jacobi elliptic function dc(u, m)
//
// Gradients computed using numerical differentiation.
//
// dc(u, m) = dn(u, m) / cn(u, m)
//
// The analytical derivative is:
// ∂dc/∂u = (cn * ∂dn/∂u - dn * ∂cn/∂u) / cn^2
//        = (cn * (-m * sn * cn / dn) - dn * (-sn * dn)) / cn^2
//        = (-m * sn * cn^2 / dn + sn * dn^2) / cn^2
//        = sn * (dn^2 - m * cn^2 / dn) / cn^2
//
// For robustness, we use numerical differentiation.

namespace detail {

template <typename T>
T jacobi_elliptic_dc_du(T u, T m) {
    // Use 5-point stencil for numerical derivative
    const T h_rel = std::cbrt(std::numeric_limits<T>::epsilon());
    T h = h_rel * std::max(static_cast<T>(std::abs(u)), T(1));
    h = std::max(h, T(1e-8));

    // 5-point stencil: f'(x) = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    T f_p2h = jacobi_elliptic_dc(u + T(2) * h, m);
    T f_ph = jacobi_elliptic_dc(u + h, m);
    T f_mh = jacobi_elliptic_dc(u - h, m);
    T f_m2h = jacobi_elliptic_dc(u - T(2) * h, m);

    return (-f_p2h + T(8) * f_ph - T(8) * f_mh + f_m2h) / (T(12) * h);
}

template <typename T>
c10::complex<T> jacobi_elliptic_dc_du(c10::complex<T> u, c10::complex<T> m) {
    const T h_rel = std::cbrt(std::numeric_limits<T>::epsilon());
    T h = h_rel * std::max(static_cast<T>(std::abs(u)), T(1));
    h = std::max(h, T(1e-8));
    c10::complex<T> ch(h, T(0));

    c10::complex<T> f_p2h = jacobi_elliptic_dc(u + c10::complex<T>(T(2), T(0)) * ch, m);
    c10::complex<T> f_ph = jacobi_elliptic_dc(u + ch, m);
    c10::complex<T> f_mh = jacobi_elliptic_dc(u - ch, m);
    c10::complex<T> f_m2h = jacobi_elliptic_dc(u - c10::complex<T>(T(2), T(0)) * ch, m);

    c10::complex<T> eight(T(8), T(0));
    c10::complex<T> twelve(T(12), T(0));

    return (-f_p2h + eight * f_ph - eight * f_mh + f_m2h) / (twelve * ch);
}

template <typename T>
T jacobi_elliptic_dc_dm(T u, T m) {
    const T h_rel = std::cbrt(std::numeric_limits<T>::epsilon());
    T h = h_rel * std::max(static_cast<T>(std::abs(m)), T(0.1));

    // Clamp h to stay within valid domain [0, 1]
    if (m < T(0.5)) {
        h = std::min(h, m / T(2));
        h = std::min(h, (T(1) - m) / T(2));
    } else {
        h = std::min(h, (T(1) - m) / T(2));
        h = std::min(h, m / T(2));
    }

    h = std::max(h, T(1e-8));

    T f_p2h = jacobi_elliptic_dc(u, m + T(2) * h);
    T f_ph = jacobi_elliptic_dc(u, m + h);
    T f_mh = jacobi_elliptic_dc(u, m - h);
    T f_m2h = jacobi_elliptic_dc(u, m - T(2) * h);

    return (-f_p2h + T(8) * f_ph - T(8) * f_mh + f_m2h) / (T(12) * h);
}

template <typename T>
c10::complex<T> jacobi_elliptic_dc_dm(c10::complex<T> u, c10::complex<T> m) {
    const T h_rel = std::cbrt(std::numeric_limits<T>::epsilon());
    T h = h_rel * std::max(static_cast<T>(std::abs(m)), T(0.1));
    h = std::max(h, T(1e-8));
    c10::complex<T> ch(h, T(0));

    c10::complex<T> f_p2h = jacobi_elliptic_dc(u, m + c10::complex<T>(T(2), T(0)) * ch);
    c10::complex<T> f_ph = jacobi_elliptic_dc(u, m + ch);
    c10::complex<T> f_mh = jacobi_elliptic_dc(u, m - ch);
    c10::complex<T> f_m2h = jacobi_elliptic_dc(u, m - c10::complex<T>(T(2), T(0)) * ch);

    c10::complex<T> eight(T(8), T(0));
    c10::complex<T> twelve(T(12), T(0));

    return (-f_p2h + eight * f_ph - eight * f_mh + f_m2h) / (twelve * ch);
}

} // namespace detail

template <typename T>
std::tuple<T, T> jacobi_elliptic_dc_backward(T gradient, T u, T m) {
    T ddc_du = detail::jacobi_elliptic_dc_du(u, m);
    T ddc_dm = detail::jacobi_elliptic_dc_dm(u, m);

    return {gradient * ddc_du, gradient * ddc_dm};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>>
jacobi_elliptic_dc_backward(c10::complex<T> gradient, c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> ddc_du = detail::jacobi_elliptic_dc_du(u, m);
    c10::complex<T> ddc_dm = detail::jacobi_elliptic_dc_dm(u, m);

    // For complex inputs with Wirtinger derivatives
    return {gradient * std::conj(ddc_du), gradient * std::conj(ddc_dm)};
}

} // namespace torchscience::kernel::special_functions
