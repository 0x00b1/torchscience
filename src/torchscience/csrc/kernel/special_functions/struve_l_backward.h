#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "struve_l.h"
#include "gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Numerical derivative of L_n(z) with respect to n
template <typename T>
T struve_l_n_derivative(T n, T z) {
    const T eps = std::sqrt(struve_l_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T l_plus = struve_l(n + h, z);
    T l_minus = struve_l(n - h, z);

    return (l_plus - l_minus) / (T(2) * h);
}

// Complex version
template <typename T>
c10::complex<T> struve_l_n_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(struve_l_eps<T>());
    const c10::complex<T> h_c(eps, T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h = (n_mag > T(1)) ? h_c * c10::complex<T>(n_mag, T(0)) : h_c;

    c10::complex<T> l_plus = struve_l(n + h, z);
    c10::complex<T> l_minus = struve_l(n - h, z);

    return (l_plus - l_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Backward pass for struve_l
// Returns (grad_n, grad_z)
//
// The derivative with respect to z is:
// dL_n/dz = L_{n-1}(z) - (n/z) * L_n(z)   for z != 0
// (Same formula as H_n, verified from struve_l_1_backward: dL_1/dz = L_0 - L_1/z)
//
// For n=0: dL_0/dz = L_{-1}(z) = (2/pi) + L_1(z)  (verified from struve_l_0_backward)
//
// The derivative with respect to n is computed numerically
template <typename T>
std::tuple<T, T> struve_l_backward(T grad_output, T n, T z) {
    const T pi = static_cast<T>(M_PI);

    // Gradient w.r.t. z
    // dL_n/dz = L_{n-1}(z) - (n/z) * L_n(z)
    T l_nm1 = struve_l(n - T(1), z);
    T l_n = struve_l(n, z);

    T dl_dz;
    if (z == T(0)) {
        // At z=0, L_n(0) = 0 for n >= -1
        // For n=0: dL_0/dz |_{z=0} = 2/pi
        // For n>0: dL_n/dz |_{z=0} = 0
        if (n == T(0)) {
            dl_dz = T(2) / pi;
        } else if (n > T(0)) {
            dl_dz = T(0);
        } else {
            dl_dz = T(0);
        }
    } else {
        // MINUS sign for (n/z)*L_n (same as H_n)
        dl_dz = l_nm1 - (n / z) * l_n;
    }

    T grad_z = grad_output * dl_dz;

    // Gradient w.r.t. n (numerical)
    T dl_dn = detail::struve_l_n_derivative(n, z);
    T grad_n = grad_output * dl_dn;

    return {grad_n, grad_z};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> struve_l_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> zero(T(0), T(0));
    const T pi = static_cast<T>(M_PI);
    const T eps = detail::struve_l_eps<T>();

    // Gradient w.r.t. z
    // dL_n/dz = L_{n-1}(z) - (n/z) * L_n(z)
    c10::complex<T> l_nm1 = struve_l(n - one, z);
    c10::complex<T> l_n = struve_l(n, z);

    c10::complex<T> dl_dz;
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            dl_dz = two / c10::complex<T>(pi, T(0));
        } else {
            dl_dz = zero;
        }
    } else {
        // MINUS sign for (n/z)*L_n (same as H_n)
        dl_dz = l_nm1 - (n / z) * l_n;
    }

    c10::complex<T> grad_z = grad_output * std::conj(dl_dz);

    // Gradient w.r.t. n (numerical)
    c10::complex<T> dl_dn = detail::struve_l_n_derivative(n, z);
    c10::complex<T> grad_n = grad_output * std::conj(dl_dn);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
