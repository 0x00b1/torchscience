#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "struve_h.h"
#include "gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Numerical derivative of H_n(z) with respect to n
// The analytical formula is complex, so we use finite differences
template <typename T>
T struve_h_n_derivative(T n, T z) {
    const T eps = std::sqrt(struve_h_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T h_plus = struve_h(n + h, z);
    T h_minus = struve_h(n - h, z);

    return (h_plus - h_minus) / (T(2) * h);
}

// Complex version
template <typename T>
c10::complex<T> struve_h_n_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(struve_h_eps<T>());
    const c10::complex<T> h_c(eps, T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h = (n_mag > T(1)) ? h_c * c10::complex<T>(n_mag, T(0)) : h_c;

    c10::complex<T> h_plus = struve_h(n + h, z);
    c10::complex<T> h_minus = struve_h(n - h, z);

    return (h_plus - h_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Backward pass for struve_h
// Returns (grad_n, grad_z)
//
// The derivative with respect to z is:
// dH_n/dz = H_{n-1}(z) - (n/z) * H_n(z)   for z != 0
// Or equivalently using DLMF 11.4.27:
// dH_n/dz = (z/2)^n / [sqrt(pi) * Gamma(n+3/2)] - H_{n+1}(z) + (n/z) * H_n(z)
//
// Simpler form:
// dH_n/dz = H_{n-1}(z) - (n/z) * H_n(z)
//
// For n=0: dH_0/dz = H_{-1}(z) = (2/pi) - H_1(z)  (since H_{-1}(z) = (2/pi)*cos(...))
//   Actually, simpler: dH_0/dz = 2/pi - H_1(z) (verified from struve_h_0_backward)
//
// The derivative with respect to n is computed numerically
template <typename T>
std::tuple<T, T> struve_h_backward(T grad_output, T n, T z) {
    const T pi = static_cast<T>(M_PI);

    // Gradient w.r.t. z
    // dH_n/dz = H_{n-1}(z) - (n/z) * H_n(z)
    T h_nm1 = struve_h(n - T(1), z);
    T h_n = struve_h(n, z);

    T dh_dz;
    if (z == T(0)) {
        // At z=0, H_n(0) = 0 for n >= -1
        // The derivative depends on n:
        // For n=0: dH_0/dz |_{z=0} = 2/pi
        // For n>0: dH_n/dz |_{z=0} = 0 (from series)
        if (n == T(0)) {
            dh_dz = T(2) / pi;
        } else if (n > T(0)) {
            dh_dz = T(0);
        } else {
            // For n < 0, need to evaluate limit carefully
            // H_{n-1}(0) - (n/z)*H_n(0) with both H_n(0)=0 and H_{n-1}(0)=0
            dh_dz = T(0);
        }
    } else {
        dh_dz = h_nm1 - (n / z) * h_n;
    }

    T grad_z = grad_output * dh_dz;

    // Gradient w.r.t. n (numerical)
    T dh_dn = detail::struve_h_n_derivative(n, z);
    T grad_n = grad_output * dh_dn;

    return {grad_n, grad_z};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> struve_h_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> zero(T(0), T(0));
    const T pi = static_cast<T>(M_PI);
    const T eps = detail::struve_h_eps<T>();

    // Gradient w.r.t. z
    // dH_n/dz = H_{n-1}(z) - (n/z) * H_n(z)
    c10::complex<T> h_nm1 = struve_h(n - one, z);
    c10::complex<T> h_n = struve_h(n, z);

    c10::complex<T> dh_dz;
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            dh_dz = two / c10::complex<T>(pi, T(0));
        } else {
            dh_dz = zero;
        }
    } else {
        dh_dz = h_nm1 - (n / z) * h_n;
    }

    c10::complex<T> grad_z = grad_output * std::conj(dh_dz);

    // Gradient w.r.t. n (numerical)
    c10::complex<T> dh_dn = detail::struve_h_n_derivative(n, z);
    c10::complex<T> grad_n = grad_output * std::conj(dh_dn);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
