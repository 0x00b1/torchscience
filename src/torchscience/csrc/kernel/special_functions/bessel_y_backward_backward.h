#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "bessel_y.h"
#include "bessel_y_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative w.r.t. z: d^2Y_n/dz^2
// Using: d/dz[(Y_{n-1} - Y_{n+1})/2] = (Y_{n-2} - 2Y_n + Y_{n+2})/4
template <typename T>
T bessel_y_zz_derivative(T n, T z) {
    T y_nm2 = bessel_y(n - T(2), z);
    T y_n = bessel_y(n, z);
    T y_np2 = bessel_y(n + T(2), z);

    return (y_nm2 - T(2) * y_n + y_np2) / T(4);
}

// Mixed second derivative d^2Y_n/dn/dz computed numerically
template <typename T>
T bessel_y_nz_derivative(T n, T z) {
    const T eps = std::sqrt(bessel_y_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    // d/dz Y_{n+h} and d/dz Y_{n-h}
    T y_p_nm1 = bessel_y(n + h - T(1), z);
    T y_p_np1 = bessel_y(n + h + T(1), z);
    T dy_dz_plus = (y_p_nm1 - y_p_np1) / T(2);

    T y_m_nm1 = bessel_y(n - h - T(1), z);
    T y_m_np1 = bessel_y(n - h + T(1), z);
    T dy_dz_minus = (y_m_nm1 - y_m_np1) / T(2);

    return (dy_dz_plus - dy_dz_minus) / (T(2) * h);
}

// Second derivative w.r.t. n: d^2Y_n/dn^2 computed numerically
template <typename T>
T bessel_y_nn_derivative(T n, T z) {
    const T eps = std::cbrt(bessel_y_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T y_plus = bessel_y(n + h, z);
    T y_center = bessel_y(n, z);
    T y_minus = bessel_y(n - h, z);

    return (y_plus - T(2) * y_center + y_minus) / (h * h);
}

// Complex versions
template <typename T>
c10::complex<T> bessel_y_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> four(T(4), T(0));

    c10::complex<T> y_nm2 = bessel_y(n - two, z);
    c10::complex<T> y_n = bessel_y(n, z);
    c10::complex<T> y_np2 = bessel_y(n + two, z);

    return (y_nm2 - two * y_n + y_np2) / four;
}

template <typename T>
c10::complex<T> bessel_y_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(bessel_y_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> y_p_nm1 = bessel_y(n + h - one, z);
    c10::complex<T> y_p_np1 = bessel_y(n + h + one, z);
    c10::complex<T> dy_dz_plus = (y_p_nm1 - y_p_np1) / two;

    c10::complex<T> y_m_nm1 = bessel_y(n - h - one, z);
    c10::complex<T> y_m_np1 = bessel_y(n - h + one, z);
    c10::complex<T> dy_dz_minus = (y_m_nm1 - y_m_np1) / two;

    return (dy_dz_plus - dy_dz_minus) / (two * h);
}

template <typename T>
c10::complex<T> bessel_y_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(bessel_y_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> y_plus = bessel_y(n + h, z);
    c10::complex<T> y_center = bessel_y(n, z);
    c10::complex<T> y_minus = bessel_y(n - h, z);

    return (y_plus - two * y_center + y_minus) / (h * h);
}

} // namespace detail

// Real backward_backward: returns (grad_grad_output, grad_n, grad_z)
// Computes gradients of the backward pass w.r.t. (grad_output, n, z)
// given upstream gradients (gg_n, gg_z) for the outputs (grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> bessel_y_backward_backward(
    T gg_n,       // upstream gradient for grad_n output
    T gg_z,       // upstream gradient for grad_z output
    T grad_output,
    T n,
    T z
) {
    // Forward backward computes:
    // grad_n = grad_output * dY/dn
    // grad_z = grad_output * (Y_{n-1} - Y_{n+1})/2

    // We need:
    // d(grad_n)/d(grad_output) = dY/dn
    // d(grad_n)/dn = grad_output * d^2Y/dn^2
    // d(grad_n)/dz = grad_output * d^2Y/dndz

    // d(grad_z)/d(grad_output) = (Y_{n-1} - Y_{n+1})/2
    // d(grad_z)/dn = grad_output * d/dn[(Y_{n-1} - Y_{n+1})/2]
    // d(grad_z)/dz = grad_output * d^2Y/dz^2 = grad_output * (Y_{n-2} - 2Y_n + Y_{n+2})/4

    // First derivatives
    T y_nm1 = bessel_y(n - T(1), z);
    T y_np1 = bessel_y(n + T(1), z);
    T dy_dz = (y_nm1 - y_np1) / T(2);

    T dy_dn = detail::bessel_y_n_derivative(n, z);

    // Second derivatives
    T d2y_dz2 = detail::bessel_y_zz_derivative(n, z);
    T d2y_dn2 = detail::bessel_y_nn_derivative(n, z);
    T d2y_dndz = detail::bessel_y_nz_derivative(n, z);

    // d(grad_z)/dn: need d/dn[(Y_{n-1} - Y_{n+1})/2]
    // This equals (dY_{n-1}/dn - dY_{n+1}/dn)/2
    // We approximate with d^2Y/dndz
    T d_dz_dn = d2y_dndz;

    // Accumulate gradients
    // grad_grad_output = gg_n * dY/dn + gg_z * dY/dz
    T grad_grad_output = gg_n * dy_dn + gg_z * dy_dz;

    // grad_n = gg_n * grad_output * d^2Y/dn^2 + gg_z * grad_output * d^2Y/dndz
    T grad_n = grad_output * (gg_n * d2y_dn2 + gg_z * d_dz_dn);

    // grad_z = gg_n * grad_output * d^2Y/dndz + gg_z * grad_output * d^2Y/dz^2
    T grad_z = grad_output * (gg_n * d2y_dndz + gg_z * d2y_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> bessel_y_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    // First derivatives
    c10::complex<T> y_nm1 = bessel_y(n - one, z);
    c10::complex<T> y_np1 = bessel_y(n + one, z);
    c10::complex<T> dy_dz = (y_nm1 - y_np1) / two;

    c10::complex<T> dy_dn = detail::bessel_y_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2y_dz2 = detail::bessel_y_zz_derivative(n, z);
    c10::complex<T> d2y_dn2 = detail::bessel_y_nn_derivative(n, z);
    c10::complex<T> d2y_dndz = detail::bessel_y_nz_derivative(n, z);

    // Accumulate gradients (using conjugates for Wirtinger derivatives)
    c10::complex<T> grad_grad_output = gg_n * std::conj(dy_dn) + gg_z * std::conj(dy_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2y_dn2) + gg_z * std::conj(d2y_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2y_dndz) + gg_z * std::conj(d2y_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
