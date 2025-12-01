#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "modified_bessel_i.h"
#include "modified_bessel_i_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative w.r.t. z: d^2I_n/dz^2
// Using: d/dz[(I_{n-1} + I_{n+1})/2] = [(I_{n-2} + I_n)/2 + (I_n + I_{n+2})/2]/2
//                                    = (I_{n-2} + 2I_n + I_{n+2})/4
template <typename T>
T modified_bessel_i_zz_derivative(T n, T z) {
    T i_nm2 = modified_bessel_i(n - T(2), z);
    T i_n = modified_bessel_i(n, z);
    T i_np2 = modified_bessel_i(n + T(2), z);

    return (i_nm2 + T(2) * i_n + i_np2) / T(4);
}

// Mixed second derivative d^2I_n/dn/dz computed numerically
template <typename T>
T modified_bessel_i_nz_derivative(T n, T z) {
    const T eps = std::sqrt(modified_bessel_i_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    // d/dz I_{n+h} and d/dz I_{n-h}
    T i_p_nm1 = modified_bessel_i(n + h - T(1), z);
    T i_p_np1 = modified_bessel_i(n + h + T(1), z);
    T di_dz_plus = (i_p_nm1 + i_p_np1) / T(2);

    T i_m_nm1 = modified_bessel_i(n - h - T(1), z);
    T i_m_np1 = modified_bessel_i(n - h + T(1), z);
    T di_dz_minus = (i_m_nm1 + i_m_np1) / T(2);

    return (di_dz_plus - di_dz_minus) / (T(2) * h);
}

// Second derivative w.r.t. n: d^2I_n/dn^2 computed numerically
template <typename T>
T modified_bessel_i_nn_derivative(T n, T z) {
    const T eps = std::cbrt(modified_bessel_i_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T i_plus = modified_bessel_i(n + h, z);
    T i_center = modified_bessel_i(n, z);
    T i_minus = modified_bessel_i(n - h, z);

    return (i_plus - T(2) * i_center + i_minus) / (h * h);
}

// Complex versions
template <typename T>
c10::complex<T> modified_bessel_i_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> four(T(4), T(0));

    c10::complex<T> i_nm2 = modified_bessel_i(n - two, z);
    c10::complex<T> i_n = modified_bessel_i(n, z);
    c10::complex<T> i_np2 = modified_bessel_i(n + two, z);

    return (i_nm2 + two * i_n + i_np2) / four;
}

template <typename T>
c10::complex<T> modified_bessel_i_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(modified_bessel_i_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> i_p_nm1 = modified_bessel_i(n + h - one, z);
    c10::complex<T> i_p_np1 = modified_bessel_i(n + h + one, z);
    c10::complex<T> di_dz_plus = (i_p_nm1 + i_p_np1) / two;

    c10::complex<T> i_m_nm1 = modified_bessel_i(n - h - one, z);
    c10::complex<T> i_m_np1 = modified_bessel_i(n - h + one, z);
    c10::complex<T> di_dz_minus = (i_m_nm1 + i_m_np1) / two;

    return (di_dz_plus - di_dz_minus) / (two * h);
}

template <typename T>
c10::complex<T> modified_bessel_i_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(modified_bessel_i_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> i_plus = modified_bessel_i(n + h, z);
    c10::complex<T> i_center = modified_bessel_i(n, z);
    c10::complex<T> i_minus = modified_bessel_i(n - h, z);

    return (i_plus - two * i_center + i_minus) / (h * h);
}

} // namespace detail

// Real backward_backward: returns (grad_grad_output, grad_n, grad_z)
// Computes gradients of the backward pass w.r.t. (grad_output, n, z)
// given upstream gradients (gg_n, gg_z) for the outputs (grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> modified_bessel_i_backward_backward(
    T gg_n,       // upstream gradient for grad_n output
    T gg_z,       // upstream gradient for grad_z output
    T grad_output,
    T n,
    T z
) {
    // Forward backward computes:
    // grad_n = grad_output * dI/dn
    // grad_z = grad_output * (I_{n-1} + I_{n+1})/2

    // We need:
    // d(grad_n)/d(grad_output) = dI/dn
    // d(grad_n)/dn = grad_output * d^2I/dn^2
    // d(grad_n)/dz = grad_output * d^2I/dndz

    // d(grad_z)/d(grad_output) = (I_{n-1} + I_{n+1})/2
    // d(grad_z)/dn = grad_output * d/dn[(I_{n-1} + I_{n+1})/2]
    // d(grad_z)/dz = grad_output * d^2I/dz^2 = grad_output * (I_{n-2} + 2I_n + I_{n+2})/4

    // First derivatives
    T i_nm1 = modified_bessel_i(n - T(1), z);
    T i_np1 = modified_bessel_i(n + T(1), z);
    T di_dz = (i_nm1 + i_np1) / T(2);

    T di_dn = detail::modified_bessel_i_n_derivative(n, z);

    // Second derivatives
    T d2i_dz2 = detail::modified_bessel_i_zz_derivative(n, z);
    T d2i_dn2 = detail::modified_bessel_i_nn_derivative(n, z);
    T d2i_dndz = detail::modified_bessel_i_nz_derivative(n, z);

    // d(grad_z)/dn: need d/dn[(I_{n-1} + I_{n+1})/2]
    // This is approximated by d^2I/dndz
    T d_dz_dn = d2i_dndz;

    // Accumulate gradients
    // grad_grad_output = gg_n * dI/dn + gg_z * dI/dz
    T grad_grad_output = gg_n * di_dn + gg_z * di_dz;

    // grad_n = gg_n * grad_output * d^2I/dn^2 + gg_z * grad_output * d^2I/dndz
    T grad_n = grad_output * (gg_n * d2i_dn2 + gg_z * d_dz_dn);

    // grad_z = gg_n * grad_output * d^2I/dndz + gg_z * grad_output * d^2I/dz^2
    T grad_z = grad_output * (gg_n * d2i_dndz + gg_z * d2i_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> modified_bessel_i_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    // First derivatives
    c10::complex<T> i_nm1 = modified_bessel_i(n - one, z);
    c10::complex<T> i_np1 = modified_bessel_i(n + one, z);
    c10::complex<T> di_dz = (i_nm1 + i_np1) / two;

    c10::complex<T> di_dn = detail::modified_bessel_i_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2i_dz2 = detail::modified_bessel_i_zz_derivative(n, z);
    c10::complex<T> d2i_dn2 = detail::modified_bessel_i_nn_derivative(n, z);
    c10::complex<T> d2i_dndz = detail::modified_bessel_i_nz_derivative(n, z);

    // Accumulate gradients (using conjugates for Wirtinger derivatives)
    c10::complex<T> grad_grad_output = gg_n * std::conj(di_dn) + gg_z * std::conj(di_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2i_dn2) + gg_z * std::conj(d2i_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2i_dndz) + gg_z * std::conj(d2i_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
