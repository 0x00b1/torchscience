#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "modified_bessel_k.h"
#include "modified_bessel_k_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative w.r.t. z: d^2K_n/dz^2
// Using: d/dz[-(K_{n-1} + K_{n+1})/2] = (K_{n-2} + 2K_n + K_{n+2})/4
// Note: d/dz K_m(z) = -(K_{m-1}(z) + K_{m+1}(z))/2
// So: d/dz K_{n-1} = -(K_{n-2} + K_n)/2
// And: d/dz K_{n+1} = -(K_n + K_{n+2})/2
// Therefore: d^2K_n/dz^2 = (1/2)[(K_{n-2} + K_n)/2 + (K_n + K_{n+2})/2]
//                        = (K_{n-2} + 2K_n + K_{n+2})/4
template <typename T>
T modified_bessel_k_zz_derivative(T n, T z) {
    T k_nm2 = modified_bessel_k(n - T(2), z);
    T k_n = modified_bessel_k(n, z);
    T k_np2 = modified_bessel_k(n + T(2), z);

    return (k_nm2 + T(2) * k_n + k_np2) / T(4);
}

// Mixed second derivative d^2K_n/dn/dz computed numerically
template <typename T>
T modified_bessel_k_nz_derivative(T n, T z) {
    const T eps = std::sqrt(modified_bessel_k_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    // d/dz K_{n+h} and d/dz K_{n-h}
    T k_p_nm1 = modified_bessel_k(n + h - T(1), z);
    T k_p_np1 = modified_bessel_k(n + h + T(1), z);
    T dk_dz_plus = -(k_p_nm1 + k_p_np1) / T(2);

    T k_m_nm1 = modified_bessel_k(n - h - T(1), z);
    T k_m_np1 = modified_bessel_k(n - h + T(1), z);
    T dk_dz_minus = -(k_m_nm1 + k_m_np1) / T(2);

    return (dk_dz_plus - dk_dz_minus) / (T(2) * h);
}

// Second derivative w.r.t. n: d^2K_n/dn^2 computed numerically
template <typename T>
T modified_bessel_k_nn_derivative(T n, T z) {
    const T eps = std::cbrt(modified_bessel_k_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T k_plus = modified_bessel_k(n + h, z);
    T k_center = modified_bessel_k(n, z);
    T k_minus = modified_bessel_k(n - h, z);

    return (k_plus - T(2) * k_center + k_minus) / (h * h);
}

// Complex versions
template <typename T>
c10::complex<T> modified_bessel_k_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> four(T(4), T(0));

    c10::complex<T> k_nm2 = modified_bessel_k(n - two, z);
    c10::complex<T> k_n = modified_bessel_k(n, z);
    c10::complex<T> k_np2 = modified_bessel_k(n + two, z);

    return (k_nm2 + two * k_n + k_np2) / four;
}

template <typename T>
c10::complex<T> modified_bessel_k_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(modified_bessel_k_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> k_p_nm1 = modified_bessel_k(n + h - one, z);
    c10::complex<T> k_p_np1 = modified_bessel_k(n + h + one, z);
    c10::complex<T> dk_dz_plus = -(k_p_nm1 + k_p_np1) / two;

    c10::complex<T> k_m_nm1 = modified_bessel_k(n - h - one, z);
    c10::complex<T> k_m_np1 = modified_bessel_k(n - h + one, z);
    c10::complex<T> dk_dz_minus = -(k_m_nm1 + k_m_np1) / two;

    return (dk_dz_plus - dk_dz_minus) / (two * h);
}

template <typename T>
c10::complex<T> modified_bessel_k_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(modified_bessel_k_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> k_plus = modified_bessel_k(n + h, z);
    c10::complex<T> k_center = modified_bessel_k(n, z);
    c10::complex<T> k_minus = modified_bessel_k(n - h, z);

    return (k_plus - two * k_center + k_minus) / (h * h);
}

} // namespace detail

// Real backward_backward: returns (grad_grad_output, grad_n, grad_z)
// Computes gradients of the backward pass w.r.t. (grad_output, n, z)
// given upstream gradients (gg_n, gg_z) for the outputs (grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> modified_bessel_k_backward_backward(
    T gg_n,       // upstream gradient for grad_n output
    T gg_z,       // upstream gradient for grad_z output
    T grad_output,
    T n,
    T z
) {
    // Forward backward computes:
    // grad_n = grad_output * dK/dn
    // grad_z = grad_output * -(K_{n-1} + K_{n+1})/2

    // We need:
    // d(grad_n)/d(grad_output) = dK/dn
    // d(grad_n)/dn = grad_output * d^2K/dn^2
    // d(grad_n)/dz = grad_output * d^2K/dndz

    // d(grad_z)/d(grad_output) = -(K_{n-1} + K_{n+1})/2
    // d(grad_z)/dn = grad_output * d/dn[-(K_{n-1} + K_{n+1})/2]
    // d(grad_z)/dz = grad_output * d^2K/dz^2 = grad_output * (K_{n-2} + 2K_n + K_{n+2})/4

    // First derivatives
    T k_nm1 = modified_bessel_k(n - T(1), z);
    T k_np1 = modified_bessel_k(n + T(1), z);
    T dk_dz = -(k_nm1 + k_np1) / T(2);

    T dk_dn = detail::modified_bessel_k_n_derivative(n, z);

    // Second derivatives
    T d2k_dz2 = detail::modified_bessel_k_zz_derivative(n, z);
    T d2k_dn2 = detail::modified_bessel_k_nn_derivative(n, z);
    T d2k_dndz = detail::modified_bessel_k_nz_derivative(n, z);

    // d(grad_z)/dn: need d/dn[-(K_{n-1} + K_{n+1})/2]
    // This equals -(dK_{n-1}/dn + dK_{n+1}/dn)/2
    // We approximate with d^2K/dndz
    T d_dz_dn = d2k_dndz;

    // Accumulate gradients
    // grad_grad_output = gg_n * dK/dn + gg_z * dK/dz
    T grad_grad_output = gg_n * dk_dn + gg_z * dk_dz;

    // grad_n = gg_n * grad_output * d^2K/dn^2 + gg_z * grad_output * d^2K/dndz
    T grad_n = grad_output * (gg_n * d2k_dn2 + gg_z * d_dz_dn);

    // grad_z = gg_n * grad_output * d^2K/dndz + gg_z * grad_output * d^2K/dz^2
    T grad_z = grad_output * (gg_n * d2k_dndz + gg_z * d2k_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> modified_bessel_k_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    // First derivatives
    c10::complex<T> k_nm1 = modified_bessel_k(n - one, z);
    c10::complex<T> k_np1 = modified_bessel_k(n + one, z);
    c10::complex<T> dk_dz = -(k_nm1 + k_np1) / two;

    c10::complex<T> dk_dn = detail::modified_bessel_k_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2k_dz2 = detail::modified_bessel_k_zz_derivative(n, z);
    c10::complex<T> d2k_dn2 = detail::modified_bessel_k_nn_derivative(n, z);
    c10::complex<T> d2k_dndz = detail::modified_bessel_k_nz_derivative(n, z);

    // Accumulate gradients (using conjugates for Wirtinger derivatives)
    c10::complex<T> grad_grad_output = gg_n * std::conj(dk_dn) + gg_z * std::conj(dk_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2k_dn2) + gg_z * std::conj(d2k_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2k_dndz) + gg_z * std::conj(d2k_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
