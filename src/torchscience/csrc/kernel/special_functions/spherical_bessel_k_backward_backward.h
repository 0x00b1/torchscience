#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "spherical_bessel_k.h"
#include "spherical_bessel_k_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative w.r.t. z: d^2/dz^2 k_n(z)
// Using: d/dz[-k_{n-1} - ((n+1)/z)*k_n] = -dk_{n-1}/dz + ((n+1)/z^2)*k_n - ((n+1)/z)*dk_n/dz
// where dk_n/dz = -k_{n-1} - ((n+1)/z)*k_n
// and dk_{n-1}/dz = -k_{n-2} - (n/z)*k_{n-1}
template <typename T>
T spherical_bessel_k_zz_derivative(T n, T z) {
    const T eps = spherical_bessel_k_eps<T>();

    if (std::abs(z) < eps) {
        // For small z, k_n and its derivatives are singular
        return -std::numeric_limits<T>::infinity();
    }

    T k_n = spherical_bessel_k(n, z);
    T k_nm1 = spherical_bessel_k(n - T(1), z);
    T k_nm2 = spherical_bessel_k(n - T(2), z);

    // d/dz k_n = -k_{n-1} - ((n+1)/z) * k_n
    T dk_n_dz = -k_nm1 - ((n + T(1)) / z) * k_n;

    // d/dz k_{n-1} = -k_{n-2} - (n/z) * k_{n-1}
    T dk_nm1_dz = -k_nm2 - (n / z) * k_nm1;

    // d^2/dz^2 k_n = d/dz[-k_{n-1} - ((n+1)/z)*k_n]
    //             = -dk_{n-1}/dz + ((n+1)/z^2)*k_n - ((n+1)/z)*dk_n/dz
    T z2 = z * z;
    return -dk_nm1_dz + ((n + T(1)) / z2) * k_n - ((n + T(1)) / z) * dk_n_dz;
}

// Mixed second derivative d^2/(dn dz) k_n(z) computed numerically
template <typename T>
T spherical_bessel_k_nz_derivative(T n, T z) {
    const T eps = std::sqrt(spherical_bessel_k_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    // Compute d/dz k_{n+h} and d/dz k_{n-h}
    T k_p = spherical_bessel_k(n + h, z);
    T k_p_m1 = spherical_bessel_k(n + h - T(1), z);
    T dk_dz_plus = -k_p_m1 - ((n + h + T(1)) / z) * k_p;

    T k_m = spherical_bessel_k(n - h, z);
    T k_m_m1 = spherical_bessel_k(n - h - T(1), z);
    T dk_dz_minus = -k_m_m1 - ((n - h + T(1)) / z) * k_m;

    return (dk_dz_plus - dk_dz_minus) / (T(2) * h);
}

// Second derivative w.r.t. n: d^2/dn^2 k_n(z) computed numerically
template <typename T>
T spherical_bessel_k_nn_derivative(T n, T z) {
    const T eps = std::cbrt(spherical_bessel_k_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T k_plus = spherical_bessel_k(n + h, z);
    T k_center = spherical_bessel_k(n, z);
    T k_minus = spherical_bessel_k(n - h, z);

    return (k_plus - T(2) * k_center + k_minus) / (h * h);
}

// Complex versions
template <typename T>
c10::complex<T> spherical_bessel_k_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = spherical_bessel_k_eps<T>();
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));

    if (std::abs(z) < eps) {
        return c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    }

    c10::complex<T> k_n = spherical_bessel_k(n, z);
    c10::complex<T> k_nm1 = spherical_bessel_k(n - one, z);
    c10::complex<T> k_nm2 = spherical_bessel_k(n - two, z);

    c10::complex<T> dk_n_dz = -k_nm1 - ((n + one) / z) * k_n;
    c10::complex<T> dk_nm1_dz = -k_nm2 - (n / z) * k_nm1;

    c10::complex<T> z2 = z * z;
    return -dk_nm1_dz + ((n + one) / z2) * k_n - ((n + one) / z) * dk_n_dz;
}

template <typename T>
c10::complex<T> spherical_bessel_k_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(spherical_bessel_k_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> k_p = spherical_bessel_k(n + h, z);
    c10::complex<T> k_p_m1 = spherical_bessel_k(n + h - one, z);
    c10::complex<T> dk_dz_plus = -k_p_m1 - ((n + h + one) / z) * k_p;

    c10::complex<T> k_m = spherical_bessel_k(n - h, z);
    c10::complex<T> k_m_m1 = spherical_bessel_k(n - h - one, z);
    c10::complex<T> dk_dz_minus = -k_m_m1 - ((n - h + one) / z) * k_m;

    return (dk_dz_plus - dk_dz_minus) / (two * h);
}

template <typename T>
c10::complex<T> spherical_bessel_k_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(spherical_bessel_k_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> k_plus = spherical_bessel_k(n + h, z);
    c10::complex<T> k_center = spherical_bessel_k(n, z);
    c10::complex<T> k_minus = spherical_bessel_k(n - h, z);

    return (k_plus - two * k_center + k_minus) / (h * h);
}

} // namespace detail

// Real backward_backward: returns (grad_grad_output, grad_n, grad_z)
// Computes gradients of the backward pass w.r.t. (grad_output, n, z)
// given upstream gradients (gg_n, gg_z) for the outputs (grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> spherical_bessel_k_backward_backward(
    T gg_n,       // upstream gradient for grad_n output
    T gg_z,       // upstream gradient for grad_z output
    T grad_output,
    T n,
    T z
) {
    const T eps = detail::spherical_bessel_k_eps<T>();

    // Forward backward computes:
    // grad_n = grad_output * dk/dn
    // grad_z = grad_output * dk/dz

    // First derivatives
    T k_n = spherical_bessel_k(n, z);
    T k_nm1 = spherical_bessel_k(n - T(1), z);

    T dk_dz;
    if (std::abs(z) < eps) {
        dk_dz = -std::numeric_limits<T>::infinity();
    } else {
        dk_dz = -k_nm1 - ((n + T(1)) / z) * k_n;
    }

    T dk_dn = detail::spherical_bessel_k_n_derivative(n, z);

    // Second derivatives
    T d2k_dz2 = detail::spherical_bessel_k_zz_derivative(n, z);
    T d2k_dn2 = detail::spherical_bessel_k_nn_derivative(n, z);
    T d2k_dndz = detail::spherical_bessel_k_nz_derivative(n, z);

    // Accumulate gradients
    // grad_grad_output = gg_n * dk/dn + gg_z * dk/dz
    T grad_grad_output = gg_n * dk_dn + gg_z * dk_dz;

    // grad_n = gg_n * grad_output * d^2k/dn^2 + gg_z * grad_output * d^2k/(dn dz)
    T grad_n = grad_output * (gg_n * d2k_dn2 + gg_z * d2k_dndz);

    // grad_z = gg_n * grad_output * d^2k/(dn dz) + gg_z * grad_output * d^2k/dz^2
    T grad_z = grad_output * (gg_n * d2k_dndz + gg_z * d2k_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> spherical_bessel_k_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const T eps = detail::spherical_bessel_k_eps<T>();
    const c10::complex<T> one(T(1), T(0));

    // First derivatives
    c10::complex<T> k_n = spherical_bessel_k(n, z);
    c10::complex<T> k_nm1 = spherical_bessel_k(n - one, z);

    c10::complex<T> dk_dz;
    if (std::abs(z) < eps) {
        dk_dz = c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    } else {
        dk_dz = -k_nm1 - ((n + one) / z) * k_n;
    }

    c10::complex<T> dk_dn = detail::spherical_bessel_k_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2k_dz2 = detail::spherical_bessel_k_zz_derivative(n, z);
    c10::complex<T> d2k_dn2 = detail::spherical_bessel_k_nn_derivative(n, z);
    c10::complex<T> d2k_dndz = detail::spherical_bessel_k_nz_derivative(n, z);

    // Accumulate gradients (using conjugates for Wirtinger derivatives)
    c10::complex<T> grad_grad_output = gg_n * std::conj(dk_dn) + gg_z * std::conj(dk_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2k_dn2) + gg_z * std::conj(d2k_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2k_dndz) + gg_z * std::conj(d2k_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
