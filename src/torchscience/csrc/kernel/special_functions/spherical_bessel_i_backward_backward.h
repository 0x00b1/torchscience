#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "spherical_bessel_i.h"
#include "spherical_bessel_i_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative w.r.t. z: d^2/dz^2 i_n(z)
// Using: d/dz[(n/z)*i_n + i_{n+1}] = -n/z^2 * i_n + n/z * di_n/dz + di_{n+1}/dz
// where di_n/dz = (n/z)*i_n + i_{n+1}
// and di_{n+1}/dz = ((n+1)/z)*i_{n+1} + i_{n+2}
template <typename T>
T spherical_bessel_i_zz_derivative(T n, T z) {
    const T eps = spherical_bessel_i_eps<T>();

    if (std::abs(z) < eps) {
        // For small z, use limiting behavior
        if (std::abs(n) < eps) {
            // i_0''(0) = 1/3
            return T(1) / T(3);
        } else if (std::abs(n - T(1)) < eps) {
            // i_1''(0) = 0 (from Taylor expansion)
            return T(0);
        } else if (std::abs(n - T(2)) < eps) {
            // i_2''(0) = 2/15
            return T(2) / T(15);
        } else {
            return T(0);  // Higher orders vanish faster
        }
    }

    T i_n = spherical_bessel_i(n, z);
    T i_np1 = spherical_bessel_i(n + T(1), z);
    T i_np2 = spherical_bessel_i(n + T(2), z);

    // d/dz i_n = (n/z) * i_n + i_{n+1}
    T di_n_dz = (n / z) * i_n + i_np1;

    // d/dz i_{n+1} = ((n+1)/z) * i_{n+1} + i_{n+2}
    T di_np1_dz = ((n + T(1)) / z) * i_np1 + i_np2;

    // d^2/dz^2 i_n = d/dz[(n/z)*i_n + i_{n+1}]
    //             = -n/z^2 * i_n + n/z * di_n/dz + di_{n+1}/dz
    T z2 = z * z;
    return -n / z2 * i_n + (n / z) * di_n_dz + di_np1_dz;
}

// Mixed second derivative d^2/(dn dz) i_n(z) computed numerically
template <typename T>
T spherical_bessel_i_nz_derivative(T n, T z) {
    const T eps = std::sqrt(spherical_bessel_i_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    // Compute d/dz i_{n+h} and d/dz i_{n-h}
    T i_p = spherical_bessel_i(n + h, z);
    T i_p_p1 = spherical_bessel_i(n + h + T(1), z);
    T di_dz_plus = ((n + h) / z) * i_p + i_p_p1;

    T i_m = spherical_bessel_i(n - h, z);
    T i_m_p1 = spherical_bessel_i(n - h + T(1), z);
    T di_dz_minus = ((n - h) / z) * i_m + i_m_p1;

    return (di_dz_plus - di_dz_minus) / (T(2) * h);
}

// Second derivative w.r.t. n: d^2/dn^2 i_n(z) computed numerically
template <typename T>
T spherical_bessel_i_nn_derivative(T n, T z) {
    const T eps = std::cbrt(spherical_bessel_i_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T i_plus = spherical_bessel_i(n + h, z);
    T i_center = spherical_bessel_i(n, z);
    T i_minus = spherical_bessel_i(n - h, z);

    return (i_plus - T(2) * i_center + i_minus) / (h * h);
}

// Complex versions
template <typename T>
c10::complex<T> spherical_bessel_i_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = spherical_bessel_i_eps<T>();
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));

    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            return c10::complex<T>(T(1) / T(3), T(0));
        } else if (std::abs(n - one) < eps) {
            return c10::complex<T>(T(0), T(0));
        } else if (std::abs(n - two) < eps) {
            return c10::complex<T>(T(2) / T(15), T(0));
        } else {
            return c10::complex<T>(T(0), T(0));
        }
    }

    c10::complex<T> i_n = spherical_bessel_i(n, z);
    c10::complex<T> i_np1 = spherical_bessel_i(n + one, z);
    c10::complex<T> i_np2 = spherical_bessel_i(n + two, z);

    c10::complex<T> di_n_dz = (n / z) * i_n + i_np1;
    c10::complex<T> di_np1_dz = ((n + one) / z) * i_np1 + i_np2;

    c10::complex<T> z2 = z * z;
    return -n / z2 * i_n + (n / z) * di_n_dz + di_np1_dz;
}

template <typename T>
c10::complex<T> spherical_bessel_i_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(spherical_bessel_i_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> i_p = spherical_bessel_i(n + h, z);
    c10::complex<T> i_p_p1 = spherical_bessel_i(n + h + one, z);
    c10::complex<T> di_dz_plus = ((n + h) / z) * i_p + i_p_p1;

    c10::complex<T> i_m = spherical_bessel_i(n - h, z);
    c10::complex<T> i_m_p1 = spherical_bessel_i(n - h + one, z);
    c10::complex<T> di_dz_minus = ((n - h) / z) * i_m + i_m_p1;

    return (di_dz_plus - di_dz_minus) / (two * h);
}

template <typename T>
c10::complex<T> spherical_bessel_i_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(spherical_bessel_i_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> i_plus = spherical_bessel_i(n + h, z);
    c10::complex<T> i_center = spherical_bessel_i(n, z);
    c10::complex<T> i_minus = spherical_bessel_i(n - h, z);

    return (i_plus - two * i_center + i_minus) / (h * h);
}

} // namespace detail

// Real backward_backward: returns (grad_grad_output, grad_n, grad_z)
// Computes gradients of the backward pass w.r.t. (grad_output, n, z)
// given upstream gradients (gg_n, gg_z) for the outputs (grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> spherical_bessel_i_backward_backward(
    T gg_n,       // upstream gradient for grad_n output
    T gg_z,       // upstream gradient for grad_z output
    T grad_output,
    T n,
    T z
) {
    const T eps = detail::spherical_bessel_i_eps<T>();

    // Forward backward computes:
    // grad_n = grad_output * di/dn
    // grad_z = grad_output * di/dz

    // First derivatives
    T i_n = spherical_bessel_i(n, z);
    T i_np1 = spherical_bessel_i(n + T(1), z);

    T di_dz;
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            di_dz = T(0);
        } else if (std::abs(n - T(1)) < eps) {
            di_dz = T(1) / T(3);
        } else {
            di_dz = T(0);
        }
    } else {
        di_dz = (n / z) * i_n + i_np1;
    }

    T di_dn = detail::spherical_bessel_i_n_derivative(n, z);

    // Second derivatives
    T d2i_dz2 = detail::spherical_bessel_i_zz_derivative(n, z);
    T d2i_dn2 = detail::spherical_bessel_i_nn_derivative(n, z);
    T d2i_dndz = detail::spherical_bessel_i_nz_derivative(n, z);

    // Accumulate gradients
    // grad_grad_output = gg_n * di/dn + gg_z * di/dz
    T grad_grad_output = gg_n * di_dn + gg_z * di_dz;

    // grad_n = gg_n * grad_output * d^2i/dn^2 + gg_z * grad_output * d^2i/(dn dz)
    T grad_n = grad_output * (gg_n * d2i_dn2 + gg_z * d2i_dndz);

    // grad_z = gg_n * grad_output * d^2i/(dn dz) + gg_z * grad_output * d^2i/dz^2
    T grad_z = grad_output * (gg_n * d2i_dndz + gg_z * d2i_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> spherical_bessel_i_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const T eps = detail::spherical_bessel_i_eps<T>();
    const c10::complex<T> one(T(1), T(0));

    // First derivatives
    c10::complex<T> i_n = spherical_bessel_i(n, z);
    c10::complex<T> i_np1 = spherical_bessel_i(n + one, z);

    c10::complex<T> di_dz;
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            di_dz = c10::complex<T>(T(0), T(0));
        } else if (std::abs(n - one) < eps) {
            di_dz = c10::complex<T>(T(1) / T(3), T(0));
        } else {
            di_dz = c10::complex<T>(T(0), T(0));
        }
    } else {
        di_dz = (n / z) * i_n + i_np1;
    }

    c10::complex<T> di_dn = detail::spherical_bessel_i_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2i_dz2 = detail::spherical_bessel_i_zz_derivative(n, z);
    c10::complex<T> d2i_dn2 = detail::spherical_bessel_i_nn_derivative(n, z);
    c10::complex<T> d2i_dndz = detail::spherical_bessel_i_nz_derivative(n, z);

    // Accumulate gradients (using conjugates for Wirtinger derivatives)
    c10::complex<T> grad_grad_output = gg_n * std::conj(di_dn) + gg_z * std::conj(di_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2i_dn2) + gg_z * std::conj(d2i_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2i_dndz) + gg_z * std::conj(d2i_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
