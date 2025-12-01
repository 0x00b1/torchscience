#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "spherical_bessel_j.h"
#include "spherical_bessel_j_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative w.r.t. z: d^2/dz^2 j_n(z)
// Using: d/dz[(n/z)*j_n - j_{n+1}] = -n/z^2 * j_n + n/z * dj_n/dz - dj_{n+1}/dz
// where dj_n/dz = (n/z)*j_n - j_{n+1}
// and dj_{n+1}/dz = ((n+1)/z)*j_{n+1} - j_{n+2}
template <typename T>
T spherical_bessel_j_zz_derivative(T n, T z) {
    const T eps = spherical_bessel_j_eps<T>();

    if (std::abs(z) < eps) {
        // For small z, use limiting behavior
        if (std::abs(n) < eps) {
            // j_0''(0) = -1/3
            return -T(1) / T(3);
        } else if (std::abs(n - T(1)) < eps) {
            // j_1''(0) = 0 (from Taylor expansion)
            return T(0);
        } else if (std::abs(n - T(2)) < eps) {
            // j_2''(0) = 2/15
            return T(2) / T(15);
        } else {
            return T(0);  // Higher orders vanish faster
        }
    }

    T j_n = spherical_bessel_j(n, z);
    T j_np1 = spherical_bessel_j(n + T(1), z);
    T j_np2 = spherical_bessel_j(n + T(2), z);

    // d/dz j_n = (n/z) * j_n - j_{n+1}
    T dj_n_dz = (n / z) * j_n - j_np1;

    // d/dz j_{n+1} = ((n+1)/z) * j_{n+1} - j_{n+2}
    T dj_np1_dz = ((n + T(1)) / z) * j_np1 - j_np2;

    // d^2/dz^2 j_n = d/dz[(n/z)*j_n - j_{n+1}]
    //             = -n/z^2 * j_n + n/z * dj_n/dz - dj_{n+1}/dz
    T z2 = z * z;
    return -n / z2 * j_n + (n / z) * dj_n_dz - dj_np1_dz;
}

// Mixed second derivative d^2/(dn dz) j_n(z) computed numerically
template <typename T>
T spherical_bessel_j_nz_derivative(T n, T z) {
    const T eps = std::sqrt(spherical_bessel_j_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    // Compute d/dz j_{n+h} and d/dz j_{n-h}
    T j_p = spherical_bessel_j(n + h, z);
    T j_p_p1 = spherical_bessel_j(n + h + T(1), z);
    T dj_dz_plus = ((n + h) / z) * j_p - j_p_p1;

    T j_m = spherical_bessel_j(n - h, z);
    T j_m_p1 = spherical_bessel_j(n - h + T(1), z);
    T dj_dz_minus = ((n - h) / z) * j_m - j_m_p1;

    return (dj_dz_plus - dj_dz_minus) / (T(2) * h);
}

// Second derivative w.r.t. n: d^2/dn^2 j_n(z) computed numerically
template <typename T>
T spherical_bessel_j_nn_derivative(T n, T z) {
    const T eps = std::cbrt(spherical_bessel_j_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T j_plus = spherical_bessel_j(n + h, z);
    T j_center = spherical_bessel_j(n, z);
    T j_minus = spherical_bessel_j(n - h, z);

    return (j_plus - T(2) * j_center + j_minus) / (h * h);
}

// Complex versions
template <typename T>
c10::complex<T> spherical_bessel_j_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = spherical_bessel_j_eps<T>();
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));

    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            return c10::complex<T>(-T(1) / T(3), T(0));
        } else if (std::abs(n - one) < eps) {
            return c10::complex<T>(T(0), T(0));
        } else if (std::abs(n - two) < eps) {
            return c10::complex<T>(T(2) / T(15), T(0));
        } else {
            return c10::complex<T>(T(0), T(0));
        }
    }

    c10::complex<T> j_n = spherical_bessel_j(n, z);
    c10::complex<T> j_np1 = spherical_bessel_j(n + one, z);
    c10::complex<T> j_np2 = spherical_bessel_j(n + two, z);

    c10::complex<T> dj_n_dz = (n / z) * j_n - j_np1;
    c10::complex<T> dj_np1_dz = ((n + one) / z) * j_np1 - j_np2;

    c10::complex<T> z2 = z * z;
    return -n / z2 * j_n + (n / z) * dj_n_dz - dj_np1_dz;
}

template <typename T>
c10::complex<T> spherical_bessel_j_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(spherical_bessel_j_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> j_p = spherical_bessel_j(n + h, z);
    c10::complex<T> j_p_p1 = spherical_bessel_j(n + h + one, z);
    c10::complex<T> dj_dz_plus = ((n + h) / z) * j_p - j_p_p1;

    c10::complex<T> j_m = spherical_bessel_j(n - h, z);
    c10::complex<T> j_m_p1 = spherical_bessel_j(n - h + one, z);
    c10::complex<T> dj_dz_minus = ((n - h) / z) * j_m - j_m_p1;

    return (dj_dz_plus - dj_dz_minus) / (two * h);
}

template <typename T>
c10::complex<T> spherical_bessel_j_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(spherical_bessel_j_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> j_plus = spherical_bessel_j(n + h, z);
    c10::complex<T> j_center = spherical_bessel_j(n, z);
    c10::complex<T> j_minus = spherical_bessel_j(n - h, z);

    return (j_plus - two * j_center + j_minus) / (h * h);
}

} // namespace detail

// Real backward_backward: returns (grad_grad_output, grad_n, grad_z)
// Computes gradients of the backward pass w.r.t. (grad_output, n, z)
// given upstream gradients (gg_n, gg_z) for the outputs (grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> spherical_bessel_j_backward_backward(
    T gg_n,       // upstream gradient for grad_n output
    T gg_z,       // upstream gradient for grad_z output
    T grad_output,
    T n,
    T z
) {
    const T eps = detail::spherical_bessel_j_eps<T>();

    // Forward backward computes:
    // grad_n = grad_output * dj/dn
    // grad_z = grad_output * dj/dz

    // First derivatives
    T j_n = spherical_bessel_j(n, z);
    T j_np1 = spherical_bessel_j(n + T(1), z);

    T dj_dz;
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            dj_dz = T(0);
        } else if (std::abs(n - T(1)) < eps) {
            dj_dz = T(1) / T(3);
        } else {
            dj_dz = T(0);
        }
    } else {
        dj_dz = (n / z) * j_n - j_np1;
    }

    T dj_dn = detail::spherical_bessel_j_n_derivative(n, z);

    // Second derivatives
    T d2j_dz2 = detail::spherical_bessel_j_zz_derivative(n, z);
    T d2j_dn2 = detail::spherical_bessel_j_nn_derivative(n, z);
    T d2j_dndz = detail::spherical_bessel_j_nz_derivative(n, z);

    // Accumulate gradients
    // grad_grad_output = gg_n * dj/dn + gg_z * dj/dz
    T grad_grad_output = gg_n * dj_dn + gg_z * dj_dz;

    // grad_n = gg_n * grad_output * d^2j/dn^2 + gg_z * grad_output * d^2j/(dn dz)
    T grad_n = grad_output * (gg_n * d2j_dn2 + gg_z * d2j_dndz);

    // grad_z = gg_n * grad_output * d^2j/(dn dz) + gg_z * grad_output * d^2j/dz^2
    T grad_z = grad_output * (gg_n * d2j_dndz + gg_z * d2j_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> spherical_bessel_j_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const T eps = detail::spherical_bessel_j_eps<T>();
    const c10::complex<T> one(T(1), T(0));

    // First derivatives
    c10::complex<T> j_n = spherical_bessel_j(n, z);
    c10::complex<T> j_np1 = spherical_bessel_j(n + one, z);

    c10::complex<T> dj_dz;
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            dj_dz = c10::complex<T>(T(0), T(0));
        } else if (std::abs(n - one) < eps) {
            dj_dz = c10::complex<T>(T(1) / T(3), T(0));
        } else {
            dj_dz = c10::complex<T>(T(0), T(0));
        }
    } else {
        dj_dz = (n / z) * j_n - j_np1;
    }

    c10::complex<T> dj_dn = detail::spherical_bessel_j_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2j_dz2 = detail::spherical_bessel_j_zz_derivative(n, z);
    c10::complex<T> d2j_dn2 = detail::spherical_bessel_j_nn_derivative(n, z);
    c10::complex<T> d2j_dndz = detail::spherical_bessel_j_nz_derivative(n, z);

    // Accumulate gradients (using conjugates for Wirtinger derivatives)
    c10::complex<T> grad_grad_output = gg_n * std::conj(dj_dn) + gg_z * std::conj(dj_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2j_dn2) + gg_z * std::conj(d2j_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2j_dndz) + gg_z * std::conj(d2j_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
