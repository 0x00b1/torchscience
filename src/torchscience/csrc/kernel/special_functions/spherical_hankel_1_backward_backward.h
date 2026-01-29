#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "spherical_hankel_1.h"
#include "spherical_hankel_1_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative w.r.t. z: d^2/dz^2 h_n^(1)(z)
// Using: d/dz[(n/z)*h_n - h_{n+1}]
//      = -n/z^2 * h_n + n/z * dh_n/dz - dh_{n+1}/dz
template <typename T>
c10::complex<T> spherical_hankel_1_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = spherical_hankel_1_eps<T>();
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));

    if (std::abs(z) < eps) {
        return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                               std::numeric_limits<T>::quiet_NaN());
    }

    c10::complex<T> h_n = spherical_hankel_1(n, z);
    c10::complex<T> h_np1 = spherical_hankel_1(n + one, z);
    c10::complex<T> h_np2 = spherical_hankel_1(n + two, z);

    c10::complex<T> dh_n_dz = (n / z) * h_n - h_np1;
    c10::complex<T> dh_np1_dz = ((n + one) / z) * h_np1 - h_np2;

    c10::complex<T> z2 = z * z;
    return -n / z2 * h_n + (n / z) * dh_n_dz - dh_np1_dz;
}

// Mixed second derivative d^2/(dn dz) h_n^(1)(z) computed numerically
template <typename T>
c10::complex<T> spherical_hankel_1_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(spherical_hankel_1_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> h_p = spherical_hankel_1(n + h, z);
    c10::complex<T> h_p_p1 = spherical_hankel_1(n + h + one, z);
    c10::complex<T> dh_dz_plus = ((n + h) / z) * h_p - h_p_p1;

    c10::complex<T> h_m = spherical_hankel_1(n - h, z);
    c10::complex<T> h_m_p1 = spherical_hankel_1(n - h + one, z);
    c10::complex<T> dh_dz_minus = ((n - h) / z) * h_m - h_m_p1;

    return (dh_dz_plus - dh_dz_minus) / (two * h);
}

// Second derivative w.r.t. n: d^2/dn^2 h_n^(1)(z) computed numerically
template <typename T>
c10::complex<T> spherical_hankel_1_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(spherical_hankel_1_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> h_plus = spherical_hankel_1(n + h, z);
    c10::complex<T> h_center = spherical_hankel_1(n, z);
    c10::complex<T> h_minus = spherical_hankel_1(n - h, z);

    return (h_plus - two * h_center + h_minus) / (h * h);
}

} // namespace detail

// Complex backward_backward: returns (grad_grad_output, grad_n, grad_z)
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> spherical_hankel_1_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const T eps = detail::spherical_hankel_1_eps<T>();
    const c10::complex<T> one(T(1), T(0));

    // First derivatives
    c10::complex<T> h_n = spherical_hankel_1(n, z);
    c10::complex<T> h_np1 = spherical_hankel_1(n + one, z);

    c10::complex<T> dh_dz;
    if (std::abs(z) < eps) {
        dh_dz = c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                                std::numeric_limits<T>::quiet_NaN());
    } else {
        dh_dz = (n / z) * h_n - h_np1;
    }

    c10::complex<T> dh_dn = detail::spherical_hankel_1_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2h_dz2 = detail::spherical_hankel_1_zz_derivative(n, z);
    c10::complex<T> d2h_dn2 = detail::spherical_hankel_1_nn_derivative(n, z);
    c10::complex<T> d2h_dndz = detail::spherical_hankel_1_nz_derivative(n, z);

    // Accumulate gradients (using conjugates for Wirtinger derivatives)
    c10::complex<T> grad_grad_output = gg_n * std::conj(dh_dn) + gg_z * std::conj(dh_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2h_dn2) + gg_z * std::conj(d2h_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2h_dndz) + gg_z * std::conj(d2h_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
