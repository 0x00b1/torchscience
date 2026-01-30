#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "spherical_hankel_2.h"
#include "spherical_bessel_j.h"
#include "spherical_bessel_y.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute d/dn h_n^(2)(z) using finite differences
// The analytical formula is complex, so we use numerical approximation
template <typename T>
c10::complex<T> spherical_hankel_2_n_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(spherical_hankel_2_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    // Scale h based on |n|
    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> h_plus = spherical_hankel_2(n + h, z);
    c10::complex<T> h_minus = spherical_hankel_2(n - h, z);

    return (h_plus - h_minus) / (two * h);
}

} // namespace detail

// Complex backward: returns (grad_n, grad_z)
// d/dz h_n^(2)(z) = (n/z) * h_n^(2)(z) - h_{n+1}^(2)(z)
// d/dn h_n^(2)(z) computed numerically
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> spherical_hankel_2_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const T eps = detail::spherical_hankel_2_eps<T>();
    const c10::complex<T> one(T(1), T(0));

    // Gradient w.r.t. z: d/dz h_n^(2)(z) = (n/z) * h_n^(2)(z) - h_{n+1}^(2)(z)
    c10::complex<T> h_n = spherical_hankel_2(n, z);
    c10::complex<T> h_np1 = spherical_hankel_2(n + one, z);

    c10::complex<T> dh_dz;
    if (std::abs(z) < eps) {
        dh_dz = c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                                std::numeric_limits<T>::quiet_NaN());
    } else {
        dh_dz = (n / z) * h_n - h_np1;
    }

    // For complex gradients, we use the conjugate (Wirtinger derivative)
    c10::complex<T> grad_z = grad_output * std::conj(dh_dz);

    // Gradient w.r.t. n: computed numerically
    c10::complex<T> dh_dn = detail::spherical_hankel_2_n_derivative(n, z);
    c10::complex<T> grad_n = grad_output * std::conj(dh_dn);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
