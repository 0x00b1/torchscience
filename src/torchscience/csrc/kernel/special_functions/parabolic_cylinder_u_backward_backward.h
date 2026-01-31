#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "parabolic_cylinder_u.h"
#include "parabolic_cylinder_u_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative d²U/da²
template <typename T>
T parabolic_cylinder_u_aa_derivative(T a, T z) {
    const T eps = std::cbrt(pcf_eps<T>());
    T h = eps * (std::abs(a) > T(1) ? std::abs(a) : T(1));

    T u_plus = parabolic_cylinder_u(a + h, z);
    T u_center = parabolic_cylinder_u(a, z);
    T u_minus = parabolic_cylinder_u(a - h, z);

    return (u_plus - T(2) * u_center + u_minus) / (h * h);
}

// Second derivative d²U/dz²
// From DLMF 12.2.2: U''(a,z) = (z²/4 + a) * U(a,z)
template <typename T>
T parabolic_cylinder_u_zz_derivative(T a, T z) {
    T u = parabolic_cylinder_u(a, z);
    return (z * z / T(4) + a) * u;
}

// Mixed second derivative d²U/dadz
template <typename T>
T parabolic_cylinder_u_az_derivative(T a, T z) {
    const T eps = std::sqrt(pcf_eps<T>());
    T h = eps * (std::abs(a) > T(1) ? std::abs(a) : T(1));

    T dz_plus = parabolic_cylinder_u_z_derivative(a + h, z);
    T dz_minus = parabolic_cylinder_u_z_derivative(a - h, z);

    return (dz_plus - dz_minus) / (T(2) * h);
}

// Complex versions
template <typename T>
c10::complex<T> parabolic_cylinder_u_aa_derivative(c10::complex<T> a, c10::complex<T> z) {
    const T eps = std::cbrt(pcf_eps<T>());
    T a_mag = std::abs(a);
    c10::complex<T> h(eps * (a_mag > T(1) ? a_mag : T(1)), T(0));

    c10::complex<T> u_plus = parabolic_cylinder_u(a + h, z);
    c10::complex<T> u_center = parabolic_cylinder_u(a, z);
    c10::complex<T> u_minus = parabolic_cylinder_u(a - h, z);

    return (u_plus - c10::complex<T>(T(2), T(0)) * u_center + u_minus) / (h * h);
}

template <typename T>
c10::complex<T> parabolic_cylinder_u_zz_derivative(c10::complex<T> a, c10::complex<T> z) {
    c10::complex<T> u = parabolic_cylinder_u(a, z);
    return (z * z / c10::complex<T>(T(4), T(0)) + a) * u;
}

template <typename T>
c10::complex<T> parabolic_cylinder_u_az_derivative(c10::complex<T> a, c10::complex<T> z) {
    const T eps = std::sqrt(pcf_eps<T>());
    T a_mag = std::abs(a);
    c10::complex<T> h(eps * (a_mag > T(1) ? a_mag : T(1)), T(0));

    c10::complex<T> dz_plus = parabolic_cylinder_u_z_derivative(a + h, z);
    c10::complex<T> dz_minus = parabolic_cylinder_u_z_derivative(a - h, z);

    return (dz_plus - dz_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Real backward_backward: returns (grad_grad_output, grad_a, grad_z)
template <typename T>
std::tuple<T, T, T> parabolic_cylinder_u_backward_backward(
    T gg_a,        // upstream gradient for grad_a output
    T gg_z,        // upstream gradient for grad_z output
    T grad_output,
    T a,
    T z
) {
    // First derivatives
    T du_da = detail::parabolic_cylinder_u_a_derivative(a, z);
    T du_dz = detail::parabolic_cylinder_u_z_derivative(a, z);

    // Second derivatives
    T d2u_da2 = detail::parabolic_cylinder_u_aa_derivative(a, z);
    T d2u_dz2 = detail::parabolic_cylinder_u_zz_derivative(a, z);
    T d2u_dadz = detail::parabolic_cylinder_u_az_derivative(a, z);

    // Accumulate gradients
    T grad_grad_output = gg_a * du_da + gg_z * du_dz;
    T grad_a = grad_output * (gg_a * d2u_da2 + gg_z * d2u_dadz);
    T grad_z = grad_output * (gg_a * d2u_dadz + gg_z * d2u_dz2);

    return {grad_grad_output, grad_a, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> parabolic_cylinder_u_backward_backward(
    c10::complex<T> gg_a,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> a,
    c10::complex<T> z
) {
    // First derivatives
    c10::complex<T> du_da = detail::parabolic_cylinder_u_a_derivative(a, z);
    c10::complex<T> du_dz = detail::parabolic_cylinder_u_z_derivative(a, z);

    // Second derivatives
    c10::complex<T> d2u_da2 = detail::parabolic_cylinder_u_aa_derivative(a, z);
    c10::complex<T> d2u_dz2 = detail::parabolic_cylinder_u_zz_derivative(a, z);
    c10::complex<T> d2u_dadz = detail::parabolic_cylinder_u_az_derivative(a, z);

    // Accumulate gradients (using conjugates for Wirtinger derivatives)
    c10::complex<T> grad_grad_output = gg_a * std::conj(du_da) + gg_z * std::conj(du_dz);
    c10::complex<T> grad_a = grad_output * (gg_a * std::conj(d2u_da2) + gg_z * std::conj(d2u_dadz));
    c10::complex<T> grad_z = grad_output * (gg_a * std::conj(d2u_dadz) + gg_z * std::conj(d2u_dz2));

    return {grad_grad_output, grad_a, grad_z};
}

} // namespace torchscience::kernel::special_functions
