#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "parabolic_cylinder_v.h"
#include "parabolic_cylinder_v_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative d²V/da²
template <typename T>
T parabolic_cylinder_v_aa_derivative(T a, T z) {
    const T eps = std::cbrt(pcf_eps<T>());
    T h = eps * (std::abs(a) > T(1) ? std::abs(a) : T(1));

    T v_plus = parabolic_cylinder_v(a + h, z);
    T v_center = parabolic_cylinder_v(a, z);
    T v_minus = parabolic_cylinder_v(a - h, z);

    return (v_plus - T(2) * v_center + v_minus) / (h * h);
}

// Second derivative d²V/dz²
// From differential equation: V''(a,z) = (z²/4 + a) * V(a,z)
template <typename T>
T parabolic_cylinder_v_zz_derivative(T a, T z) {
    T v = parabolic_cylinder_v(a, z);
    return (z * z / T(4) + a) * v;
}

// Mixed second derivative d²V/dadz
template <typename T>
T parabolic_cylinder_v_az_derivative(T a, T z) {
    const T eps = std::sqrt(pcf_eps<T>());
    T h = eps * (std::abs(a) > T(1) ? std::abs(a) : T(1));

    T dz_plus = parabolic_cylinder_v_z_derivative(a + h, z);
    T dz_minus = parabolic_cylinder_v_z_derivative(a - h, z);

    return (dz_plus - dz_minus) / (T(2) * h);
}

// Complex versions
template <typename T>
c10::complex<T> parabolic_cylinder_v_aa_derivative(c10::complex<T> a, c10::complex<T> z) {
    const T eps = std::cbrt(pcf_eps<T>());
    T a_mag = std::abs(a);
    c10::complex<T> h(eps * (a_mag > T(1) ? a_mag : T(1)), T(0));

    c10::complex<T> v_plus = parabolic_cylinder_v(a + h, z);
    c10::complex<T> v_center = parabolic_cylinder_v(a, z);
    c10::complex<T> v_minus = parabolic_cylinder_v(a - h, z);

    return (v_plus - c10::complex<T>(T(2), T(0)) * v_center + v_minus) / (h * h);
}

template <typename T>
c10::complex<T> parabolic_cylinder_v_zz_derivative(c10::complex<T> a, c10::complex<T> z) {
    c10::complex<T> v = parabolic_cylinder_v(a, z);
    return (z * z / c10::complex<T>(T(4), T(0)) + a) * v;
}

template <typename T>
c10::complex<T> parabolic_cylinder_v_az_derivative(c10::complex<T> a, c10::complex<T> z) {
    const T eps = std::sqrt(pcf_eps<T>());
    T a_mag = std::abs(a);
    c10::complex<T> h(eps * (a_mag > T(1) ? a_mag : T(1)), T(0));

    c10::complex<T> dz_plus = parabolic_cylinder_v_z_derivative(a + h, z);
    c10::complex<T> dz_minus = parabolic_cylinder_v_z_derivative(a - h, z);

    return (dz_plus - dz_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Real backward_backward: returns (grad_grad_output, grad_a, grad_z)
template <typename T>
std::tuple<T, T, T> parabolic_cylinder_v_backward_backward(
    T gg_a,
    T gg_z,
    T grad_output,
    T a,
    T z
) {
    T dv_da = detail::parabolic_cylinder_v_a_derivative(a, z);
    T dv_dz = detail::parabolic_cylinder_v_z_derivative(a, z);

    T d2v_da2 = detail::parabolic_cylinder_v_aa_derivative(a, z);
    T d2v_dz2 = detail::parabolic_cylinder_v_zz_derivative(a, z);
    T d2v_dadz = detail::parabolic_cylinder_v_az_derivative(a, z);

    T grad_grad_output = gg_a * dv_da + gg_z * dv_dz;
    T grad_a = grad_output * (gg_a * d2v_da2 + gg_z * d2v_dadz);
    T grad_z = grad_output * (gg_a * d2v_dadz + gg_z * d2v_dz2);

    return {grad_grad_output, grad_a, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> parabolic_cylinder_v_backward_backward(
    c10::complex<T> gg_a,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> a,
    c10::complex<T> z
) {
    c10::complex<T> dv_da = detail::parabolic_cylinder_v_a_derivative(a, z);
    c10::complex<T> dv_dz = detail::parabolic_cylinder_v_z_derivative(a, z);

    c10::complex<T> d2v_da2 = detail::parabolic_cylinder_v_aa_derivative(a, z);
    c10::complex<T> d2v_dz2 = detail::parabolic_cylinder_v_zz_derivative(a, z);
    c10::complex<T> d2v_dadz = detail::parabolic_cylinder_v_az_derivative(a, z);

    c10::complex<T> grad_grad_output = gg_a * std::conj(dv_da) + gg_z * std::conj(dv_dz);
    c10::complex<T> grad_a = grad_output * (gg_a * std::conj(d2v_da2) + gg_z * std::conj(d2v_dadz));
    c10::complex<T> grad_z = grad_output * (gg_a * std::conj(d2v_dadz) + gg_z * std::conj(d2v_dz2));

    return {grad_grad_output, grad_a, grad_z};
}

} // namespace torchscience::kernel::special_functions
