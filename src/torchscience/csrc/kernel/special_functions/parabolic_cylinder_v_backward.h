#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "parabolic_cylinder_v.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute dV/da using finite differences
template <typename T>
T parabolic_cylinder_v_a_derivative(T a, T z) {
    const T eps = std::sqrt(pcf_eps<T>());
    T h = eps * (std::abs(a) > T(1) ? std::abs(a) : T(1));

    T v_plus = parabolic_cylinder_v(a + h, z);
    T v_minus = parabolic_cylinder_v(a - h, z);

    return (v_plus - v_minus) / (T(2) * h);
}

// Compute dV/dz using recurrence relation
// DLMF 12.8.2: V'(a,z) = z/2 * V(a,z) + Gamma(1/2+a) * V(a-1,z)
// Or: V'(a,z) = -z/2 * V(a,z) + V(a+1,z)
template <typename T>
T parabolic_cylinder_v_z_derivative(T a, T z) {
    T v_a = parabolic_cylinder_v(a, z);
    T v_ap1 = parabolic_cylinder_v(a + T(1), z);

    return -z / T(2) * v_a + v_ap1;
}

// Complex versions
template <typename T>
c10::complex<T> parabolic_cylinder_v_a_derivative(c10::complex<T> a, c10::complex<T> z) {
    const T eps = std::sqrt(pcf_eps<T>());
    T a_mag = std::abs(a);
    c10::complex<T> h(eps * (a_mag > T(1) ? a_mag : T(1)), T(0));

    c10::complex<T> v_plus = parabolic_cylinder_v(a + h, z);
    c10::complex<T> v_minus = parabolic_cylinder_v(a - h, z);

    return (v_plus - v_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

template <typename T>
c10::complex<T> parabolic_cylinder_v_z_derivative(c10::complex<T> a, c10::complex<T> z) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    c10::complex<T> v_a = parabolic_cylinder_v(a, z);
    c10::complex<T> v_ap1 = parabolic_cylinder_v(a + one, z);

    return -z / two * v_a + v_ap1;
}

} // namespace detail

// Real backward: returns (grad_a, grad_z)
template <typename T>
std::tuple<T, T> parabolic_cylinder_v_backward(T grad_output, T a, T z) {
    T dv_da = detail::parabolic_cylinder_v_a_derivative(a, z);
    T dv_dz = detail::parabolic_cylinder_v_z_derivative(a, z);

    return {grad_output * dv_da, grad_output * dv_dz};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> parabolic_cylinder_v_backward(
    c10::complex<T> grad_output,
    c10::complex<T> a,
    c10::complex<T> z
) {
    c10::complex<T> dv_da = detail::parabolic_cylinder_v_a_derivative(a, z);
    c10::complex<T> dv_dz = detail::parabolic_cylinder_v_z_derivative(a, z);

    return {grad_output * std::conj(dv_da), grad_output * std::conj(dv_dz)};
}

} // namespace torchscience::kernel::special_functions
