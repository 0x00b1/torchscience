#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "parabolic_cylinder_u.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute dU/da using finite differences (numerical)
template <typename T>
T parabolic_cylinder_u_a_derivative(T a, T z) {
    const T eps = std::sqrt(pcf_eps<T>());
    T h = eps * (std::abs(a) > T(1) ? std::abs(a) : T(1));

    T u_plus = parabolic_cylinder_u(a + h, z);
    T u_minus = parabolic_cylinder_u(a - h, z);

    return (u_plus - u_minus) / (T(2) * h);
}

// Compute dU/dz using recurrence relation
// DLMF 12.8.1: U'(a,z) = -z/2 * U(a,z) - (a + 1/2) * U(a+1,z)
// Or equivalently: U'(a,z) = z/2 * U(a,z) - U(a-1,z)
template <typename T>
T parabolic_cylinder_u_z_derivative(T a, T z) {
    T u_a = parabolic_cylinder_u(a, z);
    T u_am1 = parabolic_cylinder_u(a - T(1), z);

    return z / T(2) * u_a - u_am1;
}

// Complex versions
template <typename T>
c10::complex<T> parabolic_cylinder_u_a_derivative(c10::complex<T> a, c10::complex<T> z) {
    const T eps = std::sqrt(pcf_eps<T>());
    T a_mag = std::abs(a);
    c10::complex<T> h(eps * (a_mag > T(1) ? a_mag : T(1)), T(0));

    c10::complex<T> u_plus = parabolic_cylinder_u(a + h, z);
    c10::complex<T> u_minus = parabolic_cylinder_u(a - h, z);

    return (u_plus - u_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

template <typename T>
c10::complex<T> parabolic_cylinder_u_z_derivative(c10::complex<T> a, c10::complex<T> z) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    c10::complex<T> u_a = parabolic_cylinder_u(a, z);
    c10::complex<T> u_am1 = parabolic_cylinder_u(a - one, z);

    return z / two * u_a - u_am1;
}

} // namespace detail

// Real backward: returns (grad_a, grad_z)
template <typename T>
std::tuple<T, T> parabolic_cylinder_u_backward(T grad_output, T a, T z) {
    T du_da = detail::parabolic_cylinder_u_a_derivative(a, z);
    T du_dz = detail::parabolic_cylinder_u_z_derivative(a, z);

    return {grad_output * du_da, grad_output * du_dz};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> parabolic_cylinder_u_backward(
    c10::complex<T> grad_output,
    c10::complex<T> a,
    c10::complex<T> z
) {
    c10::complex<T> du_da = detail::parabolic_cylinder_u_a_derivative(a, z);
    c10::complex<T> du_dz = detail::parabolic_cylinder_u_z_derivative(a, z);

    // Wirtinger derivatives
    return {grad_output * std::conj(du_da), grad_output * std::conj(du_dz)};
}

} // namespace torchscience::kernel::special_functions
