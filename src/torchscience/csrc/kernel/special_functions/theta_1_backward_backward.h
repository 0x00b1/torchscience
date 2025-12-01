#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "theta_1.h"
#include "theta_1_backward.h"

namespace torchscience::kernel::special_functions {

// Backward backward pass for Jacobi theta function theta_1(z, q)
// Uses numerical finite differences for second derivatives.

namespace detail {

template <typename T>
inline T theta1_finite_diff_step2() {
    return std::pow(std::numeric_limits<T>::epsilon(), T(0.25));
}

template <>
inline float theta1_finite_diff_step2<float>() {
    return 1e-2f;
}

template <>
inline double theta1_finite_diff_step2<double>() {
    return 1e-4;
}

} // namespace detail

template <typename T>
std::tuple<T, T, T> theta_1_backward_backward(
    T gg_z,
    T gg_q,
    T gradient,
    T z,
    T q
) {
    T h = detail::theta1_finite_diff_step2<T>();

    auto [dz, dq] = theta_1_backward(T(1), z, q);

    // Gradient w.r.t. incoming gradient
    T grad_gradient = gg_z * dz + gg_q * dq;

    // Second derivatives via finite differences
    auto [dz_pz, dq_pz] = theta_1_backward(T(1), z + h, q);
    auto [dz_pq, dq_pq] = theta_1_backward(T(1), z, q + h);

    T d2zz = (dz_pz - dz) / h;
    T d2zq = (dz_pq - dz) / h;
    T d2qz = (dq_pz - dq) / h;
    T d2qq = (dq_pq - dq) / h;

    T grad_z = gradient * (gg_z * d2zz + gg_q * d2qz);
    T grad_q = gradient * (gg_z * d2zq + gg_q * d2qq);

    return {grad_gradient, grad_z, grad_q};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
theta_1_backward_backward(
    c10::complex<T> gg_z,
    c10::complex<T> gg_q,
    c10::complex<T> gradient,
    c10::complex<T> z,
    c10::complex<T> q
) {
    T h = detail::theta1_finite_diff_step2<T>();
    c10::complex<T> h_complex(h, T(0));

    auto [dz, dq] = theta_1_backward(c10::complex<T>(T(1), T(0)), z, q);

    c10::complex<T> grad_gradient = gg_z * dz + gg_q * dq;

    auto [dz_pz, dq_pz] = theta_1_backward(c10::complex<T>(T(1), T(0)), z + h_complex, q);
    auto [dz_pq, dq_pq] = theta_1_backward(c10::complex<T>(T(1), T(0)), z, q + h_complex);

    c10::complex<T> d2zz = (dz_pz - dz) / h_complex;
    c10::complex<T> d2zq = (dz_pq - dz) / h_complex;
    c10::complex<T> d2qz = (dq_pz - dq) / h_complex;
    c10::complex<T> d2qq = (dq_pq - dq) / h_complex;

    c10::complex<T> grad_z = gradient * (gg_z * d2zz + gg_q * d2qz);
    c10::complex<T> grad_q = gradient * (gg_z * d2zq + gg_q * d2qq);

    return {grad_gradient, grad_z, grad_q};
}

} // namespace torchscience::kernel::special_functions
