#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "struve_l_0.h"
#include "struve_l_1.h"
#include "struve_l_0_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// 2/pi constant
constexpr double STRUVE_L0_BB_TWO_OVER_PI = 0.6366197723675813430755350534900574;

template <typename T>
inline T struve_l_0_bb_zero_tolerance() {
    return T(1e-12);
}

template <>
inline float struve_l_0_bb_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double struve_l_0_bb_zero_tolerance<double>() { return 1e-12; }

} // namespace detail

// Real backward_backward
// Returns gradients for (grad_output, z)
//
// d/dz L_0(z) = (2/pi) + L_1(z)
// d^2/dz^2 L_0(z) = d/dz L_1(z) = L_0(z) - L_1(z)/z
//
// At z=0: limit of [L_0(z) - L_1(z)/z] is 0 (see struve_l_1_backward.h)
template <typename T>
std::tuple<T, T> struve_l_0_backward_backward(T gg_z, T grad_output, T z) {
    T l0 = struve_l_0(z);
    T l1 = struve_l_1(z);
    T two_over_pi = T(detail::STRUVE_L0_BB_TWO_OVER_PI);

    // d(backward)/d(grad_output) = (2/pi) + L_1(z)
    T first_deriv = two_over_pi + l1;
    T grad_grad_output = gg_z * first_deriv;

    // d(backward)/dz = grad_output * d^2L_0/dz^2
    // d^2L_0/dz^2 = d/dz L_1(z) = L_0(z) - L_1(z)/z
    T d2_l0;
    if (std::abs(z) < detail::struve_l_0_bb_zero_tolerance<T>()) {
        // Limit as z -> 0 is 0
        d2_l0 = T(0);
    } else {
        d2_l0 = l0 - l1 / z;
    }
    T grad_z = gg_z * grad_output * d2_l0;

    return {grad_grad_output, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> struve_l_0_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> l0 = struve_l_0(z);
    c10::complex<T> l1 = struve_l_1(z);
    c10::complex<T> two_over_pi(T(detail::STRUVE_L0_BB_TWO_OVER_PI), T(0));

    // d(backward)/d(grad_output) = (2/pi) + L_1(z)
    c10::complex<T> first_deriv = two_over_pi + l1;
    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);

    // d^2L_0/dz^2 = L_0(z) - L_1(z)/z
    c10::complex<T> d2_l0;
    if (std::abs(z) < detail::struve_l_0_bb_zero_tolerance<T>()) {
        // Limit as z -> 0 is 0
        d2_l0 = c10::complex<T>(T(0), T(0));
    } else {
        d2_l0 = l0 - l1 / z;
    }
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(d2_l0);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
