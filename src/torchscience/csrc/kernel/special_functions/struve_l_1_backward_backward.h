#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "struve_l_0.h"
#include "struve_l_1.h"
#include "struve_l_1_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// 2/pi constant
constexpr double STRUVE_L1_BB_TWO_OVER_PI = 0.6366197723675813430755350534900574;

template <typename T>
inline T struve_l_1_bb_zero_tolerance() {
    return T(1e-12);
}

template <>
inline float struve_l_1_bb_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double struve_l_1_bb_zero_tolerance<double>() { return 1e-12; }

} // namespace detail

// Real backward_backward
// Returns gradients for (grad_output, z)
//
// d/dz L_1(z) = L_0(z) - L_1(z)/z
// d^2/dz^2 L_1(z) = d/dz [L_0(z) - L_1(z)/z]
//                 = L_0'(z) - [L_1'(z)*z - L_1(z)]/z^2
//                 = L_0'(z) - L_1'(z)/z + L_1(z)/z^2
//                 = (2/pi + L_1) - (L_0 - L_1/z)/z + L_1/z^2
//                 = 2/pi + L_1 - L_0/z + L_1/z^2 + L_1/z^2
//                 = 2/pi + L_1 - L_0/z + 2*L_1/z^2
//
// At z=0: we need the limit of this expression.
// L_1(z) = (4/(3*pi^2))*z^2 + O(z^4)
// L_0(z) = (4/pi^2)*z + O(z^3)
// L_0/z ~ 4/pi^2
// L_1/z^2 ~ 4/(3*pi^2)
//
// So the limit is: 2/pi + 0 - 4/pi^2 + 2*4/(3*pi^2)
//                = 2/pi - 4/pi^2 + 8/(3*pi^2)
//                = 2/pi - 12/(3*pi^2) + 8/(3*pi^2)
//                = 2/pi - 4/(3*pi^2)
//
// Actually, let's verify using series expansion:
// L_1(z) = sum_{k>=0} b_k * z^{2k+2}
// L_1'(z) = sum_{k>=0} (2k+2) * b_k * z^{2k+1}
// L_1''(z) = sum_{k>=0} (2k+2)(2k+1) * b_k * z^{2k}
// L_1''(0) = 2*1 * b_0 = 2 * b_0
//
// where b_0 = (1/4) / [Gamma(3/2)*Gamma(5/2)]
//           = (1/4) / (3*pi/8)
//           = (1/4) * (8/(3*pi))
//           = 2/(3*pi)
//
// So L_1''(0) = 2 * 2/(3*pi) = 4/(3*pi)
template <typename T>
std::tuple<T, T> struve_l_1_backward_backward(T gg_z, T grad_output, T z) {
    T l0 = struve_l_0(z);
    T l1 = struve_l_1(z);

    // d(backward)/d(grad_output) = L_0(z) - L_1(z)/z
    T first_deriv;
    if (std::abs(z) < detail::struve_l_1_bb_zero_tolerance<T>()) {
        // Limit as z -> 0 is 0 (see struve_l_1_backward.h)
        first_deriv = T(0);
    } else {
        first_deriv = l0 - l1 / z;
    }
    T grad_grad_output = gg_z * first_deriv;

    // d(backward)/dz = grad_output * d^2L_1/dz^2
    // d^2L_1/dz^2 = 2/pi + L_1 - L_0/z + 2*L_1/z^2
    T two_over_pi = T(detail::STRUVE_L1_BB_TWO_OVER_PI);
    T d2_l1;
    if (std::abs(z) < detail::struve_l_1_bb_zero_tolerance<T>()) {
        // L_1''(0) = 4/(3*pi)
        constexpr double FOUR_OVER_3PI = 0.4244131815783875620503567023267050;  // 4/(3*pi)
        d2_l1 = T(FOUR_OVER_3PI);
    } else {
        T z2 = z * z;
        d2_l1 = two_over_pi + l1 - l0 / z + T(2) * l1 / z2;
    }
    T grad_z = gg_z * grad_output * d2_l1;

    return {grad_grad_output, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> struve_l_1_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> l0 = struve_l_0(z);
    c10::complex<T> l1 = struve_l_1(z);

    // d(backward)/d(grad_output) = L_0(z) - L_1(z)/z
    c10::complex<T> first_deriv;
    if (std::abs(z) < detail::struve_l_1_bb_zero_tolerance<T>()) {
        first_deriv = c10::complex<T>(T(0), T(0));
    } else {
        first_deriv = l0 - l1 / z;
    }
    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);

    // d^2L_1/dz^2 = 2/pi + L_1 - L_0/z + 2*L_1/z^2
    c10::complex<T> two_over_pi(T(detail::STRUVE_L1_BB_TWO_OVER_PI), T(0));
    c10::complex<T> d2_l1;
    if (std::abs(z) < detail::struve_l_1_bb_zero_tolerance<T>()) {
        // L_1''(0) = 4/(3*pi)
        constexpr double FOUR_OVER_3PI = 0.4244131815783875620503567023267050;
        d2_l1 = c10::complex<T>(T(FOUR_OVER_3PI), T(0));
    } else {
        c10::complex<T> z2 = z * z;
        d2_l1 = two_over_pi + l1 - l0 / z + c10::complex<T>(T(2), T(0)) * l1 / z2;
    }
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(d2_l1);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
