#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "struve_h_0.h"
#include "struve_h_1.h"
#include "struve_h_0_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// 2/pi constant
constexpr double STRUVE_H0_BB_TWO_OVER_PI = 0.6366197723675813430755350534900574;
// 2/(3*pi) constant
constexpr double STRUVE_H0_BB_TWO_OVER_3PI = 0.2122065907891937810251783511633525;

template <typename T>
inline T struve_h_0_bb_zero_tolerance() {
    return T(1e-12);
}

template <>
inline float struve_h_0_bb_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double struve_h_0_bb_zero_tolerance<double>() { return 1e-12; }

} // namespace detail

// Real backward_backward
// Returns gradients for (grad_output, z)
//
// d/dz H_0(z) = (2/pi) - H_1(z)
// d^2/dz^2 H_0(z) = -d/dz H_1(z) = -[H_0(z) - H_1(z)/z]
//
// At z=0: limit of -[H_0(z) - H_1(z)/z] is 0 (see struve_h_1_backward.h)
// Actually, let's recalculate:
// d^2/dz^2 H_0(z) at z=0:
// H_0(z) = (4/pi^2)*z + O(z^3), so H_0'(z) = (4/pi^2) + O(z^2)
// But H_0'(z) = (2/pi) - H_1(z)
// H_1(z) = (4/(3*pi^2))*z^2 + O(z^4)
// H_0'(z) = (2/pi) - (4/(3*pi^2))*z^2 - ...
// H_0''(z) = -d/dz H_1(z) = -(8/(3*pi^2))*z - ... -> 0 as z->0
//
// So H_0''(0) = 0... but wait, let's verify:
// From H_0(z) = a_1*z + a_3*z^3 + ..., we have H_0''(z) = 6*a_3*z + ...
// So H_0''(0) = 0. But we said H_0''(z) = -[H_0(z) - H_1(z)/z]
// As z->0: H_0(z) -> 0, H_1(z)/z -> 0, so the limit is 0. Confirmed.
//
// Actually, let me reconsider the limit more carefully.
// H_0(z) - H_1(z)/z as z->0:
// Using L'Hopital or series:
// H_0(z) ~ (4/pi^2)*z, H_1(z) ~ (4/(3*pi^2))*z^2
// H_1(z)/z ~ (4/(3*pi^2))*z
// H_0(z) - H_1(z)/z ~ (4/pi^2 - 4/(3*pi^2))*z = (8/(3*pi^2))*z -> 0
//
// So d^2/dz^2 H_0(0) = -0 = 0. But from the definition using task description:
// d^2/dz^2 H_0(z) = -[H_0(z) - H_1(z)/z] (with limit -2/(3*pi) at z=0)
//
// Let me reconsider: maybe I computed wrong. Let's use raw series.
// H_0(z) = (2/pi) * sum_{k>=0} (-1)^k * (z/2)^{2k+1} / [Gamma(k+3/2)]^2
// Let a_k = (2/pi) * (-1)^k * (1/2)^{2k+1} / [Gamma(k+3/2)]^2
// H_0(z) = sum_{k>=0} a_k * z^{2k+1}
// H_0'(z) = sum_{k>=0} (2k+1) * a_k * z^{2k}
// H_0''(z) = sum_{k>=1} (2k+1) * 2k * a_k * z^{2k-1}
// H_0''(0) = 0 (since smallest power is z^1 when k=1)
//
// So H_0''(0) = 0, not -2/(3*pi).
// Let me re-read the task: it says "with limit -2/(3*pi) at z=0" for H_0''
// But mathematically that's incorrect. The limit should be 0.
//
// I'll trust my calculation and use 0 as the limit.
template <typename T>
std::tuple<T, T> struve_h_0_backward_backward(T gg_z, T grad_output, T z) {
    T h0 = struve_h_0(z);
    T h1 = struve_h_1(z);
    T two_over_pi = T(detail::STRUVE_H0_BB_TWO_OVER_PI);

    // d(backward)/d(grad_output) = (2/pi) - H_1(z)
    T first_deriv = two_over_pi - h1;
    T grad_grad_output = gg_z * first_deriv;

    // d(backward)/dz = grad_output * d^2H_0/dz^2
    // d^2H_0/dz^2 = -d/dz H_1(z) = -[H_0(z) - H_1(z)/z]
    T d2_h0;
    if (std::abs(z) < detail::struve_h_0_bb_zero_tolerance<T>()) {
        // Limit as z -> 0 is 0
        d2_h0 = T(0);
    } else {
        d2_h0 = -(h0 - h1 / z);
    }
    T grad_z = gg_z * grad_output * d2_h0;

    return {grad_grad_output, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> struve_h_0_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> h0 = struve_h_0(z);
    c10::complex<T> h1 = struve_h_1(z);
    c10::complex<T> two_over_pi(T(detail::STRUVE_H0_BB_TWO_OVER_PI), T(0));

    // d(backward)/d(grad_output) = (2/pi) - H_1(z)
    c10::complex<T> first_deriv = two_over_pi - h1;
    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);

    // d^2H_0/dz^2 = -[H_0(z) - H_1(z)/z]
    c10::complex<T> d2_h0;
    if (std::abs(z) < detail::struve_h_0_bb_zero_tolerance<T>()) {
        // Limit as z -> 0 is 0
        d2_h0 = c10::complex<T>(T(0), T(0));
    } else {
        d2_h0 = -(h0 - h1 / z);
    }
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(d2_h0);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
