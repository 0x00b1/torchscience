#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "struve_h_0.h"
#include "struve_h_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// 2/pi constant for derivative computation
constexpr double STRUVE_H0_BACKWARD_TWO_OVER_PI = 0.6366197723675813430755350534900574;

} // namespace detail

// Real backward: d/dz H_0(z) = (2/pi) - H_1(z)
template <typename T>
T struve_h_0_backward(T grad_output, T z) {
    T h1 = struve_h_1(z);
    T two_over_pi = T(detail::STRUVE_H0_BACKWARD_TWO_OVER_PI);

    // d/dz H_0(z) = (2/pi) - H_1(z)
    T derivative = two_over_pi - h1;

    return grad_output * derivative;
}

// Complex backward
template <typename T>
c10::complex<T> struve_h_0_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> h1 = struve_h_1(z);
    c10::complex<T> two_over_pi(T(detail::STRUVE_H0_BACKWARD_TWO_OVER_PI), T(0));

    // d/dz H_0(z) = (2/pi) - H_1(z)
    c10::complex<T> derivative = two_over_pi - h1;

    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
