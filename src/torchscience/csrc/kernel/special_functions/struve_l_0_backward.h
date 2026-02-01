#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "struve_l_0.h"
#include "struve_l_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// 2/pi constant for derivative computation
constexpr double STRUVE_L0_BACKWARD_TWO_OVER_PI = 0.6366197723675813430755350534900574;

} // namespace detail

// Real backward: d/dz L_0(z) = (2/pi) + L_1(z)
// Note: Unlike H_0 where d/dz H_0(z) = (2/pi) - H_1(z),
//       for L_0 we have d/dz L_0(z) = (2/pi) + L_1(z) (positive sign)
template <typename T>
T struve_l_0_backward(T grad_output, T z) {
    T l1 = struve_l_1(z);
    T two_over_pi = T(detail::STRUVE_L0_BACKWARD_TWO_OVER_PI);

    // d/dz L_0(z) = (2/pi) + L_1(z)
    T derivative = two_over_pi + l1;

    return grad_output * derivative;
}

// Complex backward
template <typename T>
c10::complex<T> struve_l_0_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> l1 = struve_l_1(z);
    c10::complex<T> two_over_pi(T(detail::STRUVE_L0_BACKWARD_TWO_OVER_PI), T(0));

    // d/dz L_0(z) = (2/pi) + L_1(z)
    c10::complex<T> derivative = two_over_pi + l1;

    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
