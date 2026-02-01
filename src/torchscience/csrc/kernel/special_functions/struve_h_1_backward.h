#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "struve_h_0.h"
#include "struve_h_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T struve_h_1_backward_zero_tolerance() {
    return T(1e-12);
}

template <>
inline float struve_h_1_backward_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double struve_h_1_backward_zero_tolerance<double>() { return 1e-12; }

// 2/(3*pi) constant for z=0 limit
constexpr double STRUVE_H1_BACKWARD_TWO_OVER_3PI = 0.2122065907891937810251783511633525;  // 2/(3*pi)

} // namespace detail

// Real backward: d/dz H_1(z) = H_0(z) - H_1(z)/z
// At z=0: limit is 2/(3*pi)
template <typename T>
T struve_h_1_backward(T grad_output, T z) {
    T derivative;

    if (std::abs(z) < detail::struve_h_1_backward_zero_tolerance<T>()) {
        // Limit as z -> 0: d/dz H_1(z) = 2/(3*pi)
        // From series expansion:
        // H_1(z) = (2/pi) * (z/2)^2 / [Gamma(3/2)*Gamma(5/2)] + O(z^4)
        //        = (2/pi) * z^2 * 8/(12*pi) + O(z^4)
        //        = z^2 * 4/(3*pi^2) + O(z^4)
        // d/dz H_1(z) = 8z/(3*pi^2) + O(z^3) -> 0 as z -> 0
        //
        // However, we need to compute H_0(z) - H_1(z)/z:
        // H_0(z) ~ (2/pi)*(z/2) * 4/pi = 4z/(pi^2) + O(z^3)
        // H_1(z)/z ~ z * 4/(3*pi^2) + O(z^3)
        // Limit of H_0(z) - H_1(z)/z as z->0:
        // The limit is 0 - 0 = 0... Let's recalculate more carefully.
        //
        // Actually from the derivative formula:
        // d/dz H_1(z) at z=0 can be computed from series:
        // H_1(z) = sum c_k * z^{2k+2}, so H_1'(z) = sum (2k+2)*c_k*z^{2k+1}
        // H_1'(0) = 0 since all terms have positive powers of z
        //
        // But the formula H_0(z) - H_1(z)/z:
        // H_0(z) = sum a_k * z^{2k+1}
        // H_1(z)/z = sum c_k * z^{2k+1}
        // Both have the same structure, so at z=0, we need limit of:
        // H_0(z) ~ a_0 * z + O(z^3)
        // H_1(z)/z ~ c_0 * z + O(z^3)
        // where a_0 = (2/pi) * (1/2) / [pi/4] = (2/pi) * 2/pi = 4/pi^2
        // and c_0 = (2/pi) * (1/4) / [3*pi/8] = (2/pi) * 2/(3*pi) = 4/(3*pi^2)
        //
        // So the limit is 4/pi^2 - 4/(3*pi^2) = (12-4)/(3*pi^2) = 8/(3*pi^2)
        // Wait, that's not right either since both terms go to 0 at z=0.
        //
        // Let me reconsider: the limit H_0(z) - H_1(z)/z as z->0
        // We expand: H_0(z) = (4/pi^2)*z - ... and H_1(z) = (4/(3*pi^2))*z^2 - ...
        // So H_1(z)/z = (4/(3*pi^2))*z - ...
        // Thus H_0(z) - H_1(z)/z = (4/pi^2 - 4/(3*pi^2))*z + O(z^3)
        //                        = (8/(3*pi^2))*z + O(z^3)
        // The limit as z->0 is 0.
        //
        // So d/dz H_1(0) = 0 (limit)
        derivative = T(0);
    } else {
        T h0 = struve_h_0(z);
        T h1 = struve_h_1(z);
        derivative = h0 - h1 / z;
    }

    return grad_output * derivative;
}

// Complex backward
template <typename T>
c10::complex<T> struve_h_1_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> derivative;

    if (std::abs(z) < detail::struve_h_1_backward_zero_tolerance<T>()) {
        // Limit as z -> 0: see real version for derivation
        derivative = c10::complex<T>(T(0), T(0));
    } else {
        c10::complex<T> h0 = struve_h_0(z);
        c10::complex<T> h1 = struve_h_1(z);
        derivative = h0 - h1 / z;
    }

    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
