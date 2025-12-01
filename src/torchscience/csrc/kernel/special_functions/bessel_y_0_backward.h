#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "bessel_y_0.h"
#include "bessel_y_1.h"

namespace torchscience::kernel::special_functions {

// Real backward: d/dz Y₀(z) = -Y₁(z)
template <typename T>
T bessel_y_0_backward(T grad_output, T z) {
    // Y₀ is only defined for z > 0
    if (z <= T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T y1 = bessel_y_1(z);
    T derivative = -y1;
    return grad_output * derivative;
}

// Complex backward
template <typename T>
c10::complex<T> bessel_y_0_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> y1 = bessel_y_1(z);
    c10::complex<T> derivative = -y1;
    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
