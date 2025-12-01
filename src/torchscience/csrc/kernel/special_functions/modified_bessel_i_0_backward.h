#pragma once

#include <c10/util/complex.h>
#include "modified_bessel_i_1.h"

namespace torchscience::kernel::special_functions {

// Real backward: d/dz I₀(z) = I₁(z)
template <typename T>
T modified_bessel_i_0_backward(T grad_output, T z) {
    return grad_output * modified_bessel_i_1(z);
}

// Complex backward: PyTorch expects grad * conj(derivative) for holomorphic functions
template <typename T>
c10::complex<T> modified_bessel_i_0_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> deriv = modified_bessel_i_1(z);
    return grad_output * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
