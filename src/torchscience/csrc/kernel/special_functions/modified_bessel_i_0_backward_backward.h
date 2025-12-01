#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "modified_bessel_i_0.h"
#include "modified_bessel_i_1.h"
#include "modified_bessel_i_1_backward.h"

namespace torchscience::kernel::special_functions {

// Real backward_backward
// Returns gradients for (grad_output, z)
template <typename T>
std::tuple<T, T> modified_bessel_i_0_backward_backward(T gg_z, T grad_output, T z) {
    T i0 = modified_bessel_i_0(z);
    T i1 = modified_bessel_i_1(z);

    // d(backward)/d(grad_output) = I₁(z)
    T grad_grad_output = gg_z * i1;

    // d(backward)/dz = grad_output * d²I₀/dz² = grad_output * (I₀(z) - I₁(z)/z)
    T d2_i0;
    if (std::abs(z) < detail::bessel_zero_tolerance<T>()) {
        // lim[I₀(z) - I₁(z)/z] as z→0 = 1 - 0.5 = 0.5
        d2_i0 = T(0.5);
    } else {
        d2_i0 = i0 - i1 / z;
    }
    T grad_z = gg_z * grad_output * d2_i0;

    return {grad_grad_output, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> modified_bessel_i_0_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> i0 = modified_bessel_i_0(z);
    c10::complex<T> i1 = modified_bessel_i_1(z);

    c10::complex<T> grad_grad_output = gg_z * std::conj(i1);

    c10::complex<T> d2_i0;
    if (std::abs(z) < detail::bessel_zero_tolerance<T>()) {
        d2_i0 = c10::complex<T>(T(0.5), T(0));
    } else {
        d2_i0 = i0 - i1 / z;
    }
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(d2_i0);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
