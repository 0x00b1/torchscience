#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "bessel_j_0.h"
#include "bessel_j_1.h"
#include "bessel_j_0_backward.h"

namespace torchscience::kernel::special_functions {

// Real backward_backward
// Returns gradients for (grad_output, z)
// d/dz J₀(z) = -J₁(z)
// d²/dz² J₀(z) = -J₁'(z) = -(J₀(z) - J₁(z)/z) = J₁(z)/z - J₀(z)
template <typename T>
std::tuple<T, T> bessel_j_0_backward_backward(T gg_z, T grad_output, T z) {
    T j0 = bessel_j_0(z);
    T j1 = bessel_j_1(z);

    // d(backward)/d(grad_output) = -J₁(z)
    T first_deriv = -j1;
    T grad_grad_output = gg_z * first_deriv;

    // d(backward)/dz = grad_output * d²J₀/dz²
    // d²J₀/dz² = J₁(z)/z - J₀(z)
    T d2_j0;
    if (std::abs(z) < detail::bessel_j0_zero_tolerance<T>()) {
        // Use series expansion for limit:
        // J₀(z) = 1 - z²/4 + O(z⁴), J₁(z) = z/2 - z³/16 + O(z⁵)
        // J₁(z)/z = 1/2 - z²/16 + O(z⁴)
        // d²J₀/dz² at z=0: J₁(0)/z - J₀(0) = 1/2 - 1 = -1/2
        d2_j0 = T(-0.5);
    } else {
        d2_j0 = j1 / z - j0;
    }
    T grad_z = gg_z * grad_output * d2_j0;

    return {grad_grad_output, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> bessel_j_0_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> j0 = bessel_j_0(z);
    c10::complex<T> j1 = bessel_j_1(z);

    c10::complex<T> first_deriv = -j1;
    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);

    c10::complex<T> d2_j0;
    if (std::abs(z) < detail::bessel_j0_zero_tolerance<T>()) {
        // J₀''(0) = -1/2 (see real version for derivation)
        d2_j0 = c10::complex<T>(T(-0.5), T(0));
    } else {
        d2_j0 = j1 / z - j0;
    }
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(d2_j0);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
