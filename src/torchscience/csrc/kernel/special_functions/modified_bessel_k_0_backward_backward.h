#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "modified_bessel_k_0.h"
#include "modified_bessel_k_1.h"
#include "modified_bessel_k_0_backward.h"

namespace torchscience::kernel::special_functions {

// Real backward_backward
// Returns gradients for (grad_output, z)
// First derivative: d/dz K_0(z) = -K_1(z)
// Second derivative: d^2/dz^2 K_0(z) = -d/dz K_1(z) = -(-K_0(z) - K_1(z)/z) = K_0(z) + K_1(z)/z
template <typename T>
std::tuple<T, T> modified_bessel_k_0_backward_backward(T gg_z, T grad_output, T z) {
    // K_0 is only defined for z > 0
    if (z <= T(0)) {
        return {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
    }

    T k0 = modified_bessel_k_0(z);
    T k1 = modified_bessel_k_1(z);

    // d(backward)/d(grad_output) = -K_1(z)
    T first_deriv = -k1;
    T grad_grad_output = gg_z * first_deriv;

    // d(backward)/dz = grad_output * d^2K_0/dz^2
    // d^2K_0/dz^2 = K_0(z) + K_1(z)/z
    T d2_k0;
    if (z < detail::bessel_k_zero_tolerance<T>()) {
        // Near z=0+:
        // K_0(z) ~ -ln(z/2) - gamma
        // K_1(z)/z ~ 1/z^2
        // d^2K_0/dz^2 ~ 1/z^2 (K_1/z term dominates)
        d2_k0 = T(1) / (z * z);
    } else {
        d2_k0 = k0 + k1 / z;
    }
    T grad_z = gg_z * grad_output * d2_k0;

    return {grad_grad_output, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> modified_bessel_k_0_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> k0 = modified_bessel_k_0(z);
    c10::complex<T> k1 = modified_bessel_k_1(z);

    // d(backward)/d(grad_output) = -K_1(z)
    c10::complex<T> first_deriv = -k1;
    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);

    c10::complex<T> d2_k0;
    if (std::abs(z) < detail::bessel_k_zero_tolerance<T>()) {
        d2_k0 = c10::complex<T>(T(1), T(0)) / (z * z);
    } else {
        d2_k0 = k0 + k1 / z;
    }
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(d2_k0);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
