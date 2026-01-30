#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "airy_ai.h"
#include "airy_ai_backward.h"

namespace torchscience::kernel::special_functions {

// Backward backward pass for airy_ai
// Returns gradients for (grad_output, x)
//
// From the Airy differential equation: Ai''(x) = x * Ai(x)
// So d/dx Ai'(x) = x * Ai(x)
//
// The backward function computes: grad_output * Ai'(x)
// We need:
//   d(backward)/d(grad_output) = Ai'(x)
//   d(backward)/dx = grad_output * Ai''(x) = grad_output * x * Ai(x)

template <typename T>
std::tuple<T, T> airy_ai_backward_backward(T gg_x, T grad_output, T x) {
    T ai = airy_ai(x);
    T aip = airy_ai_prime(x);

    // d(backward)/d(grad_output) = Ai'(x)
    T grad_grad_output = gg_x * aip;

    // d(backward)/dx = grad_output * Ai''(x) = grad_output * x * Ai(x)
    T grad_x = gg_x * grad_output * x * ai;

    return {grad_grad_output, grad_x};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> airy_ai_backward_backward(
    c10::complex<T> gg_x, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> ai = airy_ai(z);
    c10::complex<T> aip = airy_ai_prime(z);

    // d(backward)/d(grad_output) = Ai'(z)
    c10::complex<T> grad_grad_output = gg_x * std::conj(aip);

    // d(backward)/dz = grad_output * Ai''(z) = grad_output * z * Ai(z)
    c10::complex<T> second_deriv = z * ai;
    c10::complex<T> grad_z = gg_x * grad_output * std::conj(second_deriv);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
