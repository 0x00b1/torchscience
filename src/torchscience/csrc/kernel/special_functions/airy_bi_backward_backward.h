#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "airy_bi.h"
#include "airy_bi_backward.h"

namespace torchscience::kernel::special_functions {

// Backward backward pass for airy_bi
// Returns gradients for (grad_output, x)
//
// From the Airy differential equation: Bi''(x) = x * Bi(x)
// So d/dx Bi'(x) = x * Bi(x)
//
// The backward function computes: grad_output * Bi'(x)
// We need:
//   d(backward)/d(grad_output) = Bi'(x)
//   d(backward)/dx = grad_output * Bi''(x) = grad_output * x * Bi(x)

template <typename T>
std::tuple<T, T> airy_bi_backward_backward(T gg_x, T grad_output, T x) {
    T bi = airy_bi(x);
    T bip = airy_bi_prime(x);

    // d(backward)/d(grad_output) = Bi'(x)
    T grad_grad_output = gg_x * bip;

    // d(backward)/dx = grad_output * Bi''(x) = grad_output * x * Bi(x)
    T grad_x = gg_x * grad_output * x * bi;

    return {grad_grad_output, grad_x};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> airy_bi_backward_backward(
    c10::complex<T> gg_x, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> bi = airy_bi(z);
    c10::complex<T> bip = airy_bi_prime(z);

    // d(backward)/d(grad_output) = Bi'(z)
    c10::complex<T> grad_grad_output = gg_x * std::conj(bip);

    // d(backward)/dz = grad_output * Bi''(z) = grad_output * z * Bi(z)
    c10::complex<T> second_deriv = z * bi;
    c10::complex<T> grad_z = gg_x * grad_output * std::conj(second_deriv);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
