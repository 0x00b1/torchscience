// src/torchscience/csrc/kernel/test/sum_squares.h
#pragma once

#include <cmath>
#include <limits>

#include <c10/macros/Macros.h>

namespace torchscience::kernel::test {

/**
 * sum_squares: Compute sum of squares of elements.
 *
 * Forward:  f(x) = sum(x_i^2)
 * Backward: df/dx_i = 2 * x_i
 * Hessian:  d2f/dx_i dx_j = 2 * delta_ij
 *
 * This is a simple reduction useful for testing macro infrastructure.
 */

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T sum_squares(const T* data, int64_t n) {
    T result = T(0);
    for (int64_t i = 0; i < n; ++i) {
        result += data[i] * data[i];
    }
    return result;
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void sum_squares_backward(
    T grad_output,
    const T* data,
    int64_t n,
    T* grad_input
) {
    // df/dx_i = 2 * x_i
    for (int64_t i = 0; i < n; ++i) {
        grad_input[i] = grad_output * T(2) * data[i];
    }
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void sum_squares_backward_backward(
    const T* grad_grad_input,
    T grad_output,
    const T* data,
    int64_t n,
    T& grad_grad_output,
    T* new_grad_input
) {
    // The Hessian is 2*I (diagonal matrix with 2's)
    // grad_grad_output = sum_i(grad_grad_input[i] * df/dx_i / grad_output)
    //                  = sum_i(grad_grad_input[i] * 2 * x_i / grad_output)
    // But we need the derivative of the backward w.r.t. grad_output and data.
    //
    // backward: grad_input[i] = grad_output * 2 * data[i]
    //
    // d(grad_input[i])/d(grad_output) = 2 * data[i]
    // d(grad_input[i])/d(data[j]) = grad_output * 2 * delta_ij
    //
    // grad_grad_output = sum_i(grad_grad_input[i] * d(grad_input[i])/d(grad_output))
    //                  = sum_i(grad_grad_input[i] * 2 * data[i])
    // new_grad_input[j] = sum_i(grad_grad_input[i] * d(grad_input[i])/d(data[j]))
    //                   = grad_grad_input[j] * grad_output * 2

    grad_grad_output = T(0);
    for (int64_t i = 0; i < n; ++i) {
        grad_grad_output += grad_grad_input[i] * T(2) * data[i];
        new_grad_input[i] = grad_grad_input[i] * grad_output * T(2);
    }
}

}  // namespace torchscience::kernel::test
