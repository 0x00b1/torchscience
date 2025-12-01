#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for polynomial negation
//
// backward(grad_output, p) returns:
//   grad_p = -grad_output
//
// backward_backward needs to compute gradients w.r.t. the backward inputs:
//   grad_grad_output[k] = -gg_p[k]  (since d(grad_p)/d(grad_output) = -1)
//   grad_p = 0 (backward doesn't depend on p values)
//
// Parameters:
//   grad_grad_output: output, size N
//   grad_p: output, size N (zeros)
//   gg_p: upstream gradient for grad_p, size N
//   N: coefficient count
template <typename T>
void polynomial_negate_backward_backward(
    T* grad_grad_output,
    T* grad_p,
    const T* gg_p,
    int64_t N
) {
    for (int64_t k = 0; k < N; ++k) {
        grad_grad_output[k] = -gg_p[k];
        grad_p[k] = T(0);
    }
}

// Complex specialization
template <typename T>
void polynomial_negate_backward_backward(
    c10::complex<T>* grad_grad_output,
    c10::complex<T>* grad_p,
    const c10::complex<T>* gg_p,
    int64_t N
) {
    for (int64_t k = 0; k < N; ++k) {
        grad_grad_output[k] = -gg_p[k];
        grad_p[k] = c10::complex<T>(T(0), T(0));
    }
}

} // namespace torchscience::kernel::polynomial
