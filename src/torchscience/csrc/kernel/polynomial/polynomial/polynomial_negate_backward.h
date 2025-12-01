#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for polynomial negation
// grad_p[k] = -grad_output[k]
// Negation is its own inverse, so backward is the same as forward
//
// Parameters:
//   grad_p: output gradient w.r.t. p, size N
//   grad_output: upstream gradient, size N
//   N: coefficient count
template <typename T>
void polynomial_negate_backward(
    T* grad_p,
    const T* grad_output,
    int64_t N
) {
    for (int64_t k = 0; k < N; ++k) {
        grad_p[k] = -grad_output[k];
    }
}

// Complex specialization
template <typename T>
void polynomial_negate_backward(
    c10::complex<T>* grad_p,
    const c10::complex<T>* grad_output,
    int64_t N
) {
    for (int64_t k = 0; k < N; ++k) {
        grad_p[k] = -grad_output[k];
    }
}

} // namespace torchscience::kernel::polynomial
