#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for polynomial scaling
// Forward: output[k] = c * p[k]
// Backward has cross-terms since output depends on both c and p:
//   grad_p[k] = c * grad_output[k]
//   grad_c = sum_k(p[k] * grad_output[k]) = dot(p, grad_output)
//
// Parameters:
//   grad_p: output gradient w.r.t. p, size N
//   grad_c: output gradient w.r.t. c (scalar, accumulated)
//   grad_output: upstream gradient, size N
//   p: original polynomial coefficients, size N
//   c: scalar value
//   N: coefficient count
template <typename T>
void polynomial_scale_backward(
    T* grad_p,
    T* grad_c,
    const T* grad_output,
    const T* p,
    T c,
    int64_t N
) {
    T gc = T(0);
    for (int64_t k = 0; k < N; ++k) {
        grad_p[k] = c * grad_output[k];
        gc += p[k] * grad_output[k];
    }
    *grad_c = gc;
}

// Complex specialization
template <typename T>
void polynomial_scale_backward(
    c10::complex<T>* grad_p,
    c10::complex<T>* grad_c,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* p,
    c10::complex<T> c,
    int64_t N
) {
    c10::complex<T> gc(T(0), T(0));
    for (int64_t k = 0; k < N; ++k) {
        grad_p[k] = c * grad_output[k];
        // For complex: use conjugate of p in dot product for proper gradient
        gc += std::conj(p[k]) * grad_output[k];
    }
    *grad_c = gc;
}

} // namespace torchscience::kernel::polynomial
