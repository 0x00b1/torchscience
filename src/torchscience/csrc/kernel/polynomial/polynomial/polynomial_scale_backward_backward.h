#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for polynomial scaling
//
// Forward: output[k] = c * p[k]
// backward(grad_output, p, c) returns:
//   grad_p[k] = c * grad_output[k]
//   grad_c = sum_k(p[k] * grad_output[k])
//
// backward_backward needs to compute gradients w.r.t. the backward inputs.
// Let gg_p and gg_c be the upstream gradients for grad_p and grad_c respectively.
//
// For grad_grad_output:
//   d(grad_p[k])/d(grad_output[k]) = c, so contributes c * gg_p[k]
//   d(grad_c)/d(grad_output[k]) = p[k], so contributes p[k] * gg_c
//   => grad_grad_output[k] = c * gg_p[k] + p[k] * gg_c
//
// For grad_p:
//   d(grad_p[k])/d(p[k]) = 0 (grad_p doesn't depend on p)
//   d(grad_c)/d(p[k]) = grad_output[k], so contributes grad_output[k] * gg_c
//   => grad_p[k] = grad_output[k] * gg_c
//
// For grad_c:
//   d(grad_p[k])/d(c) = grad_output[k], contributes sum_k(grad_output[k] * gg_p[k])
//   d(grad_c)/d(c) = 0 (grad_c doesn't depend on c itself)
//   => grad_c = sum_k(grad_output[k] * gg_p[k]) = dot(grad_output, gg_p)
//
// Parameters:
//   grad_grad_output: output, size N
//   grad_p: output, size N
//   grad_c: output scalar
//   gg_p: upstream gradient for grad_p, size N
//   gg_c: upstream gradient for grad_c (scalar)
//   grad_output: original backward input, size N
//   p: original polynomial, size N
//   c: original scalar
//   N: coefficient count
template <typename T>
void polynomial_scale_backward_backward(
    T* grad_grad_output,
    T* grad_p,
    T* grad_c,
    const T* gg_p,
    T gg_c,
    const T* grad_output,
    const T* p,
    T c,
    int64_t N
) {
    T gc = T(0);
    for (int64_t k = 0; k < N; ++k) {
        // grad_grad_output[k] = c * gg_p[k] + p[k] * gg_c
        grad_grad_output[k] = c * gg_p[k] + p[k] * gg_c;
        // grad_p[k] = grad_output[k] * gg_c
        grad_p[k] = grad_output[k] * gg_c;
        // grad_c accumulates dot(grad_output, gg_p)
        gc += grad_output[k] * gg_p[k];
    }
    *grad_c = gc;
}

// Complex specialization
template <typename T>
void polynomial_scale_backward_backward(
    c10::complex<T>* grad_grad_output,
    c10::complex<T>* grad_p,
    c10::complex<T>* grad_c,
    const c10::complex<T>* gg_p,
    c10::complex<T> gg_c,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* p,
    c10::complex<T> c,
    int64_t N
) {
    c10::complex<T> gc(T(0), T(0));
    for (int64_t k = 0; k < N; ++k) {
        grad_grad_output[k] = c * gg_p[k] + p[k] * gg_c;
        grad_p[k] = grad_output[k] * gg_c;
        gc += std::conj(grad_output[k]) * gg_p[k];
    }
    *grad_c = gc;
}

} // namespace torchscience::kernel::polynomial
