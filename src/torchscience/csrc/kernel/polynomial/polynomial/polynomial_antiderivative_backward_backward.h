#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Second-order backward for polynomial antiderivative
// The antiderivative operator is linear, so second-order gradients are straightforward.
//
// From backward: grad_coeffs[k] = grad_output[k+1] / (k+1)
//
// Second-order wrt gg_coeffs:
//   grad_grad_output[k+1] = gg_coeffs[k] / (k+1) for k = 0..N-1
//   grad_grad_output[0] = 0 (no dependency on coeffs)
//
// Second-order wrt gg_constant:
//   grad_grad_output[0] = gg_constant (direct)
//
// This function computes the full grad_grad_output; gg_constant contribution
// is added at the CPU level.
template <typename T>
void polynomial_antiderivative_backward_backward(
    T* grad_grad_output,
    const T* gg_coeffs,
    int64_t N,
    int64_t output_size  // = N + 1
) {
    grad_grad_output[0] = T(0);  // Will be overwritten/added at CPU level for gg_constant
    for (int64_t k = 0; k < N; ++k) {
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] = gg_coeffs[k] / T(k + 1);
        }
    }
}

// Complex specialization
template <typename T>
void polynomial_antiderivative_backward_backward(
    c10::complex<T>* grad_grad_output,
    const c10::complex<T>* gg_coeffs,
    int64_t N,
    int64_t output_size  // = N + 1
) {
    grad_grad_output[0] = c10::complex<T>(T(0), T(0));  // Will be handled at CPU level
    for (int64_t k = 0; k < N; ++k) {
        if (k + 1 < output_size) {
            grad_grad_output[k + 1] = gg_coeffs[k] / c10::complex<T>(T(k + 1), T(0));
        }
    }
}

} // namespace torchscience::kernel::polynomial
