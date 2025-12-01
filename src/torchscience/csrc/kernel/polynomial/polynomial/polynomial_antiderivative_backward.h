#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for polynomial antiderivative
// Forward: output[0] = constant
//          output[k+1] = coeffs[k] / (k+1) for k = 0..N-1
//
// Backward wrt coeffs:
//   grad_coeffs[k] = grad_output[k+1] / (k+1) for k = 0..N-1
//
// Backward wrt constant:
//   grad_constant = grad_output[0]
//
// This function computes grad_coeffs only; grad_constant is handled at CPU level
template <typename T>
void polynomial_antiderivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    int64_t N
) {
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output[k + 1] / T(k + 1);
    }
}

// Complex specialization
template <typename T>
void polynomial_antiderivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    int64_t N
) {
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = grad_output[k + 1] / c10::complex<T>(T(k + 1), T(0));
    }
}

} // namespace torchscience::kernel::polynomial
