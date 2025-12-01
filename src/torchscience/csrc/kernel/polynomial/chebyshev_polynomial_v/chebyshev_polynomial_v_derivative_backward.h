#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Backward pass for Chebyshev V polynomial derivative.
//
// Forward recurrence (from chebyshev_polynomial_v_derivative):
//   d_{deg-1} = 2 * deg * c_{deg}
//   d_k = d_{k+2} + 2*(k+1)*c_{k+1}  for k = deg-2, ..., 1
//   d_0 = 2*c_1 + d_2
//
// We need to compute grad_coeffs given grad_output (gradient w.r.t. d).
//
// The recurrence can be rewritten in terms of how each c_k contributes to outputs:
//   c_{deg} contributes to d_{deg-1}: d_{deg-1} += 2*deg * c_{deg}
//   c_{k+1} contributes to d_k for k >= 1: d_k += 2*(k+1) * c_{k+1}
//   c_1 contributes to d_0: d_0 += 2*c_1
//
// Additionally, through the recurrence d_k = d_{k+2} + ..., the gradient flows:
//   grad_d[k+2] += grad_d[k] for k = deg-2, ..., 1
//   grad_d[2] += grad_d[0]
//
// Parameters:
//   grad_coeffs: output gradient for coefficients, size N
//   grad_output: incoming gradient w.r.t. derivative output, size output_size
//   N: number of input coefficients
//   output_size: size of grad_output (should be max(N-1, 1))
template <typename T>
void chebyshev_polynomial_v_derivative_backward(
    T* grad_coeffs,
    const T* grad_output,
    int64_t N,
    int64_t output_size
) {
    // Initialize all gradients to zero
    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = T(0);
    }

    if (N <= 1) {
        // Derivative of constant is zero, no gradient flows back
        return;
    }

    const int64_t deg = N - 1;

    // We need to accumulate gradients through the recurrence.
    // First, propagate gradients through the recurrence relations.
    // Create temporary gradient accumulator for d values
    // grad_d[k] represents the total gradient w.r.t. d_k after backprop through recurrence

    // Allocate temporary storage for accumulated d gradients
    T* grad_d = new T[output_size]();
    for (int64_t k = 0; k < output_size; ++k) {
        grad_d[k] = grad_output[k];
    }

    // d_0 = 2*c_1 + d_2
    // grad_c[1] += 2 * grad_d[0]
    // grad_d[2] += grad_d[0]
    if (output_size >= 1) {
        if (N >= 2) {
            grad_coeffs[1] += T(2) * grad_d[0];
        }
        if (output_size >= 3) {
            grad_d[2] += grad_d[0];
        }
    }

    // d_k = d_{k+2} + 2*(k+1)*c_{k+1} for k = 1, ..., deg-2
    // Backward: grad_d[k+2] += grad_d[k], grad_c[k+1] += 2*(k+1)*grad_d[k]
    // Process k from 1 to deg-2 in forward order to accumulate grad_d[k+2]
    for (int64_t k = 1; k <= deg - 2; ++k) {
        if (k < output_size) {
            if (k + 1 < N) {
                grad_coeffs[k + 1] += T(2 * (k + 1)) * grad_d[k];
            }
            if (k + 2 < output_size) {
                grad_d[k + 2] += grad_d[k];
            }
        }
    }

    // d_{deg-1} = 2*deg*c_{deg}
    // grad_c[deg] += 2*deg * grad_d[deg-1]
    if (deg - 1 < output_size && deg < N) {
        grad_coeffs[deg] += T(2 * deg) * grad_d[deg - 1];
    }

    delete[] grad_d;
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_v_derivative_backward(
    c10::complex<T>* grad_coeffs,
    const c10::complex<T>* grad_output,
    int64_t N,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));

    for (int64_t k = 0; k < N; ++k) {
        grad_coeffs[k] = zero;
    }

    if (N <= 1) {
        return;
    }

    const int64_t deg = N - 1;

    c10::complex<T>* grad_d = new c10::complex<T>[output_size]();
    for (int64_t k = 0; k < output_size; ++k) {
        grad_d[k] = grad_output[k];
    }

    if (output_size >= 1) {
        if (N >= 2) {
            grad_coeffs[1] += c10::complex<T>(T(2), T(0)) * grad_d[0];
        }
        if (output_size >= 3) {
            grad_d[2] += grad_d[0];
        }
    }

    for (int64_t k = 1; k <= deg - 2; ++k) {
        if (k < output_size) {
            if (k + 1 < N) {
                grad_coeffs[k + 1] += c10::complex<T>(T(2 * (k + 1)), T(0)) * grad_d[k];
            }
            if (k + 2 < output_size) {
                grad_d[k + 2] += grad_d[k];
            }
        }
    }

    if (deg - 1 < output_size && deg < N) {
        grad_coeffs[deg] += c10::complex<T>(T(2 * deg), T(0)) * grad_d[deg - 1];
    }

    delete[] grad_d;
}

} // namespace torchscience::kernel::polynomial
