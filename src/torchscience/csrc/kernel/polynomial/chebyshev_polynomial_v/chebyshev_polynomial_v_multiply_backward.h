#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Backward pass for Chebyshev V polynomial multiplication.
//
// Forward used linearization formula:
// V_m(x) * V_n(x) = 0.5 * (V_{m+n}(x) + V_{|m-n|}(x))  for m,n >= 1
// V_0(x) * V_k(x) = V_k(x)
//
// For backward, we trace how each output coefficient c[k] depends on inputs:
// - If i == 0 or j == 0: c[i+j] += a[i] * b[j]
// - Otherwise: c[i+j] += 0.5 * a[i] * b[j] and c[|i-j|] += 0.5 * a[i] * b[j]
//
// Backward pass reverses this:
// grad_a[i] = sum over j: contribution from grad_c based on the formula
// grad_b[j] = sum over i: contribution from grad_c based on the formula
//
// Parameters:
//   grad_a: output gradient for a, size N
//   grad_b: output gradient for b, size M
//   grad_output: incoming gradient, size output_size
//   a: first polynomial coefficients, size N
//   b: second polynomial coefficients, size M
//   N: number of coefficients in a
//   M: number of coefficients in b
//   output_size: size of grad_output
template <typename T>
void chebyshev_polynomial_v_multiply_backward(
    T* grad_a,
    T* grad_b,
    const T* grad_output,
    const T* a,
    const T* b,
    int64_t N,
    int64_t M,
    int64_t output_size
) {
    // Initialize gradients to zero
    for (int64_t i = 0; i < N; ++i) {
        grad_a[i] = T(0);
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b[j] = T(0);
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Reverse the linearization formula
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const int64_t sum_idx = i + j;
            const int64_t diff_idx = std::abs(i - j);

            if (i == 0 || j == 0) {
                // Forward: c[i+j] += a[i] * b[j]
                // Backward: grad_a[i] += grad_c[i+j] * b[j]
                //           grad_b[j] += grad_c[i+j] * a[i]
                if (sum_idx < output_size) {
                    grad_a[i] += grad_output[sum_idx] * b[j];
                    grad_b[j] += grad_output[sum_idx] * a[i];
                }
            } else {
                // Forward: c[i+j] += 0.5 * a[i] * b[j]
                //          c[|i-j|] += 0.5 * a[i] * b[j]
                // Backward: grad_a[i] += 0.5 * (grad_c[i+j] + grad_c[|i-j|]) * b[j]
                //           grad_b[j] += 0.5 * (grad_c[i+j] + grad_c[|i-j|]) * a[i]
                T grad_sum = T(0);
                if (sum_idx < output_size) {
                    grad_sum += grad_output[sum_idx];
                }
                if (diff_idx < output_size) {
                    grad_sum += grad_output[diff_idx];
                }
                grad_a[i] += T(0.5) * grad_sum * b[j];
                grad_b[j] += T(0.5) * grad_sum * a[i];
            }
        }
    }
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_v_multiply_backward(
    c10::complex<T>* grad_a,
    c10::complex<T>* grad_b,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* a,
    const c10::complex<T>* b,
    int64_t N,
    int64_t M,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> half(T(0.5), T(0));

    // Initialize gradients to zero
    for (int64_t i = 0; i < N; ++i) {
        grad_a[i] = zero;
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b[j] = zero;
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Reverse the linearization formula
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const int64_t sum_idx = i + j;
            const int64_t diff_idx = std::abs(i - j);

            if (i == 0 || j == 0) {
                if (sum_idx < output_size) {
                    grad_a[i] += grad_output[sum_idx] * b[j];
                    grad_b[j] += grad_output[sum_idx] * a[i];
                }
            } else {
                c10::complex<T> grad_sum = zero;
                if (sum_idx < output_size) {
                    grad_sum += grad_output[sum_idx];
                }
                if (diff_idx < output_size) {
                    grad_sum += grad_output[diff_idx];
                }
                grad_a[i] += half * grad_sum * b[j];
                grad_b[j] += half * grad_sum * a[i];
            }
        }
    }
}

} // namespace torchscience::kernel::polynomial
