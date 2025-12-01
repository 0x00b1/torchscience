#pragma once

#include <c10/util/complex.h>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Second-order backward for Chebyshev W polynomial multiplication.
//
// The multiplication operation is bilinear: c = f(a, b)
// First backward: grad_a, grad_b = f_backward(grad_c, a, b)
//
// For second-order gradients, we compute:
// - grad_grad_output (gg_output): gradient w.r.t. grad_output from backward
// - grad_a_from_gg: gradient w.r.t. a from second-order terms
// - grad_b_from_gg: gradient w.r.t. b from second-order terms
//
// Since the operation is bilinear (linear in each argument):
// - grad_a[i] = sum_j f_ij * grad_c * b[j]  (linear in b and grad_c)
// - grad_b[j] = sum_i f_ij * grad_c * a[i]  (linear in a and grad_c)
//
// The second-order terms are:
// - d(grad_a)/d(grad_c) -> gg_a flows to grad_grad_output
// - d(grad_b)/d(grad_c) -> gg_b flows to grad_grad_output
// - d(grad_a)/d(b) -> gg_a flows to grad_b
// - d(grad_b)/d(a) -> gg_b flows to grad_a
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   grad_a_from_gg: additional gradient for a from gg_b, size N
//   grad_b_from_gg: additional gradient for b from gg_a, size M
//   gg_a: incoming second-order gradient for a, size N
//   gg_b: incoming second-order gradient for b, size M
//   grad_output: gradient from forward backward, size output_size
//   a: first polynomial coefficients, size N
//   b: second polynomial coefficients, size M
//   N: number of coefficients in a
//   M: number of coefficients in b
//   output_size: size of grad_output and grad_grad_output
template <typename T>
void chebyshev_polynomial_w_multiply_backward_backward(
    T* grad_grad_output,
    T* grad_a_from_gg,
    T* grad_b_from_gg,
    const T* gg_a,
    const T* gg_b,
    const T* grad_output,
    const T* a,
    const T* b,
    int64_t N,
    int64_t M,
    int64_t output_size
) {
    // Initialize outputs to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = T(0);
    }
    for (int64_t i = 0; i < N; ++i) {
        grad_a_from_gg[i] = T(0);
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b_from_gg[j] = T(0);
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Compute second-order terms
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const int64_t sum_idx = i + j;
            const int64_t diff_idx = std::abs(i - j);

            if (i == 0 || j == 0) {
                // Forward backward was:
                // grad_a[i] += grad_c[i+j] * b[j]
                // grad_b[j] += grad_c[i+j] * a[i]
                //
                // Second-order:
                // d(grad_a[i])/d(grad_c[i+j]) = b[j] -> gg_a[i] * b[j] to grad_grad_output[i+j]
                // d(grad_a[i])/d(b[j]) = grad_c[i+j] -> gg_a[i] * grad_c[i+j] to grad_b_from_gg[j]
                // d(grad_b[j])/d(grad_c[i+j]) = a[i] -> gg_b[j] * a[i] to grad_grad_output[i+j]
                // d(grad_b[j])/d(a[i]) = grad_c[i+j] -> gg_b[j] * grad_c[i+j] to grad_a_from_gg[i]
                if (sum_idx < output_size) {
                    grad_grad_output[sum_idx] += gg_a[i] * b[j];
                    grad_grad_output[sum_idx] += gg_b[j] * a[i];
                    grad_b_from_gg[j] += gg_a[i] * grad_output[sum_idx];
                    grad_a_from_gg[i] += gg_b[j] * grad_output[sum_idx];
                }
            } else {
                // Forward backward was:
                // grad_a[i] += 0.5 * (grad_c[i+j] + grad_c[|i-j|]) * b[j]
                // grad_b[j] += 0.5 * (grad_c[i+j] + grad_c[|i-j|]) * a[i]
                //
                // Second-order for gg_a:
                // d(grad_a[i])/d(grad_c[i+j]) = 0.5 * b[j]
                // d(grad_a[i])/d(grad_c[|i-j|]) = 0.5 * b[j]
                // d(grad_a[i])/d(b[j]) = 0.5 * (grad_c[i+j] + grad_c[|i-j|])
                //
                // Second-order for gg_b:
                // d(grad_b[j])/d(grad_c[i+j]) = 0.5 * a[i]
                // d(grad_b[j])/d(grad_c[|i-j|]) = 0.5 * a[i]
                // d(grad_b[j])/d(a[i]) = 0.5 * (grad_c[i+j] + grad_c[|i-j|])
                T grad_c_sum = T(0);
                if (sum_idx < output_size) {
                    grad_grad_output[sum_idx] += T(0.5) * gg_a[i] * b[j];
                    grad_grad_output[sum_idx] += T(0.5) * gg_b[j] * a[i];
                    grad_c_sum += grad_output[sum_idx];
                }
                if (diff_idx < output_size) {
                    grad_grad_output[diff_idx] += T(0.5) * gg_a[i] * b[j];
                    grad_grad_output[diff_idx] += T(0.5) * gg_b[j] * a[i];
                    grad_c_sum += grad_output[diff_idx];
                }
                grad_b_from_gg[j] += T(0.5) * gg_a[i] * grad_c_sum;
                grad_a_from_gg[i] += T(0.5) * gg_b[j] * grad_c_sum;
            }
        }
    }
}

// Complex specialization
template <typename T>
void chebyshev_polynomial_w_multiply_backward_backward(
    c10::complex<T>* grad_grad_output,
    c10::complex<T>* grad_a_from_gg,
    c10::complex<T>* grad_b_from_gg,
    const c10::complex<T>* gg_a,
    const c10::complex<T>* gg_b,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* a,
    const c10::complex<T>* b,
    int64_t N,
    int64_t M,
    int64_t output_size
) {
    const c10::complex<T> zero(T(0), T(0));
    const c10::complex<T> half(T(0.5), T(0));

    // Initialize outputs to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = zero;
    }
    for (int64_t i = 0; i < N; ++i) {
        grad_a_from_gg[i] = zero;
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b_from_gg[j] = zero;
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Compute second-order terms
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const int64_t sum_idx = i + j;
            const int64_t diff_idx = std::abs(i - j);

            if (i == 0 || j == 0) {
                if (sum_idx < output_size) {
                    grad_grad_output[sum_idx] += gg_a[i] * b[j];
                    grad_grad_output[sum_idx] += gg_b[j] * a[i];
                    grad_b_from_gg[j] += gg_a[i] * grad_output[sum_idx];
                    grad_a_from_gg[i] += gg_b[j] * grad_output[sum_idx];
                }
            } else {
                c10::complex<T> grad_c_sum = zero;
                if (sum_idx < output_size) {
                    grad_grad_output[sum_idx] += half * gg_a[i] * b[j];
                    grad_grad_output[sum_idx] += half * gg_b[j] * a[i];
                    grad_c_sum += grad_output[sum_idx];
                }
                if (diff_idx < output_size) {
                    grad_grad_output[diff_idx] += half * gg_a[i] * b[j];
                    grad_grad_output[diff_idx] += half * gg_b[j] * a[i];
                    grad_c_sum += grad_output[diff_idx];
                }
                grad_b_from_gg[j] += half * gg_a[i] * grad_c_sum;
                grad_a_from_gg[i] += half * gg_b[j] * grad_c_sum;
            }
        }
    }
}

} // namespace torchscience::kernel::polynomial
