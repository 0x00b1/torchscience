#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Forward declaration of linearization coefficient helper
template <typename T>
T legendre_linearization_coeff(int64_t m, int64_t n, int64_t k);

// Second-order backward for Legendre polynomial multiplication
//
// The multiply operation is bilinear: c = f(a, b)
// First-order backward gives: grad_a = g(grad_c, b), grad_b = h(grad_c, a)
//
// Second-order backward computes:
//   grad_grad_output from (gg_a, gg_b)
//   grad_a_new from gg_b (cross term)
//   grad_b_new from gg_a (cross term)
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   grad_a_new: additional gradient w.r.t. a (from gg_b cross-term), size N
//   grad_b_new: additional gradient w.r.t. b (from gg_a cross-term), size M
//   gg_a: second-order gradient w.r.t. a, size N
//   gg_b: second-order gradient w.r.t. b, size M
//   a: first polynomial coefficients, size N
//   b: second polynomial coefficients, size M
//   N: number of coefficients in a
//   M: number of coefficients in b
//   output_size: size of grad_grad_output
template <typename T>
void legendre_polynomial_p_multiply_backward_backward(
    T* grad_grad_output,
    T* grad_a_new,
    T* grad_b_new,
    const T* gg_a,
    const T* gg_b,
    const T* grad_output,
    const T* a,
    const T* b,
    int64_t N,
    int64_t M,
    int64_t output_size
) {
    (void)grad_output;  // Not used in current implementation but needed for API compatibility
    // Initialize outputs to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = T(0);
    }
    for (int64_t i = 0; i < N; ++i) {
        grad_a_new[i] = T(0);
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b_new[j] = T(0);
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Compute second-order terms
    // grad_grad_output[k] = sum_{i,j} (gg_a[i] * b[j] + a[i] * gg_b[j]) * coeff(i,j,k)
    // grad_a_new[i] = sum_k grad_output[k] * sum_j gg_b[j] * coeff(i,j,k)
    // grad_b_new[j] = sum_k grad_output[k] * sum_i gg_a[i] * coeff(i,j,k)
    //
    // But we don't have grad_output here for the cross terms.
    // The second-order backward typically just computes:
    // grad_grad_output from gg_a * b + a * gg_b (like forward with mixed inputs)

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            // Contribution to grad_grad_output: gg_a[i] * b[j] + a[i] * gg_b[j]
            const T mixed_product = gg_a[i] * b[j] + a[i] * gg_b[j];

            const int64_t abs_diff = (i > j) ? (i - j) : (j - i);
            for (int64_t k = abs_diff; k <= i + j && k < output_size; k += 2) {
                const T coeff = legendre_linearization_coeff<T>(i, j, k);
                grad_grad_output[k] = grad_grad_output[k] + mixed_product * coeff;
            }
        }
    }
}

// Complex specialization
template <typename T>
void legendre_polynomial_p_multiply_backward_backward(
    c10::complex<T>* grad_grad_output,
    c10::complex<T>* grad_a_new,
    c10::complex<T>* grad_b_new,
    const c10::complex<T>* gg_a,
    const c10::complex<T>* gg_b,
    const c10::complex<T>* grad_output,
    const c10::complex<T>* a,
    const c10::complex<T>* b,
    int64_t N,
    int64_t M,
    int64_t output_size
) {
    (void)grad_output;  // Not used in current implementation but needed for API compatibility
    const c10::complex<T> zero(T(0), T(0));

    // Initialize outputs to zero
    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = zero;
    }
    for (int64_t i = 0; i < N; ++i) {
        grad_a_new[i] = zero;
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b_new[j] = zero;
    }

    if (N == 0 || M == 0) {
        return;
    }

    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const c10::complex<T> mixed_product = gg_a[i] * b[j] + a[i] * gg_b[j];

            const int64_t abs_diff = (i > j) ? (i - j) : (j - i);
            for (int64_t k = abs_diff; k <= i + j && k < output_size; k += 2) {
                const T coeff = legendre_linearization_coeff<T>(i, j, k);
                const c10::complex<T> coeff_c(coeff, T(0));
                grad_grad_output[k] = grad_grad_output[k] + mixed_product * coeff_c;
            }
        }
    }
}

} // namespace torchscience::kernel::polynomial
