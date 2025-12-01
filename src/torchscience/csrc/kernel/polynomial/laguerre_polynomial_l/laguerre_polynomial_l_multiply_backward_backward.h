#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace torchscience::kernel::polynomial {

// Forward declaration of helpers
template <typename T>
T laguerre_binomial(int64_t n, int64_t k);

template <typename T>
T laguerre_factorial(int64_t n);

// Forward declaration of multiply
template <typename T>
void laguerre_polynomial_l_multiply(T* output, const T* a, const T* b, int64_t N, int64_t M);

// Second-order backward for Laguerre polynomial multiplication
//
// The multiply operation is bilinear: c = f(a, b)
// First-order backward: grad_a = g(grad_c, b), grad_b = h(grad_c, a)
//
// Second-order backward computes:
//   grad_grad_output from (gg_a, gg_b)
//   grad_a_new from gg_b (cross term)
//   grad_b_new from gg_a (cross term)
//
// Since multiply is bilinear:
// grad_grad_output = multiply(gg_a, b) + multiply(a, gg_b)
//
// Parameters:
//   grad_grad_output: output gradient w.r.t. grad_output, size output_size
//   grad_a_new: additional gradient w.r.t. a (from gg_b cross-term), size N
//   grad_b_new: additional gradient w.r.t. b (from gg_a cross-term), size M
//   gg_a: second-order gradient w.r.t. a, size N
//   gg_b: second-order gradient w.r.t. b, size M
//   grad_output: original grad_output (unused for bilinear, but needed for API)
//   a: first polynomial coefficients, size N
//   b: second polynomial coefficients, size M
//   N: number of coefficients in a
//   M: number of coefficients in b
//   output_size: size of grad_grad_output
template <typename T>
void laguerre_polynomial_l_multiply_backward_backward(
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
    (void)grad_output;  // Not used for bilinear operations

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

    // grad_grad_output = multiply(gg_a, b) + multiply(a, gg_b)
    // Use temporary storage
    std::vector<T> temp1(output_size, T(0));
    std::vector<T> temp2(output_size, T(0));

    laguerre_polynomial_l_multiply(temp1.data(), gg_a, b, N, M);
    laguerre_polynomial_l_multiply(temp2.data(), a, gg_b, N, M);

    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = temp1[k] + temp2[k];
    }

    // grad_a_new and grad_b_new are zero for bilinear operations
    // (no second-order dependencies on a and b through the mixed terms)
}

// Complex specialization
template <typename T>
void laguerre_polynomial_l_multiply_backward_backward(
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
    (void)grad_output;
    const c10::complex<T> zero(T(0), T(0));

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

    std::vector<c10::complex<T>> temp1(output_size, zero);
    std::vector<c10::complex<T>> temp2(output_size, zero);

    laguerre_polynomial_l_multiply(temp1.data(), gg_a, b, N, M);
    laguerre_polynomial_l_multiply(temp2.data(), a, gg_b, N, M);

    for (int64_t k = 0; k < output_size; ++k) {
        grad_grad_output[k] = temp1[k] + temp2[k];
    }
}

} // namespace torchscience::kernel::polynomial
