#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <cstdlib>

namespace torchscience::kernel::polynomial {

// Forward declaration of linearization coefficient helper
template <typename T>
T legendre_linearization_coeff(int64_t m, int64_t n, int64_t k);

// Backward pass for Legendre polynomial multiplication
//
// Forward: c[k] = sum_{i,j} a[i] * b[j] * coeff(i,j,k)
//
// Backward:
//   grad_a[i] = sum_k grad_c[k] * sum_j b[j] * coeff(i,j,k)
//   grad_b[j] = sum_k grad_c[k] * sum_i a[i] * coeff(i,j,k)
//
// Parameters:
//   grad_a: output gradient w.r.t. first polynomial, size N
//   grad_b: output gradient w.r.t. second polynomial, size M
//   grad_output: incoming gradient, size output_size
//   a: first polynomial coefficients, size N
//   b: second polynomial coefficients, size M
//   N: number of coefficients in a
//   M: number of coefficients in b
//   output_size: size of grad_output
template <typename T>
void legendre_polynomial_p_multiply_backward(
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

    // Compute gradients using the linearization formula
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const int64_t abs_diff = (i > j) ? (i - j) : (j - i);
            for (int64_t k = abs_diff; k <= i + j && k < output_size; k += 2) {
                const T coeff = legendre_linearization_coeff<T>(i, j, k);
                grad_a[i] = grad_a[i] + grad_output[k] * b[j] * coeff;
                grad_b[j] = grad_b[j] + grad_output[k] * a[i] * coeff;
            }
        }
    }
}

// Complex specialization
template <typename T>
void legendre_polynomial_p_multiply_backward(
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

    // Compute gradients
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            const int64_t abs_diff = (i > j) ? (i - j) : (j - i);
            for (int64_t k = abs_diff; k <= i + j && k < output_size; k += 2) {
                const T coeff = legendre_linearization_coeff<T>(i, j, k);
                const c10::complex<T> coeff_c(coeff, T(0));
                grad_a[i] = grad_a[i] + grad_output[k] * b[j] * coeff_c;
                grad_b[j] = grad_b[j] + grad_output[k] * a[i] * coeff_c;
            }
        }
    }
}

} // namespace torchscience::kernel::polynomial
