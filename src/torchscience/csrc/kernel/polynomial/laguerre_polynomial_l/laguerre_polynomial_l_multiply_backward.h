#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace torchscience::kernel::polynomial {

// Forward declaration of helpers from multiply.h
template <typename T>
T laguerre_binomial(int64_t n, int64_t k);

template <typename T>
T laguerre_factorial(int64_t n);

// Backward for Laguerre polynomial multiplication
//
// The multiply is bilinear: c = f(a, b)
// grad_a = d_loss/d_a = sum_k grad_output[k] * d_c[k]/d_a
// grad_b = d_loss/d_b = sum_k grad_output[k] * d_c[k]/d_b
//
// Since c = multiply(a, b), we have:
// d_c/d_a[i] = multiply(e_i, b) where e_i is unit vector
// d_c/d_b[j] = multiply(a, e_j)
//
// So grad_a[i] = sum_k grad_output[k] * [multiply(e_i, b)]_k
//              = [multiply(grad_output, b)]_i (by linearity, reversed)
//
// But we need to be careful about dimensions. Let's use the chain rule directly.
//
// For efficiency, we compute:
// grad_a = multiply_backward_a(grad_output, b)
// grad_b = multiply_backward_b(grad_output, a)
//
// These use the adjoint of the linearization.
//
// Parameters:
//   grad_a: output gradient w.r.t. a, size N
//   grad_b: output gradient w.r.t. b, size M
//   grad_output: incoming gradient, size output_size
//   a: first input coefficients, size N
//   b: second input coefficients, size M
//   N: size of a
//   M: size of b
//   output_size: size of grad_output
template <typename T>
void laguerre_polynomial_l_multiply_backward(
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

    // The forward pass does:
    // 1. a -> pa (Laguerre to power)
    // 2. b -> pb (Laguerre to power)
    // 3. pa, pb -> pc (convolution)
    // 4. pc -> c (power to Laguerre)
    //
    // For backward, we reverse:
    // 1. grad_c -> grad_pc (adjoint of power to Laguerre)
    // 2. grad_pc -> grad_pa, grad_pb (adjoint of convolution)
    // 3. grad_pa -> grad_a, grad_pb -> grad_b (adjoint of Laguerre to power)

    // Step 1: Convert a and b to power series (same as forward)
    std::vector<T> pa(N, T(0));
    for (int64_t k = 0; k < N; ++k) {
        T sign_k = (k % 2 == 0) ? T(1) : T(-1);
        T fact_k = laguerre_factorial<T>(k);
        for (int64_t j = k; j < N; ++j) {
            pa[k] = pa[k] + a[j] * sign_k * laguerre_binomial<T>(j, k) / fact_k;
        }
    }

    std::vector<T> pb(M, T(0));
    for (int64_t k = 0; k < M; ++k) {
        T sign_k = (k % 2 == 0) ? T(1) : T(-1);
        T fact_k = laguerre_factorial<T>(k);
        for (int64_t j = k; j < M; ++j) {
            pb[k] = pb[k] + b[j] * sign_k * laguerre_binomial<T>(j, k) / fact_k;
        }
    }

    // Step 2: Adjoint of power to Laguerre (grad_output -> grad_pc)
    // Forward: c[k] = sum_{j>=k} pc[j] * j! * (-1)^{j-k} * C(j,k)
    // Adjoint: grad_pc[j] = sum_{k<=j} grad_c[k] * j! * (-1)^{j-k} * C(j,k)
    std::vector<T> grad_pc(output_size, T(0));
    for (int64_t j = 0; j < output_size; ++j) {
        T fact_j = laguerre_factorial<T>(j);
        for (int64_t k = 0; k <= j; ++k) {
            T sign_jk = ((j - k) % 2 == 0) ? T(1) : T(-1);
            grad_pc[j] = grad_pc[j] + grad_output[k] * fact_j * sign_jk * laguerre_binomial<T>(j, k);
        }
    }

    // Step 3: Adjoint of convolution (grad_pc -> grad_pa, grad_pb)
    // Forward: pc[i+j] += pa[i] * pb[j]
    // Adjoint: grad_pa[i] = sum_j grad_pc[i+j] * pb[j]
    //          grad_pb[j] = sum_i grad_pc[i+j] * pa[i]
    std::vector<T> grad_pa(N, T(0));
    std::vector<T> grad_pb(M, T(0));
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            if (i + j < output_size) {
                grad_pa[i] = grad_pa[i] + grad_pc[i + j] * pb[j];
                grad_pb[j] = grad_pb[j] + grad_pc[i + j] * pa[i];
            }
        }
    }

    // Step 4: Adjoint of Laguerre to power (grad_pa -> grad_a, grad_pb -> grad_b)
    // Forward: pa[k] = sum_{j>=k} a[j] * (-1)^k * C(j,k) / k!
    // Adjoint: grad_a[j] = sum_{k<=j} grad_pa[k] * (-1)^k * C(j,k) / k!
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = 0; k <= j; ++k) {
            T sign_k = (k % 2 == 0) ? T(1) : T(-1);
            T fact_k = laguerre_factorial<T>(k);
            grad_a[j] = grad_a[j] + grad_pa[k] * sign_k * laguerre_binomial<T>(j, k) / fact_k;
        }
    }

    for (int64_t j = 0; j < M; ++j) {
        for (int64_t k = 0; k <= j; ++k) {
            T sign_k = (k % 2 == 0) ? T(1) : T(-1);
            T fact_k = laguerre_factorial<T>(k);
            grad_b[j] = grad_b[j] + grad_pb[k] * sign_k * laguerre_binomial<T>(j, k) / fact_k;
        }
    }
}

// Complex specialization
template <typename T>
void laguerre_polynomial_l_multiply_backward(
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

    for (int64_t i = 0; i < N; ++i) {
        grad_a[i] = zero;
    }
    for (int64_t j = 0; j < M; ++j) {
        grad_b[j] = zero;
    }

    if (N == 0 || M == 0) {
        return;
    }

    // Step 1: Convert a and b to power series
    std::vector<c10::complex<T>> pa(N, zero);
    for (int64_t k = 0; k < N; ++k) {
        T sign_k = (k % 2 == 0) ? T(1) : T(-1);
        T fact_k = laguerre_factorial<T>(k);
        c10::complex<T> coeff(sign_k / fact_k, T(0));
        for (int64_t j = k; j < N; ++j) {
            T binom = laguerre_binomial<T>(j, k);
            pa[k] = pa[k] + a[j] * c10::complex<T>(binom, T(0)) * coeff;
        }
    }

    std::vector<c10::complex<T>> pb(M, zero);
    for (int64_t k = 0; k < M; ++k) {
        T sign_k = (k % 2 == 0) ? T(1) : T(-1);
        T fact_k = laguerre_factorial<T>(k);
        c10::complex<T> coeff(sign_k / fact_k, T(0));
        for (int64_t j = k; j < M; ++j) {
            T binom = laguerre_binomial<T>(j, k);
            pb[k] = pb[k] + b[j] * c10::complex<T>(binom, T(0)) * coeff;
        }
    }

    // Step 2: Adjoint of power to Laguerre
    std::vector<c10::complex<T>> grad_pc(output_size, zero);
    for (int64_t j = 0; j < output_size; ++j) {
        T fact_j = laguerre_factorial<T>(j);
        for (int64_t k = 0; k <= j; ++k) {
            T sign_jk = ((j - k) % 2 == 0) ? T(1) : T(-1);
            T binom = laguerre_binomial<T>(j, k);
            grad_pc[j] = grad_pc[j] + grad_output[k] * c10::complex<T>(fact_j * sign_jk * binom, T(0));
        }
    }

    // Step 3: Adjoint of convolution
    std::vector<c10::complex<T>> grad_pa(N, zero);
    std::vector<c10::complex<T>> grad_pb(M, zero);
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            if (i + j < output_size) {
                grad_pa[i] = grad_pa[i] + grad_pc[i + j] * pb[j];
                grad_pb[j] = grad_pb[j] + grad_pc[i + j] * pa[i];
            }
        }
    }

    // Step 4: Adjoint of Laguerre to power
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = 0; k <= j; ++k) {
            T sign_k = (k % 2 == 0) ? T(1) : T(-1);
            T fact_k = laguerre_factorial<T>(k);
            T binom = laguerre_binomial<T>(j, k);
            grad_a[j] = grad_a[j] + grad_pa[k] * c10::complex<T>(sign_k * binom / fact_k, T(0));
        }
    }

    for (int64_t j = 0; j < M; ++j) {
        for (int64_t k = 0; k <= j; ++k) {
            T sign_k = (k % 2 == 0) ? T(1) : T(-1);
            T fact_k = laguerre_factorial<T>(k);
            T binom = laguerre_binomial<T>(j, k);
            grad_b[j] = grad_b[j] + grad_pb[k] * c10::complex<T>(sign_k * binom / fact_k, T(0));
        }
    }
}

} // namespace torchscience::kernel::polynomial
