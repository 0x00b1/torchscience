#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Polynomial subtraction with zero-padding
// output[k] = p[k] - q[k], where missing coefficients are treated as 0
//
// Parameters:
//   output: result array, size max(N, M)
//   p: first polynomial, size N
//   q: second polynomial, size M
//   N, M: coefficient counts
template <typename T>
void polynomial_subtract(T* output, const T* p, const T* q, int64_t N, int64_t M) {
    const int64_t K = (N > M) ? N : M;
    for (int64_t k = 0; k < K; ++k) {
        T p_k = (k < N) ? p[k] : T(0);
        T q_k = (k < M) ? q[k] : T(0);
        output[k] = p_k - q_k;
    }
}

// Complex specialization
template <typename T>
void polynomial_subtract(
    c10::complex<T>* output,
    const c10::complex<T>* p,
    const c10::complex<T>* q,
    int64_t N,
    int64_t M
) {
    const int64_t K = (N > M) ? N : M;
    for (int64_t k = 0; k < K; ++k) {
        auto p_k = (k < N) ? p[k] : c10::complex<T>(T(0), T(0));
        auto q_k = (k < M) ? q[k] : c10::complex<T>(T(0), T(0));
        output[k] = p_k - q_k;
    }
}

} // namespace torchscience::kernel::polynomial
