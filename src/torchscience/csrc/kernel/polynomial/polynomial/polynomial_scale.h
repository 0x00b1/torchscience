#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Polynomial scaling by scalar
// output[k] = c * p[k] for all k
//
// Parameters:
//   output: result array, size N
//   p: input polynomial, size N
//   c: scalar value
//   N: coefficient count
template <typename T>
void polynomial_scale(T* output, const T* p, T c, int64_t N) {
    for (int64_t k = 0; k < N; ++k) {
        output[k] = c * p[k];
    }
}

// Complex specialization
template <typename T>
void polynomial_scale(
    c10::complex<T>* output,
    const c10::complex<T>* p,
    c10::complex<T> c,
    int64_t N
) {
    for (int64_t k = 0; k < N; ++k) {
        output[k] = c * p[k];
    }
}

} // namespace torchscience::kernel::polynomial
