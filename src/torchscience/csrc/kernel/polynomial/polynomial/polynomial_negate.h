#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::polynomial {

// Polynomial negation
// output[k] = -p[k]
//
// Parameters:
//   output: result array, size N
//   p: input polynomial, size N
//   N: coefficient count
template <typename T>
void polynomial_negate(T* output, const T* p, int64_t N) {
    for (int64_t k = 0; k < N; ++k) {
        output[k] = -p[k];
    }
}

// Complex specialization
template <typename T>
void polynomial_negate(
    c10::complex<T>* output,
    const c10::complex<T>* p,
    int64_t N
) {
    for (int64_t k = 0; k < N; ++k) {
        output[k] = -p[k];
    }
}

} // namespace torchscience::kernel::polynomial
