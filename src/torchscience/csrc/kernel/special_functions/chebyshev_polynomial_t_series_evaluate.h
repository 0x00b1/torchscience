#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

// Evaluate a Chebyshev series using Cephes chbevl algorithm.
//
// This matches the Cephes library's chbevl function exactly.
// Evaluates: 0.5 * coef[0] + sum_{k=1}^{n-1} coef[k] * T_k(x)
// (where T_k is the k-th Chebyshev polynomial of the first kind)
//
// The algorithm uses the recurrence:
//   b0 = coef[0], b1 = 0
//   for k = 1 to n-1:
//     b2 = b1; b1 = b0; b0 = x*b1 - b2 + coef[k]
//   result = 0.5 * (b0 - b2)
template <typename T>
T chebyshev_polynomial_t_series_evaluate(T x, const double* coef, int n) {
    if (n == 0) return T(0);
    if (n == 1) return T(0.5) * T(coef[0]);

    T b0 = T(coef[0]);
    T b1 = T(0);
    T b2 = T(0);

    for (int k = 1; k < n; ++k) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + T(coef[k]);
    }

    return T(0.5) * (b0 - b2);
}

// Complex version
template <typename T>
c10::complex<T> chebyshev_polynomial_t_series_evaluate(
    c10::complex<T> x, const double* coef, int n) {
    if (n == 0) return c10::complex<T>(T(0), T(0));
    if (n == 1) return c10::complex<T>(T(0.5) * T(coef[0]), T(0));

    c10::complex<T> b0(T(coef[0]), T(0));
    c10::complex<T> b1(T(0), T(0));
    c10::complex<T> b2(T(0), T(0));

    for (int k = 1; k < n; ++k) {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + c10::complex<T>(T(coef[k]), T(0));
    }

    return c10::complex<T>(T(0.5), T(0)) * (b0 - b2);
}

} // namespace torchscience::kernel::special_functions
