#pragma once

#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

// Rational polynomial evaluation (Cephes-style)
// Evaluates: coef[0] + coef[1]*x + coef[2]*x^2 + ... + coef[n]*x^n
template <typename T>
T polevl(T x, const double* coef, int n) {
    T result = T(coef[0]);
    for (int i = 1; i <= n; ++i) {
        result = result * x + T(coef[i]);
    }
    return result;
}

// Rational polynomial evaluation with implicit leading coefficient 1
// Evaluates: x + coef[0] + coef[1]*x^{-1} + ...
// More precisely: (x + coef[0]) * x^{n-1} + coef[1] * x^{n-2} + ... + coef[n-1]
// Which equals: x^n + coef[0]*x^{n-1} + coef[1]*x^{n-2} + ... + coef[n-1]
template <typename T>
T p1evl(T x, const double* coef, int n) {
    T result = x + T(coef[0]);
    for (int i = 1; i < n; ++i) {
        result = result * x + T(coef[i]);
    }
    return result;
}

// Complex versions
template <typename T>
c10::complex<T> polevl(c10::complex<T> x, const double* coef, int n) {
    c10::complex<T> result(T(coef[0]), T(0));
    for (int i = 1; i <= n; ++i) {
        result = result * x + c10::complex<T>(T(coef[i]), T(0));
    }
    return result;
}

template <typename T>
c10::complex<T> p1evl(c10::complex<T> x, const double* coef, int n) {
    c10::complex<T> result = x + c10::complex<T>(T(coef[0]), T(0));
    for (int i = 1; i < n; ++i) {
        result = result * x + c10::complex<T>(T(coef[i]), T(0));
    }
    return result;
}

} // namespace detail

} // namespace torchscience::kernel::special_functions
