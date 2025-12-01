#pragma once

#include <c10/util/complex.h>

#include "log_gamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T log_beta(T a, T b) {
  return log_gamma(a) + log_gamma(b) - log_gamma(a + b);
}

template <typename T>
c10::complex<T> log_beta(c10::complex<T> a, c10::complex<T> b) {
  return log_gamma(a) + log_gamma(b) - log_gamma(a + b);
}

} // namespace torchscience::kernel::special_functions
