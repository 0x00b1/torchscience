#pragma once

#include <cmath>
#include "../special_functions/log_gamma.h"

namespace torchscience::kernel::special_functions {

namespace {

// Compute sign of gamma function for a real argument
// Gamma(x) < 0 when floor(x) is an odd negative integer
template <typename T>
int gamma_sign(T x) {
  if (x > T(0)) {
    return 1;
  }
  // For x <= 0, check if floor(x) is odd
  T fl = std::floor(x);
  int64_t fl_int = static_cast<int64_t>(fl);
  return (fl_int % 2 == 0) ? 1 : -1;
}

} // anonymous namespace

template <typename T>
T binomial_coefficient(T n, T k) {
  // C(n, k) = 0 for k < 0
  if (k < T(0)) {
    return T(0);
  }

  // C(n, k) = 0 for k > n when n is a non-negative integer
  // For non-integer n, the gamma function handles this correctly
  if (n >= T(0) && k > n && std::floor(n) == n) {
    return T(0);
  }

  // C(n, 0) = 1 for all n
  if (k == T(0)) {
    return T(1);
  }

  // Use log-gamma for numerical stability:
  // C(n, k) = Gamma(n+1) / (Gamma(k+1) * Gamma(n-k+1))
  // lgamma gives log of absolute value, so we compute sign separately
  T a = n + T(1);
  T b = k + T(1);
  T c = n - k + T(1);

  T log_abs_result = special_functions::log_gamma(a)
                   - special_functions::log_gamma(b)
                   - special_functions::log_gamma(c);

  // Compute combined sign: sign(Gamma(a)) / (sign(Gamma(b)) * sign(Gamma(c)))
  int sign = gamma_sign(a) * gamma_sign(b) * gamma_sign(c);

  return static_cast<T>(sign) * std::exp(log_abs_result);
}

} // namespace torchscience::kernel::special_functions
