#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Helper to get inv_sqrt2 = 1/sqrt(2) for any type (including Half/BFloat16)
template <typename T>
inline T inv_sqrt2() {
  return T(0.7071067811865475244);
}

// Standard normal CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
template <typename T>
T standard_normal_cumulative_distribution(T z) {
  return T(0.5) * (T(1) + std::erf(z * inv_sqrt2<T>()));
}

// Normal CDF with location and scale
template <typename T>
T normal_cumulative_distribution(T x, T loc, T scale) {
  T z = (x - loc) / scale;
  return standard_normal_cumulative_distribution(z);
}

}  // namespace torchscience::kernel::probability
