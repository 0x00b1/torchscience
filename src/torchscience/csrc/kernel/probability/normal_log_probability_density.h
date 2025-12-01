#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Normal log probability density function
// logpdf(x; loc, scale) = -0.5 * log(2*pi) - log(scale) - 0.5 * z^2
// where z = (x - loc) / scale
template <typename T>
inline T half_log_2pi() {
  return T(0.9189385332046727);  // 0.5 * log(2 * pi)
}

template <typename T>
T normal_log_probability_density(T x, T loc, T scale) {
  T z = (x - loc) / scale;
  return -half_log_2pi<T>() - std::log(scale) - T(0.5) * z * z;
}

}  // namespace torchscience::kernel::probability
