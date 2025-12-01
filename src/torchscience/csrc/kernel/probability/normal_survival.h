#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Normal survival function (1 - CDF) using erfc for numerical stability
// SF(x; loc, scale) = 0.5 * erfc((x - loc) / (scale * sqrt(2)))
template <typename T>
T normal_survival(T x, T loc, T scale) {
  const T inv_sqrt2 = T(0.7071067811865476);  // 1 / sqrt(2)
  T z = (x - loc) / scale;
  return T(0.5) * std::erfc(z * inv_sqrt2);
}

}  // namespace torchscience::kernel::probability
