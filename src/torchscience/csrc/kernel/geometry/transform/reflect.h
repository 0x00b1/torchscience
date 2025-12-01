#pragma once

namespace torchscience::kernel::geometry::transform {

template <typename T>
void reflect_scalar(const T* direction, const T* normal, T* output) {
  const T dot = direction[0] * normal[0] +
                direction[1] * normal[1] +
                direction[2] * normal[2];
  const T two_dot = T(2) * dot;
  output[0] = direction[0] - two_dot * normal[0];
  output[1] = direction[1] - two_dot * normal[1];
  output[2] = direction[2] - two_dot * normal[2];
}

}  // namespace torchscience::kernel::geometry::transform
