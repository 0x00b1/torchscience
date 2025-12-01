#pragma once

#include <cmath>

namespace torchscience::kernel::geometry::transform {

/**
 * Normalize a quaternion to unit length.
 * output = q / ||q||
 *
 * Uses scalar-first (wxyz) convention.
 */
template <typename T>
void quaternion_normalize_scalar(const T* q, T* output) {
  const T norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  const T inv_norm = T(1) / norm;
  output[0] = q[0] * inv_norm;
  output[1] = q[1] * inv_norm;
  output[2] = q[2] * inv_norm;
  output[3] = q[3] * inv_norm;
}

}  // namespace torchscience::kernel::geometry::transform
