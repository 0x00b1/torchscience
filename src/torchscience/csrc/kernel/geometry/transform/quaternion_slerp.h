#pragma once

#include <cmath>

namespace torchscience::kernel::geometry::transform {

/**
 * Spherical linear interpolation between two quaternions.
 *
 * slerp(q1, q2, t) smoothly interpolates from q1 (t=0) to q2 (t=1)
 * along the shortest arc on the 4D unit sphere.
 *
 * @param q1 First quaternion [w, x, y, z], shape 4
 * @param q2 Second quaternion [w, x, y, z], shape 4
 * @param t Interpolation parameter (0 to 1)
 * @param output Interpolated quaternion, shape 4
 */
template <typename T>
void quaternion_slerp_scalar(const T* q1, const T* q2, T t, T* output) {
  // Compute dot product (cosine of angle)
  T dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];

  // If dot < 0, negate q2 to take the shorter path
  T q2_adj[4] = {q2[0], q2[1], q2[2], q2[3]};
  if (dot < T(0)) {
    dot = -dot;
    q2_adj[0] = -q2[0];
    q2_adj[1] = -q2[1];
    q2_adj[2] = -q2[2];
    q2_adj[3] = -q2[3];
  }

  // Clamp dot to [-1, 1] for numerical stability
  if (dot > T(1))
    dot = T(1);

  T s1, s2;

  // If quaternions are very close, use linear interpolation to avoid division
  // by small sin
  if (dot > T(0.9995)) {
    s1 = T(1) - t;
    s2 = t;
  } else {
    const T theta = std::acos(dot);
    const T sin_theta = std::sin(theta);
    s1 = std::sin((T(1) - t) * theta) / sin_theta;
    s2 = std::sin(t * theta) / sin_theta;
  }

  output[0] = s1 * q1[0] + s2 * q2_adj[0];
  output[1] = s1 * q1[1] + s2 * q2_adj[1];
  output[2] = s1 * q1[2] + s2 * q2_adj[2];
  output[3] = s1 * q1[3] + s2 * q2_adj[3];

  // Normalize for numerical stability (in case q1/q2 aren't perfectly unit)
  const T norm = std::sqrt(output[0] * output[0] + output[1] * output[1] +
                           output[2] * output[2] + output[3] * output[3]);
  if (norm > T(0)) {
    const T inv_norm = T(1) / norm;
    output[0] *= inv_norm;
    output[1] *= inv_norm;
    output[2] *= inv_norm;
    output[3] *= inv_norm;
  }
}

}  // namespace torchscience::kernel::geometry::transform
