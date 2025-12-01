#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Apply quaternion rotation to a 3D point.
 * Uses optimized formula: v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
 * which is equivalent to: v' = v + w*t + (q_xyz x t) where t = 2*(q_xyz x v)
 *
 * @param q Quaternion [w, x, y, z], shape 4
 * @param point 3D point [px, py, pz], shape 3
 * @param output Rotated point, shape 3
 */
template <typename T>
void quaternion_apply_scalar(const T* q, const T* point, T* output) {
  const T w = q[0], x = q[1], y = q[2], z = q[3];
  const T px = point[0], py = point[1], pz = point[2];

  // t = 2 * (q_xyz x v)
  const T tx = T(2) * (y * pz - z * py);
  const T ty = T(2) * (z * px - x * pz);
  const T tz = T(2) * (x * py - y * px);

  // v' = v + w*t + (q_xyz x t)
  output[0] = px + w * tx + (y * tz - z * ty);
  output[1] = py + w * ty + (z * tx - x * tz);
  output[2] = pz + w * tz + (x * ty - y * tx);
}

}  // namespace torchscience::kernel::geometry::transform
