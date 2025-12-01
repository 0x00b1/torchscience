#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Scalar kernel for Snell's law refraction.
 *
 * Computes: T = eta * D + (eta * cos_i - cos_t) * N
 * where:
 *   - D is incident direction (normalized, toward surface)
 *   - N is surface normal (normalized)
 *   - eta = n1/n2 (ratio of refractive indices)
 *   - cos_i = -D.N
 *   - cos_t = sqrt(1 - eta^2 * sin^2_i)
 *
 * Returns zero vector for total internal reflection (when sin^2_t > 1).
 */
template <typename T>
void refract_scalar(const T* direction, const T* normal, T eta, T* output) {
  // Compute cos_i = -D.N
  const T cos_i = -(direction[0] * normal[0] +
                    direction[1] * normal[1] +
                    direction[2] * normal[2]);

  // Compute sin^2_t = eta^2 * (1 - cos^2_i) = eta^2 * sin^2_i
  const T sin2_i = T(1) - cos_i * cos_i;
  const T sin2_t = eta * eta * sin2_i;

  // Check for total internal reflection
  if (sin2_t > T(1)) {
    output[0] = T(0);
    output[1] = T(0);
    output[2] = T(0);
    return;
  }

  // Compute cos_t = sqrt(1 - sin^2_t)
  const T cos_t = std::sqrt(T(1) - sin2_t);

  // Compute refracted direction: T = eta * D + (eta * cos_i - cos_t) * N
  const T k = eta * cos_i - cos_t;
  output[0] = eta * direction[0] + k * normal[0];
  output[1] = eta * direction[1] + k * normal[1];
  output[2] = eta * direction[2] + k * normal[2];
}

}  // namespace torchscience::kernel::geometry::transform
