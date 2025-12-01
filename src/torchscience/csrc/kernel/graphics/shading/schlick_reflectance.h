#pragma once

namespace torchscience::kernel::graphics::shading {

/**
 * Schlick approximation for Fresnel reflectance.
 *
 * Mathematical Definition:
 *   reflectance = r0 + (1 - r0) * (1 - cosine)^5
 *
 * where r0 = ((1 - ior) / (1 + ior))^2 is the reflectance at normal incidence.
 *
 * Note: r0 is precomputed and passed as an argument for efficiency.
 */
template <typename T>
T schlick_reflectance_scalar(T cosine, T r0) {
  const T one_minus_cosine = T(1) - cosine;
  const T one_minus_cosine_sq = one_minus_cosine * one_minus_cosine;
  const T one_minus_cosine_pow5 = one_minus_cosine_sq * one_minus_cosine_sq * one_minus_cosine;
  return r0 + (T(1) - r0) * one_minus_cosine_pow5;
}

}  // namespace torchscience::kernel::graphics::shading
