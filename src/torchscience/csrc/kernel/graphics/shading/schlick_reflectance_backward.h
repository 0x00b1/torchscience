#pragma once

namespace torchscience::kernel::graphics::shading {

/**
 * Backward pass for Schlick approximation.
 *
 * Mathematical Definition:
 *   reflectance = r0 + (1 - r0) * (1 - cosine)^5
 *
 * Gradient w.r.t. cosine:
 *   d(reflectance)/d(cosine) = -5 * (1 - r0) * (1 - cosine)^4
 */
template <typename T>
T schlick_reflectance_backward_scalar(T grad_output, T cosine, T r0) {
  const T one_minus_cosine = T(1) - cosine;
  const T one_minus_cosine_sq = one_minus_cosine * one_minus_cosine;
  const T one_minus_cosine_pow4 = one_minus_cosine_sq * one_minus_cosine_sq;
  const T grad_cosine = T(-5) * (T(1) - r0) * one_minus_cosine_pow4;
  return grad_output * grad_cosine;
}

}  // namespace torchscience::kernel::graphics::shading
