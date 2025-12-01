#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

namespace torchscience::kernel::information_theory {

/**
 * Get dtype-dependent epsilon for numerical stability.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T get_eps() {
  return T(1e-7);  // Default for float32
}

template <>
C10_HOST_DEVICE C10_ALWAYS_INLINE double get_eps<double>() {
  return 1e-15;
}

template <>
C10_HOST_DEVICE C10_ALWAYS_INLINE c10::Half get_eps<c10::Half>() {
  return c10::Half(1e-4f);
}

template <>
C10_HOST_DEVICE C10_ALWAYS_INLINE c10::BFloat16 get_eps<c10::BFloat16>() {
  return c10::BFloat16(1e-3f);
}

/**
 * Compute KL divergence between two probability vectors.
 *
 * D_KL(P || Q) = sum_i p_i * log(p_i / q_i)
 *              = sum_i p_i * log(p_i) - p_i * log(q_i)
 *
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @return KL divergence value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T kl_divergence_kernel(
    const T* p,
    const T* q,
    int64_t n
) {
  T eps = get_eps<T>();
  T result = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    // Only contribute if p_i > 0 (by convention, 0 * log(0) = 0)
    if (p_i > eps) {
      result += p_i * (std::log(p_i) - std::log(q_i));
    }
  }

  return result;
}

/**
 * Compute JS divergence between two probability vectors.
 *
 * D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
 * where M = 0.5 * (P + Q)
 *
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param log_base_scale Scale factor for log base conversion (1.0 for natural log)
 * @return JS divergence value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T js_divergence_kernel(
    const T* p,
    const T* q,
    int64_t n,
    T log_base_scale
) {
  T eps = get_eps<T>();
  T result = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    T m_i = T(0.5) * (p_i + q_i);

    // D_KL(P || M) contribution
    if (p_i > eps) {
      result += T(0.5) * p_i * (std::log(p_i) - std::log(m_i));
    }

    // D_KL(Q || M) contribution
    if (q_i > eps) {
      result += T(0.5) * q_i * (std::log(q_i) - std::log(m_i));
    }
  }

  return result * log_base_scale;
}

}  // namespace torchscience::kernel::information_theory
