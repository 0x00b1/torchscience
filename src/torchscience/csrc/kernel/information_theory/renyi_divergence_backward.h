#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#include "kullback_leibler_divergence.h"  // For get_eps<T>

namespace torchscience::kernel::information_theory {

/**
 * Compute gradient of Renyi divergence w.r.t. input probabilities.
 *
 * D_alpha = 1/(alpha-1) * log(sum_i p_i^alpha * q_i^(1-alpha))
 *
 * Let S = sum_i p_i^alpha * q_i^(1-alpha)
 *
 * dD_alpha/dp_j = alpha / ((alpha-1) * S) * p_j^(alpha-1) * q_j^(1-alpha)
 * dD_alpha/dq_j = -1/S * p_j^alpha * q_j^(-alpha)
 *
 * @param grad_output Upstream gradient (scalar)
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param alpha Order of Renyi divergence
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void renyi_divergence_backward_kernel(
    T grad_output,
    const T* p,
    const T* q,
    int64_t n,
    T alpha,
    T log_base_scale,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();

  // Compute S = sum_i p_i^alpha * q_i^(1-alpha)
  T S = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    S += std::pow(p_i, alpha) * std::pow(q_i, T(1) - alpha);
  }

  T coeff_p = grad_output * log_base_scale * alpha / ((alpha - T(1)) * S);
  T coeff_q = -grad_output * log_base_scale / S;

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;

    // dD/dp_j = alpha / ((alpha-1) * S) * p_j^(alpha-1) * q_j^(1-alpha)
    grad_p[i] = coeff_p * std::pow(p_i, alpha - T(1)) * std::pow(q_i, T(1) - alpha);

    // dD/dq_j = -1/S * p_j^alpha * q_j^(-alpha)
    grad_q[i] = coeff_q * std::pow(p_i, alpha) * std::pow(q_i, -alpha);
  }
}

/**
 * Compute second-order backward pass for Renyi divergence.
 *
 * For D_alpha(P || Q) = 1/(alpha-1) * log(S) where S = sum_i p_i^alpha * q_i^(1-alpha)
 *
 * First-order gradients:
 *   g_p_i = grad_output * coeff_p / S * p_i^(alpha-1) * q_i^(1-alpha)
 *   g_q_i = grad_output * coeff_q / S * p_i^alpha * q_i^(-alpha)
 * where coeff_p = log_base_scale * alpha / (alpha-1), coeff_q = -log_base_scale
 *
 * Second-order involves derivatives of g_p and g_q w.r.t. p, q, and grad_output.
 *
 * @param gg_p Upstream gradient w.r.t. grad_p
 * @param gg_q Upstream gradient w.r.t. grad_q
 * @param grad_output Original upstream gradient
 * @param p Pointer to first probability distribution
 * @param q Pointer to second probability distribution
 * @param n Size of distributions
 * @param alpha Order of Renyi divergence
 * @param log_base_scale Scale factor for log base conversion
 * @param grad_grad_output Output: gradient w.r.t. grad_output
 * @param grad_p Output: gradient w.r.t. p
 * @param grad_q Output: gradient w.r.t. q
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void renyi_divergence_backward_backward_kernel(
    const T* gg_p,
    const T* gg_q,
    T grad_output,
    const T* p,
    const T* q,
    int64_t n,
    T alpha,
    T log_base_scale,
    T& grad_grad_output,
    T* grad_p,
    T* grad_q
) {
  T eps = get_eps<T>();

  // Compute S = sum_i p_i^alpha * q_i^(1-alpha)
  T S = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    S += std::pow(p_i, alpha) * std::pow(q_i, T(1) - alpha);
  }

  T coeff_p = log_base_scale * alpha / (alpha - T(1));
  T coeff_q = -log_base_scale;

  grad_grad_output = T(0);

  // First pass: compute grad_grad_output and accumulated terms for off-diagonal contributions
  // gg_p_dot_term = Σ gg_p_i · p_i^(α-1) · q_i^(1-α)
  // gg_q_dot_term = Σ gg_q_i · p_i^α · q_i^(-α)
  T gg_p_dot_term = T(0);
  T gg_q_dot_term = T(0);

  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    T gg_p_i = gg_p ? gg_p[i] : T(0);
    T gg_q_i = gg_q ? gg_q[i] : T(0);

    T p_alpha_m1 = std::pow(p_i, alpha - T(1));
    T q_1_m_alpha = std::pow(q_i, T(1) - alpha);
    T p_alpha = std::pow(p_i, alpha);
    T q_m_alpha = std::pow(q_i, -alpha);

    // grad_grad_output = Σ gg_p_i * (∂g_p_i/∂grad_output) + Σ gg_q_i * (∂g_q_i/∂grad_output)
    // ∂g_p_i/∂grad_output = coeff_p / S * p_i^(α-1) * q_i^(1-α)
    // ∂g_q_i/∂grad_output = coeff_q / S * p_i^α * q_i^(-α)
    grad_grad_output += gg_p_i * coeff_p / S * p_alpha_m1 * q_1_m_alpha;
    grad_grad_output += gg_q_i * coeff_q / S * p_alpha * q_m_alpha;

    gg_p_dot_term += gg_p_i * p_alpha_m1 * q_1_m_alpha;
    gg_q_dot_term += gg_q_i * p_alpha * q_m_alpha;
  }

  // Second pass: compute grad_p and grad_q
  for (int64_t i = 0; i < n; ++i) {
    T p_i = p[i] > eps ? p[i] : eps;
    T q_i = q[i] > eps ? q[i] : eps;
    T gg_p_i = gg_p ? gg_p[i] : T(0);
    T gg_q_i = gg_q ? gg_q[i] : T(0);

    T p_alpha_m1 = std::pow(p_i, alpha - T(1));
    T p_alpha_m2 = std::pow(p_i, alpha - T(2));
    T p_alpha = std::pow(p_i, alpha);
    T q_1_m_alpha = std::pow(q_i, T(1) - alpha);
    T q_m_alpha = std::pow(q_i, -alpha);
    T q_m_alpha_m1 = std::pow(q_i, -alpha - T(1));

    // ∂S/∂p_i = α · p_i^(α-1) · q_i^(1-α)
    T dS_dp = alpha * p_alpha_m1 * q_1_m_alpha;
    // ∂S/∂q_i = (1-α) · p_i^α · q_i^(-α)
    T dS_dq = (T(1) - alpha) * p_alpha * q_m_alpha;

    // grad_p[i] = Σ_j gg_p_j · ∂(g_p_j)/∂p_i + Σ_j gg_q_j · ∂(g_q_j)/∂p_i
    //
    // For g_p_j = grad_out * coeff_p / S * p_j^(α-1) * q_j^(1-α):
    //   ∂(g_p_j)/∂p_i = grad_out * coeff_p * [δ_ij*(α-1)*p_i^(α-2)*q_i^(1-α)/S - p_j^(α-1)*q_j^(1-α)*dS_dp/S²]
    //
    // For g_q_j = grad_out * coeff_q / S * p_j^α * q_j^(-α):
    //   ∂(g_q_j)/∂p_i = grad_out * coeff_q * [δ_ij*α*p_i^(α-1)*q_i^(-α)/S - p_j^α*q_j^(-α)*dS_dp/S²]

    // Diagonal terms (j == i)
    T diag_p_from_gp = gg_p_i * coeff_p / S * (alpha - T(1)) * p_alpha_m2 * q_1_m_alpha;
    T diag_p_from_gq = gg_q_i * coeff_q / S * alpha * p_alpha_m1 * q_m_alpha;

    // Off-diagonal contribution from all j (including j == i, which is correct for chain rule)
    T off_diag_p = -(gg_p_dot_term * coeff_p + gg_q_dot_term * coeff_q) / (S * S) * dS_dp;

    grad_p[i] = grad_output * (diag_p_from_gp + diag_p_from_gq + off_diag_p);

    // Similarly for grad_q[i]
    // For g_p_j:
    //   ∂(g_p_j)/∂q_i = grad_out * coeff_p * [δ_ij*(1-α)*p_i^(α-1)*q_i^(-α)/S - p_j^(α-1)*q_j^(1-α)*dS_dq/S²]
    //
    // For g_q_j:
    //   ∂(g_q_j)/∂q_i = grad_out * coeff_q * [δ_ij*(-α)*p_i^α*q_i^(-α-1)/S - p_j^α*q_j^(-α)*dS_dq/S²]

    T diag_q_from_gp = gg_p_i * coeff_p / S * (T(1) - alpha) * p_alpha_m1 * q_m_alpha;
    T diag_q_from_gq = gg_q_i * coeff_q / S * (-alpha) * p_alpha * q_m_alpha_m1;
    T off_diag_q = -(gg_p_dot_term * coeff_p + gg_q_dot_term * coeff_q) / (S * S) * dS_dq;

    grad_q[i] = grad_output * (diag_q_from_gp + diag_q_from_gq + off_diag_q);
  }
}

}  // namespace torchscience::kernel::information_theory
