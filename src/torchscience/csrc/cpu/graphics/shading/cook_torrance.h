#pragma once

#include <algorithm>
#include <cmath>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::graphics::shading {

namespace {

// ============================================================================
// Cook-Torrance BRDF Helper Functions
// ============================================================================

// Minimum roughness to avoid division by zero
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T min_roughness() { return T(0.001); }

// Small epsilon for dot product clamping
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T dot_epsilon() { return T(1e-7); }

/**
 * Compute dot product of two 3D vectors.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T dot3(const T* a, const T* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Compute GGX/Trowbridge-Reitz normal distribution function.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T ggx_distribution(T n_dot_h, T alpha_squared) {
    T n_dot_h_sq = n_dot_h * n_dot_h;
    T denom = n_dot_h_sq * (alpha_squared - T(1)) + T(1);
    denom = denom * denom;
    const T PI = T(3.14159265358979323846);
    return alpha_squared / (PI * denom);
}

/**
 * Compute Schlick-GGX geometry sub-term.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T schlick_ggx_g1(T n_dot_x, T k) {
    return n_dot_x / (n_dot_x * (T(1) - k) + k);
}

/**
 * Compute Schlick-GGX geometry term (Smith masking-shadowing).
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T schlick_ggx_geometry(T n_dot_v, T n_dot_l, T roughness) {
    T r_plus_1 = roughness + T(1);
    T k = (r_plus_1 * r_plus_1) / T(8);
    return schlick_ggx_g1(n_dot_v, k) * schlick_ggx_g1(n_dot_l, k);
}

/**
 * Compute Schlick Fresnel approximation.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T schlick_fresnel(T h_dot_v, T f0) {
    T one_minus_cos = T(1) - h_dot_v;
    T pow5 = one_minus_cos * one_minus_cos;
    pow5 = pow5 * pow5 * one_minus_cos;
    return f0 + (T(1) - f0) * pow5;
}

/**
 * Compute Cook-Torrance specular BRDF for a single sample.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T cook_torrance_scalar(
    const T* normal,
    const T* view,
    const T* light,
    T roughness,
    T f0
) {
    // Clamp roughness to [0.001, 1.0] to avoid singularities
    roughness = std::clamp(roughness, min_roughness<T>(), T(1));

    // Compute dot products
    T n_dot_l = dot3(normal, light);
    T n_dot_v = dot3(normal, view);

    // Early out for back-facing geometry
    if (n_dot_l <= T(0) || n_dot_v <= T(0)) {
        return T(0);
    }

    n_dot_l = std::max(n_dot_l, dot_epsilon<T>());
    n_dot_v = std::max(n_dot_v, dot_epsilon<T>());

    // Compute halfway vector: h = normalize(l + v)
    T h[3] = { light[0] + view[0], light[1] + view[1], light[2] + view[2] };
    T h_len = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);

    if (h_len < dot_epsilon<T>()) {
        return T(0);
    }

    T inv_h_len = T(1) / h_len;
    T h_normalized[3] = { h[0] * inv_h_len, h[1] * inv_h_len, h[2] * inv_h_len };

    T n_dot_h = std::max(dot3(normal, h_normalized), dot_epsilon<T>());
    T h_dot_v = std::max(dot3(h_normalized, view), dot_epsilon<T>());

    // Compute BRDF components
    T alpha = roughness * roughness;
    T alpha_squared = alpha * alpha;

    T D = ggx_distribution(n_dot_h, alpha_squared);
    T G = schlick_ggx_geometry(n_dot_v, n_dot_l, roughness);
    T F = schlick_fresnel(h_dot_v, f0);

    // Cook-Torrance specular BRDF
    T denominator = T(4) * n_dot_l * n_dot_v;
    return (D * G * F) / denominator;
}

/**
 * Compute gradients for Cook-Torrance BRDF.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void cook_torrance_backward_scalar(
    T grad_out,
    const T* normal,
    const T* view,
    const T* light,
    T roughness,
    T f0,
    T* grad_normal,
    T* grad_view,
    T* grad_light,
    T* grad_roughness,
    T* grad_f0
) {
    // Initialize gradients to zero
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = T(0);
        grad_view[i] = T(0);
        grad_light[i] = T(0);
    }
    *grad_roughness = T(0);
    *grad_f0 = T(0);

    roughness = std::clamp(roughness, min_roughness<T>(), T(1));

    T n_dot_l = dot3(normal, light);
    T n_dot_v = dot3(normal, view);

    if (n_dot_l <= T(0) || n_dot_v <= T(0)) {
        return;
    }

    T n_dot_l_clamped = std::max(n_dot_l, dot_epsilon<T>());
    T n_dot_v_clamped = std::max(n_dot_v, dot_epsilon<T>());

    T h[3] = { light[0] + view[0], light[1] + view[1], light[2] + view[2] };
    T h_len = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);

    if (h_len < dot_epsilon<T>()) {
        return;
    }

    T inv_h_len = T(1) / h_len;
    T h_normalized[3] = { h[0] * inv_h_len, h[1] * inv_h_len, h[2] * inv_h_len };

    T n_dot_h = std::max(dot3(normal, h_normalized), dot_epsilon<T>());
    T h_dot_v = std::max(dot3(h_normalized, view), dot_epsilon<T>());

    T alpha = roughness * roughness;
    T alpha_squared = alpha * alpha;

    // D term
    T n_dot_h_sq = n_dot_h * n_dot_h;
    T denom_d = n_dot_h_sq * (alpha_squared - T(1)) + T(1);
    T denom_d_sq = denom_d * denom_d;
    const T PI = T(3.14159265358979323846);
    T D = alpha_squared / (PI * denom_d_sq);

    // G term
    T r_plus_1 = roughness + T(1);
    T k = (r_plus_1 * r_plus_1) / T(8);
    T g1_v_denom = n_dot_v_clamped * (T(1) - k) + k;
    T g1_l_denom = n_dot_l_clamped * (T(1) - k) + k;
    T G1_v = n_dot_v_clamped / g1_v_denom;
    T G1_l = n_dot_l_clamped / g1_l_denom;
    T G = G1_v * G1_l;

    // F term
    T one_minus_hdv = T(1) - h_dot_v;
    T pow4 = one_minus_hdv * one_minus_hdv;
    pow4 = pow4 * pow4;
    T pow5 = pow4 * one_minus_hdv;
    T F = f0 + (T(1) - f0) * pow5;

    T denom = T(4) * n_dot_l_clamped * n_dot_v_clamped;
    T brdf = (D * G * F) / denom;

    // Gradient w.r.t. F₀
    T dF_df0 = T(1) - pow5;
    T d_brdf_dF = D * G / denom;
    *grad_f0 = grad_out * d_brdf_dF * dF_df0;

    // Gradient w.r.t. roughness
    T d_denom_d_dalpha_sq = n_dot_h_sq;
    T dD_dalpha_sq = (denom_d - T(2) * alpha_squared * d_denom_d_dalpha_sq) / (PI * denom_d_sq * denom_d);
    T dalpha_sq_droughness = T(4) * alpha * roughness;

    T dk_droughness = (roughness + T(1)) / T(4);
    T dG1_v_dk = n_dot_v_clamped * (n_dot_v_clamped - T(1)) / (g1_v_denom * g1_v_denom);
    T dG1_l_dk = n_dot_l_clamped * (n_dot_l_clamped - T(1)) / (g1_l_denom * g1_l_denom);
    T dG_dk = dG1_v_dk * G1_l + G1_v * dG1_l_dk;
    T dG_droughness = dG_dk * dk_droughness;

    T d_brdf_dD = G * F / denom;
    T d_brdf_dG = D * F / denom;

    *grad_roughness = grad_out * (d_brdf_dD * dD_dalpha_sq * dalpha_sq_droughness + d_brdf_dG * dG_droughness);

    // Vector gradients
    T dD_dndoth = -T(4) * alpha_squared * n_dot_h * (alpha_squared - T(1)) / (PI * denom_d_sq * denom_d);
    T d_brdf_dndoth = grad_out * d_brdf_dD * dD_dndoth;

    T dF_dhdotv = -T(5) * (T(1) - f0) * pow4;
    T d_brdf_dhdotv = grad_out * d_brdf_dF * dF_dhdotv;

    T dG1_v_dndotv = k / (g1_v_denom * g1_v_denom);
    T dG_dndotv = dG1_v_dndotv * G1_l;
    T d_brdf_dndotv = grad_out * (d_brdf_dG * dG_dndotv - brdf / n_dot_v_clamped);

    T dG1_l_dndotl = k / (g1_l_denom * g1_l_denom);
    T dG_dndotl = G1_v * dG1_l_dndotl;
    T d_brdf_dndotl = grad_out * (d_brdf_dG * dG_dndotl - brdf / n_dot_l_clamped);

    // Propagate to vectors
    for (int i = 0; i < 3; ++i) {
        grad_normal[i] = d_brdf_dndotl * light[i] + d_brdf_dndotv * view[i] + d_brdf_dndoth * h_normalized[i];
    }

    T dndoth_dv[3], dndoth_dl[3];
    for (int i = 0; i < 3; ++i) {
        dndoth_dv[i] = (normal[i] - n_dot_h * h_normalized[i]) * inv_h_len;
        dndoth_dl[i] = dndoth_dv[i];
    }

    T dhdotv_dv[3], dhdotv_dl[3];
    for (int i = 0; i < 3; ++i) {
        dhdotv_dv[i] = h_normalized[i] + (view[i] - h_dot_v * h_normalized[i]) * inv_h_len;
        dhdotv_dl[i] = (view[i] - h_dot_v * h_normalized[i]) * inv_h_len;
    }

    for (int i = 0; i < 3; ++i) {
        grad_view[i] = d_brdf_dndotv * normal[i] + d_brdf_dndoth * dndoth_dv[i] + d_brdf_dhdotv * dhdotv_dv[i];
        grad_light[i] = d_brdf_dndotl * normal[i] + d_brdf_dndoth * dndoth_dl[i] + d_brdf_dhdotv * dhdotv_dl[i];
    }
}

/**
 * Compute second-order gradients for Cook-Torrance BRDF.
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void cook_torrance_backward_backward_scalar(
    // Gradients of the first backward outputs (gg = grad of grad)
    T gg_normal_x, T gg_normal_y, T gg_normal_z,
    T gg_view_x, T gg_view_y, T gg_view_z,
    T gg_light_x, T gg_light_y, T gg_light_z,
    T gg_roughness,
    T gg_f0,
    // Original forward inputs
    T grad_out,
    const T* normal,
    const T* view,
    const T* light,
    T roughness,
    T f0,
    // Outputs: gradients w.r.t. backward inputs
    T* grad_grad_out,        // scalar
    T* grad2_normal,         // [3]
    T* grad2_view,           // [3]
    T* grad2_light,          // [3]
    T* grad2_roughness,      // scalar
    T* grad2_f0              // scalar
) {
    // Initialize all outputs to zero
    *grad_grad_out = T(0);
    for (int i = 0; i < 3; ++i) {
        grad2_normal[i] = T(0);
        grad2_view[i] = T(0);
        grad2_light[i] = T(0);
    }
    *grad2_roughness = T(0);
    *grad2_f0 = T(0);

    roughness = std::clamp(roughness, min_roughness<T>(), T(1));

    T n_dot_l = dot3(normal, light);
    T n_dot_v = dot3(normal, view);

    // Early out for back-facing geometry
    if (n_dot_l <= T(0) || n_dot_v <= T(0)) {
        return;
    }

    T n_dot_l_clamped = std::max(n_dot_l, dot_epsilon<T>());
    T n_dot_v_clamped = std::max(n_dot_v, dot_epsilon<T>());

    T h[3] = { light[0] + view[0], light[1] + view[1], light[2] + view[2] };
    T h_len = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);

    if (h_len < dot_epsilon<T>()) {
        return;
    }

    T inv_h_len = T(1) / h_len;
    T inv_h_len_sq = inv_h_len * inv_h_len;
    T inv_h_len_cu = inv_h_len_sq * inv_h_len;
    T h_normalized[3] = { h[0] * inv_h_len, h[1] * inv_h_len, h[2] * inv_h_len };

    T n_dot_h = std::max(dot3(normal, h_normalized), dot_epsilon<T>());
    T h_dot_v = std::max(dot3(h_normalized, view), dot_epsilon<T>());

    T alpha = roughness * roughness;
    T alpha_squared = alpha * alpha;

    // =========================================================================
    // Forward pass values (needed for derivatives)
    // =========================================================================

    // D term
    T n_dot_h_sq = n_dot_h * n_dot_h;
    T denom_d = n_dot_h_sq * (alpha_squared - T(1)) + T(1);
    T denom_d_sq = denom_d * denom_d;
    T denom_d_cu = denom_d_sq * denom_d;
    const T PI = T(3.14159265358979323846);
    T D = alpha_squared / (PI * denom_d_sq);

    // G term
    T r_plus_1 = roughness + T(1);
    T k = (r_plus_1 * r_plus_1) / T(8);
    T one_minus_k = T(1) - k;
    T g1_v_denom = n_dot_v_clamped * one_minus_k + k;
    T g1_l_denom = n_dot_l_clamped * one_minus_k + k;
    T g1_v_denom_sq = g1_v_denom * g1_v_denom;
    T g1_l_denom_sq = g1_l_denom * g1_l_denom;
    T G1_v = n_dot_v_clamped / g1_v_denom;
    T G1_l = n_dot_l_clamped / g1_l_denom;
    T G = G1_v * G1_l;

    // F term
    T one_minus_hdv = T(1) - h_dot_v;
    T pow2 = one_minus_hdv * one_minus_hdv;
    T pow3 = pow2 * one_minus_hdv;
    T pow4 = pow2 * pow2;
    T pow5 = pow4 * one_minus_hdv;
    T one_minus_f0 = T(1) - f0;
    T F = f0 + one_minus_f0 * pow5;

    T denom = T(4) * n_dot_l_clamped * n_dot_v_clamped;
    T inv_denom = T(1) / denom;
    T brdf = (D * G * F) * inv_denom;

    // =========================================================================
    // First derivatives (from backward pass)
    // =========================================================================

    // Derivatives of F
    T dF_df0 = T(1) - pow5;
    T dF_dhdotv = -T(5) * one_minus_f0 * pow4;

    // Derivatives of D w.r.t. n_dot_h and alpha_squared
    T dD_dndoth = -T(4) * alpha_squared * n_dot_h * (alpha_squared - T(1)) / (PI * denom_d_cu);
    T dD_dalpha_sq = (denom_d - T(2) * alpha_squared * n_dot_h_sq) / (PI * denom_d_cu);

    // Derivatives of alpha_squared w.r.t. roughness
    T dalpha_droughness = T(2) * roughness;
    T dalpha_sq_droughness = T(4) * alpha * roughness;

    // Derivatives of G w.r.t. n_dot_v, n_dot_l, and k
    T dG1_v_dndotv = k / g1_v_denom_sq;
    T dG1_l_dndotl = k / g1_l_denom_sq;
    T dG1_v_dk = n_dot_v_clamped * (n_dot_v_clamped - T(1)) / g1_v_denom_sq;
    T dG1_l_dk = n_dot_l_clamped * (n_dot_l_clamped - T(1)) / g1_l_denom_sq;

    // Derivatives of k w.r.t. roughness
    T dk_droughness = r_plus_1 / T(4);

    // Derivatives of G w.r.t. inputs
    T dG_dndotv = dG1_v_dndotv * G1_l;
    T dG_dndotl = G1_v * dG1_l_dndotl;
    T dG_dk = dG1_v_dk * G1_l + G1_v * dG1_l_dk;
    T dG_droughness = dG_dk * dk_droughness;

    // Composite derivatives of brdf
    T d_brdf_dF = D * G * inv_denom;
    T d_brdf_dD = G * F * inv_denom;
    T d_brdf_dG = D * F * inv_denom;

    T d_brdf_dndotv = d_brdf_dG * dG_dndotv - brdf / n_dot_v_clamped;
    T d_brdf_dndotl = d_brdf_dG * dG_dndotl - brdf / n_dot_l_clamped;
    T d_brdf_dndoth = d_brdf_dD * dD_dndoth;
    T d_brdf_dhdotv = d_brdf_dF * dF_dhdotv;

    // Derivatives of h_normalized w.r.t. view and light
    T dndoth_dv[3], dndoth_dl[3];
    T dhdotv_dv[3], dhdotv_dl[3];
    for (int i = 0; i < 3; ++i) {
        dndoth_dv[i] = (normal[i] - n_dot_h * h_normalized[i]) * inv_h_len;
        dndoth_dl[i] = dndoth_dv[i];
        dhdotv_dv[i] = h_normalized[i] + (view[i] - h_dot_v * h_normalized[i]) * inv_h_len;
        dhdotv_dl[i] = (view[i] - h_dot_v * h_normalized[i]) * inv_h_len;
    }

    // =========================================================================
    // Collect gg vectors for convenience
    // =========================================================================
    T gg_normal[3] = {gg_normal_x, gg_normal_y, gg_normal_z};
    T gg_view[3] = {gg_view_x, gg_view_y, gg_view_z};
    T gg_light[3] = {gg_light_x, gg_light_y, gg_light_z};

    // =========================================================================
    // grad_grad_out: gradient w.r.t. grad_out
    // =========================================================================

    *grad_grad_out += gg_f0 * d_brdf_dF * dF_df0;

    T d_grad_roughness_d_grad_out = d_brdf_dD * dD_dalpha_sq * dalpha_sq_droughness + d_brdf_dG * dG_droughness;
    *grad_grad_out += gg_roughness * d_grad_roughness_d_grad_out;

    for (int i = 0; i < 3; ++i) {
        T d_grad_normal_i_d_grad_out = d_brdf_dndotl * light[i] +
                                        d_brdf_dndotv * view[i] +
                                        d_brdf_dndoth * h_normalized[i];
        *grad_grad_out += gg_normal[i] * d_grad_normal_i_d_grad_out;
    }

    for (int i = 0; i < 3; ++i) {
        T d_grad_view_i_d_grad_out = d_brdf_dndotv * normal[i] +
                                      d_brdf_dndoth * dndoth_dv[i] +
                                      d_brdf_dhdotv * dhdotv_dv[i];
        *grad_grad_out += gg_view[i] * d_grad_view_i_d_grad_out;
    }

    for (int i = 0; i < 3; ++i) {
        T d_grad_light_i_d_grad_out = d_brdf_dndotl * normal[i] +
                                       d_brdf_dndoth * dndoth_dl[i] +
                                       d_brdf_dhdotv * dhdotv_dl[i];
        *grad_grad_out += gg_light[i] * d_grad_light_i_d_grad_out;
    }

    // =========================================================================
    // Second derivatives of intermediate quantities
    // =========================================================================

    // Second derivative of D w.r.t. n_dot_h
    T d2D_dndoth2;
    {
        T A = -T(4) * alpha_squared * (alpha_squared - T(1)) / PI;
        T factor = T(1) - T(6) * n_dot_h_sq * (alpha_squared - T(1)) / denom_d;
        d2D_dndoth2 = A / denom_d_cu * factor;
    }

    // Second derivative of F w.r.t. h_dot_v
    T d2F_dhdotv2 = T(20) * one_minus_f0 * pow3;

    // Cross derivative of D w.r.t. n_dot_h and alpha_squared
    T d2D_dndoth_dalpha_sq;
    {
        T dC_dalpha_sq = -T(4) * (T(2) * alpha_squared - T(1)) / PI;
        T C = -T(4) * alpha_squared * (alpha_squared - T(1)) / PI;
        T d_inv_denom_d_cu_dalpha_sq = -T(3) / (denom_d_sq * denom_d_sq) * n_dot_h_sq;
        d2D_dndoth_dalpha_sq = dC_dalpha_sq * n_dot_h / denom_d_cu + C * n_dot_h * d_inv_denom_d_cu_dalpha_sq;
    }

    // Second derivative of D w.r.t. alpha_squared
    T d2D_dalpha_sq2;
    {
        T num = denom_d - T(2) * alpha_squared * n_dot_h_sq;
        d2D_dalpha_sq2 = (-n_dot_h_sq) / (PI * denom_d_cu) + num * (-T(3)) / (PI * denom_d_sq * denom_d_sq) * n_dot_h_sq;
    }

    // Second derivatives of G1 terms
    T d2G1_v_dndotv2 = -T(2) * k * one_minus_k / (g1_v_denom_sq * g1_v_denom);
    T d2G1_l_dndotl2 = -T(2) * k * one_minus_k / (g1_l_denom_sq * g1_l_denom);

    // Cross derivatives of G1 w.r.t. n_dot and k
    T d2G1_v_dndotv_dk = T(1)/g1_v_denom_sq + k * (-T(2)) / (g1_v_denom_sq * g1_v_denom) * (T(1) - n_dot_v_clamped);
    T d2G1_l_dndotl_dk = T(1)/g1_l_denom_sq + k * (-T(2)) / (g1_l_denom_sq * g1_l_denom) * (T(1) - n_dot_l_clamped);

    // Second derivative of G1_v w.r.t. k
    T d2G1_v_dk2 = T(2) * n_dot_v_clamped * (T(1) - n_dot_v_clamped) * (T(1) - n_dot_v_clamped) / (g1_v_denom_sq * g1_v_denom);
    T d2G1_l_dk2 = T(2) * n_dot_l_clamped * (T(1) - n_dot_l_clamped) * (T(1) - n_dot_l_clamped) / (g1_l_denom_sq * g1_l_denom);

    // Second derivative of k w.r.t. roughness
    T d2k_droughness2 = T(1) / T(4);

    // Second derivative of alpha_squared w.r.t. roughness
    T d2alpha_sq_droughness2 = T(12) * roughness * roughness;

    // =========================================================================
    // Second derivatives of composite brdf terms
    // =========================================================================

    T d_d_brdf_dF_dD = G * inv_denom;
    T d_d_brdf_dF_dG = D * inv_denom;
    T d_d_brdf_dF_d_inv_denom = D * G;

    T d_d_brdf_dD_dF = G * inv_denom;
    T d_d_brdf_dD_dG = F * inv_denom;
    T d_d_brdf_dD_d_inv_denom = G * F;

    T d_d_brdf_dG_dF = D * inv_denom;
    T d_d_brdf_dG_dD = F * inv_denom;
    T d_d_brdf_dG_d_inv_denom = D * F;

    T d_inv_denom_dndotl = -inv_denom / n_dot_l_clamped;
    T d_inv_denom_dndotv = -inv_denom / n_dot_v_clamped;

    // =========================================================================
    // grad2_f0: gradient w.r.t. f0
    // =========================================================================

    T d_d_brdf_dD_df0 = d_d_brdf_dD_dF * dF_df0;
    T d_d_brdf_dG_df0 = d_d_brdf_dG_dF * dF_df0;
    T d_grad_roughness_df0 = grad_out * (d_d_brdf_dD_df0 * dD_dalpha_sq * dalpha_sq_droughness +
                                          d_d_brdf_dG_df0 * dG_droughness);
    *grad2_f0 += gg_roughness * d_grad_roughness_df0;

    T d_brdf_df0 = d_brdf_dF * dF_df0;
    T d_d_brdf_dndotl_df0 = d_d_brdf_dG_df0 * dG_dndotl - d_brdf_df0 / n_dot_l_clamped;
    T d_d_brdf_dndotv_df0 = d_d_brdf_dG_df0 * dG_dndotv - d_brdf_df0 / n_dot_v_clamped;
    T d_d_brdf_dndoth_df0 = d_d_brdf_dD_df0 * dD_dndoth;
    T d_dF_dhdotv_df0 = T(5) * pow4;
    T d_d_brdf_dhdotv_df0 = d_brdf_dF * d_dF_dhdotv_df0;

    for (int i = 0; i < 3; ++i) {
        T d_grad_normal_i_df0 = grad_out * (d_d_brdf_dndotl_df0 * light[i] +
                                             d_d_brdf_dndotv_df0 * view[i] +
                                             d_d_brdf_dndoth_df0 * h_normalized[i]);
        *grad2_f0 += gg_normal[i] * d_grad_normal_i_df0;

        T d_grad_view_i_df0 = grad_out * (d_d_brdf_dndotv_df0 * normal[i] +
                                           d_d_brdf_dndoth_df0 * dndoth_dv[i] +
                                           d_d_brdf_dhdotv_df0 * dhdotv_dv[i]);
        *grad2_f0 += gg_view[i] * d_grad_view_i_df0;

        T d_grad_light_i_df0 = grad_out * (d_d_brdf_dndotl_df0 * normal[i] +
                                            d_d_brdf_dndoth_df0 * dndoth_dl[i] +
                                            d_d_brdf_dhdotv_df0 * dhdotv_dl[i]);
        *grad2_f0 += gg_light[i] * d_grad_light_i_df0;
    }

    // =========================================================================
    // grad2_roughness: gradient w.r.t. roughness
    // =========================================================================

    T dD_droughness = dD_dalpha_sq * dalpha_sq_droughness;
    T d_brdf_droughness = (dD_droughness * G * F + D * dG_droughness * F) * inv_denom;

    T d_d_brdf_dD_dr = dG_droughness * F * inv_denom;
    T d_d_brdf_dG_dr = dD_droughness * F * inv_denom;
    T d_d_brdf_dF_dr = (dD_droughness * G + D * dG_droughness) * inv_denom;

    T d_dD_dalpha_sq_dr = d2D_dalpha_sq2 * dalpha_sq_droughness;
    T d_dalpha_sq_dr_dr = d2alpha_sq_droughness2;

    T d_dG1_v_dk_dr = d2G1_v_dk2 * dk_droughness;
    T d_dG1_l_dk_dr = d2G1_l_dk2 * dk_droughness;
    T dG1_v_dr = dG1_v_dk * dk_droughness;
    T dG1_l_dr = dG1_l_dk * dk_droughness;
    T d_dk_dr_dr = d2k_droughness2;

    T d_dG_dr_dr = (d_dG1_v_dk_dr * G1_l + dG1_v_dk * dG1_l_dr +
                    dG1_v_dr * dG1_l_dk + G1_v * d_dG1_l_dk_dr) * dk_droughness +
                   (dG1_v_dk * G1_l + G1_v * dG1_l_dk) * d_dk_dr_dr;

    T d_dD_dndoth_dr = d2D_dndoth_dalpha_sq * dalpha_sq_droughness;

    T d_dG1_l_dndotl_dr = d2G1_l_dndotl_dk * dk_droughness;
    T d_dG1_v_dndotv_dr = d2G1_v_dndotv_dk * dk_droughness;
    T d_dG_dndotl_dr = dG1_v_dr * dG1_l_dndotl + G1_v * d_dG1_l_dndotl_dr;
    T d_dG_dndotv_dr = d_dG1_v_dndotv_dr * G1_l + dG1_v_dndotv * dG1_l_dr;

    T d_d_brdf_dndotl_dr = d_d_brdf_dG_dr * dG_dndotl + d_brdf_dG * d_dG_dndotl_dr - d_brdf_droughness / n_dot_l_clamped;
    T d_d_brdf_dndotv_dr = d_d_brdf_dG_dr * dG_dndotv + d_brdf_dG * d_dG_dndotv_dr - d_brdf_droughness / n_dot_v_clamped;
    T d_d_brdf_dndoth_dr = d_d_brdf_dD_dr * dD_dndoth + d_brdf_dD * d_dD_dndoth_dr;

    T d_grad_roughness_dr = grad_out * (d_d_brdf_dD_dr * dD_dalpha_sq * dalpha_sq_droughness +
                                         d_brdf_dD * d_dD_dalpha_sq_dr * dalpha_sq_droughness +
                                         d_brdf_dD * dD_dalpha_sq * d_dalpha_sq_dr_dr +
                                         d_d_brdf_dG_dr * dG_droughness +
                                         d_brdf_dG * d_dG_dr_dr);
    *grad2_roughness += gg_roughness * d_grad_roughness_dr;

    T d_grad_f0_dr = grad_out * d_d_brdf_dF_dr * dF_df0;
    *grad2_roughness += gg_f0 * d_grad_f0_dr;

    for (int i = 0; i < 3; ++i) {
        T d_grad_normal_i_dr = grad_out * (d_d_brdf_dndotl_dr * light[i] +
                                            d_d_brdf_dndotv_dr * view[i] +
                                            d_d_brdf_dndoth_dr * h_normalized[i]);
        *grad2_roughness += gg_normal[i] * d_grad_normal_i_dr;

        T d_d_brdf_dhdotv_dr = d_d_brdf_dF_dr * dF_dhdotv;

        T d_grad_view_i_dr = grad_out * (d_d_brdf_dndotv_dr * normal[i] +
                                          d_d_brdf_dndoth_dr * dndoth_dv[i] +
                                          d_d_brdf_dhdotv_dr * dhdotv_dv[i]);
        *grad2_roughness += gg_view[i] * d_grad_view_i_dr;

        T d_grad_light_i_dr = grad_out * (d_d_brdf_dndotl_dr * normal[i] +
                                           d_d_brdf_dndoth_dr * dndoth_dl[i] +
                                           d_d_brdf_dhdotv_dr * dhdotv_dl[i]);
        *grad2_roughness += gg_light[i] * d_grad_light_i_dr;
    }

    // =========================================================================
    // grad2_normal: gradient w.r.t. normal
    // =========================================================================

    for (int j = 0; j < 3; ++j) {
        T l_j = light[j];
        T v_j = view[j];
        T h_j = h_normalized[j];

        T d_ndotl_dnj = l_j;
        T d_ndotv_dnj = v_j;
        T d_ndoth_dnj = h_j;

        T d_D_dnj = dD_dndoth * h_j;
        T d_G_dnj = dG_dndotv * v_j + dG_dndotl * l_j;
        T d_inv_denom_dnj = -inv_denom * (l_j / n_dot_l_clamped + v_j / n_dot_v_clamped);

        T d_brdf_dnj = d_brdf_dndotl * l_j + d_brdf_dndotv * v_j + d_brdf_dndoth * h_j;

        T d_d_brdf_dD_dnj = d_G_dnj * F * inv_denom + G * F * d_inv_denom_dnj;
        T d_d_brdf_dG_dnj = d_D_dnj * F * inv_denom + D * F * d_inv_denom_dnj;
        T d_d_brdf_dF_dnj = d_D_dnj * G * inv_denom + D * d_G_dnj * inv_denom + D * G * d_inv_denom_dnj;

        T d_dG1_l_dndotl_dnj = d2G1_l_dndotl2 * l_j;
        T d_dG_dndotl_dnj = dG1_v_dndotv * v_j * dG1_l_dndotl + G1_v * d_dG1_l_dndotl_dnj;

        T d_dG1_v_dndotv_dnj = d2G1_v_dndotv2 * v_j;
        T d_dG_dndotv_dnj = d_dG1_v_dndotv_dnj * G1_l + dG1_v_dndotv * dG1_l_dndotl * l_j;

        T d_dD_dndoth_dnj = d2D_dndoth2 * h_j;

        T d_d_brdf_dndotl_dnj = d_d_brdf_dG_dnj * dG_dndotl + d_brdf_dG * d_dG_dndotl_dnj
                                - d_brdf_dnj / n_dot_l_clamped + brdf / (n_dot_l_clamped * n_dot_l_clamped) * l_j;

        T d_d_brdf_dndotv_dnj = d_d_brdf_dG_dnj * dG_dndotv + d_brdf_dG * d_dG_dndotv_dnj
                                - d_brdf_dnj / n_dot_v_clamped + brdf / (n_dot_v_clamped * n_dot_v_clamped) * v_j;

        T d_d_brdf_dndoth_dnj = d_d_brdf_dD_dnj * dD_dndoth + d_brdf_dD * d_dD_dndoth_dnj;

        grad2_normal[j] += gg_f0 * grad_out * d_d_brdf_dF_dnj * dF_df0;

        T d_dD_dalpha_sq_dnj = d2D_dndoth_dalpha_sq * h_j;
        T d_grad_roughness_dnj = grad_out * (d_d_brdf_dD_dnj * dD_dalpha_sq * dalpha_sq_droughness +
                                              d_brdf_dD * d_dD_dalpha_sq_dnj * dalpha_sq_droughness +
                                              d_d_brdf_dG_dnj * dG_droughness);
        grad2_normal[j] += gg_roughness * d_grad_roughness_dnj;

        for (int i = 0; i < 3; ++i) {
            T d_grad_normal_i_dnj = grad_out * (d_d_brdf_dndotl_dnj * light[i] +
                                                 d_d_brdf_dndotv_dnj * view[i] +
                                                 d_d_brdf_dndoth_dnj * h_normalized[i]);
            grad2_normal[j] += gg_normal[i] * d_grad_normal_i_dnj;
        }

        T d_d_brdf_dhdotv_dnj = d_d_brdf_dF_dnj * dF_dhdotv;

        T d_dndoth_dv_i_dnj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            d_dndoth_dv_i_dnj[i] = (delta_ij - h_j * h_normalized[i]) * inv_h_len;
        }

        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_grad_view_i_dnj = grad_out * (d_d_brdf_dndotv_dnj * normal[i] + d_brdf_dndotv * delta_ij +
                                               d_d_brdf_dndoth_dnj * dndoth_dv[i] + d_brdf_dndoth * d_dndoth_dv_i_dnj[i] +
                                               d_d_brdf_dhdotv_dnj * dhdotv_dv[i]);
            grad2_normal[j] += gg_view[i] * d_grad_view_i_dnj;
        }

        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_dndoth_dl_i_dnj = d_dndoth_dv_i_dnj[i];
            T d_grad_light_i_dnj = grad_out * (d_d_brdf_dndotl_dnj * normal[i] + d_brdf_dndotl * delta_ij +
                                                d_d_brdf_dndoth_dnj * dndoth_dl[i] + d_brdf_dndoth * d_dndoth_dl_i_dnj +
                                                d_d_brdf_dhdotv_dnj * dhdotv_dl[i]);
            grad2_normal[j] += gg_light[i] * d_grad_light_i_dnj;
        }
    }

    // =========================================================================
    // grad2_view: gradient w.r.t. view
    // =========================================================================

    for (int j = 0; j < 3; ++j) {
        T n_j = normal[j];
        T v_j = view[j];
        T h_j = h_normalized[j];

        T d_ndotv_dvj = n_j;
        T d_ndoth_dvj = dndoth_dv[j];
        T d_hdotv_dvj = dhdotv_dv[j];

        T d_D_dvj = dD_dndoth * d_ndoth_dvj;
        T d_G_dvj = dG_dndotv * n_j;
        T d_F_dvj = dF_dhdotv * d_hdotv_dvj;
        T d_inv_denom_dvj = -inv_denom * n_j / n_dot_v_clamped;

        T d_brdf_dvj = d_brdf_dndotv * n_j + d_brdf_dndoth * d_ndoth_dvj + d_brdf_dhdotv * d_hdotv_dvj;

        T d_d_brdf_dD_dvj = d_G_dvj * F * inv_denom + G * d_F_dvj * inv_denom + G * F * d_inv_denom_dvj;
        T d_d_brdf_dG_dvj = d_D_dvj * F * inv_denom + D * d_F_dvj * inv_denom + D * F * d_inv_denom_dvj;
        T d_d_brdf_dF_dvj = d_D_dvj * G * inv_denom + D * d_G_dvj * inv_denom + D * G * d_inv_denom_dvj;

        T d_dG_dndotv_dvj = d2G1_v_dndotv2 * n_j * G1_l;
        T d_dD_dndoth_dvj = d2D_dndoth2 * d_ndoth_dvj;
        T d_dF_dhdotv_dvj = d2F_dhdotv2 * d_hdotv_dvj;

        T d_d_brdf_dndotv_dvj = d_d_brdf_dG_dvj * dG_dndotv + d_brdf_dG * d_dG_dndotv_dvj
                                - d_brdf_dvj / n_dot_v_clamped + brdf / (n_dot_v_clamped * n_dot_v_clamped) * n_j;

        T d_d_brdf_dndotl_dvj = d_d_brdf_dG_dvj * dG_dndotl - d_brdf_dvj / n_dot_l_clamped;

        T d_d_brdf_dndoth_dvj = d_d_brdf_dD_dvj * dD_dndoth + d_brdf_dD * d_dD_dndoth_dvj;

        T d_d_brdf_dhdotv_dvj = d_d_brdf_dF_dvj * dF_dhdotv + d_brdf_dF * d_dF_dhdotv_dvj;

        grad2_view[j] += gg_f0 * grad_out * d_d_brdf_dF_dvj * dF_df0;

        T d_dD_dalpha_sq_dvj = d2D_dndoth_dalpha_sq * d_ndoth_dvj;
        T d_grad_roughness_dvj = grad_out * (d_d_brdf_dD_dvj * dD_dalpha_sq * dalpha_sq_droughness +
                                              d_brdf_dD * d_dD_dalpha_sq_dvj * dalpha_sq_droughness +
                                              d_d_brdf_dG_dvj * dG_droughness);
        grad2_view[j] += gg_roughness * d_grad_roughness_dvj;

        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_h_normalized_i_dvj = (delta_ij * inv_h_len - h_normalized[i] * h_j * inv_h_len);
            T d_grad_normal_i_dvj = grad_out * (d_d_brdf_dndotl_dvj * light[i] +
                                                 d_d_brdf_dndotv_dvj * view[i] + d_brdf_dndotv * delta_ij +
                                                 d_d_brdf_dndoth_dvj * h_normalized[i] + d_brdf_dndoth * d_h_normalized_i_dvj);
            grad2_view[j] += gg_normal[i] * d_grad_normal_i_dvj;
        }

        T d_inv_h_len_dvj = -h_j * inv_h_len_sq;
        T d2ndoth_dv_i_dvj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_ndoth_times_hi_dvj = dndoth_dv[j] * h_normalized[i] + n_dot_h * (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            d2ndoth_dv_i_dvj[i] = (-d_ndoth_times_hi_dvj) * inv_h_len + (normal[i] - n_dot_h * h_normalized[i]) * d_inv_h_len_dvj;
        }

        T d2hdotv_dv_i_dvj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_h_i_dvj = (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            T d_hdotv_times_hi_dvj = dhdotv_dv[j] * h_normalized[i] + h_dot_v * d_h_i_dvj;
            T d_vi_minus_term_dvj = delta_ij - d_hdotv_times_hi_dvj;
            d2hdotv_dv_i_dvj[i] = d_h_i_dvj + d_vi_minus_term_dvj * inv_h_len + (view[i] - h_dot_v * h_normalized[i]) * d_inv_h_len_dvj;
        }

        for (int i = 0; i < 3; ++i) {
            T d_grad_view_i_dvj = grad_out * (d_d_brdf_dndotv_dvj * normal[i] +
                                               d_d_brdf_dndoth_dvj * dndoth_dv[i] + d_brdf_dndoth * d2ndoth_dv_i_dvj[i] +
                                               d_d_brdf_dhdotv_dvj * dhdotv_dv[i] + d_brdf_dhdotv * d2hdotv_dv_i_dvj[i]);
            grad2_view[j] += gg_view[i] * d_grad_view_i_dvj;
        }

        T d2hdotv_dl_i_dvj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_h_i_dvj = (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            T d_hdotv_times_hi_dvj = dhdotv_dv[j] * h_normalized[i] + h_dot_v * d_h_i_dvj;
            T d_vi_minus_term_dvj = delta_ij - d_hdotv_times_hi_dvj;
            d2hdotv_dl_i_dvj[i] = d_vi_minus_term_dvj * inv_h_len + (view[i] - h_dot_v * h_normalized[i]) * d_inv_h_len_dvj;
        }

        for (int i = 0; i < 3; ++i) {
            T d_grad_light_i_dvj = grad_out * (d_d_brdf_dndotl_dvj * normal[i] +
                                                d_d_brdf_dndoth_dvj * dndoth_dl[i] + d_brdf_dndoth * d2ndoth_dv_i_dvj[i] +
                                                d_d_brdf_dhdotv_dvj * dhdotv_dl[i] + d_brdf_dhdotv * d2hdotv_dl_i_dvj[i]);
            grad2_view[j] += gg_light[i] * d_grad_light_i_dvj;
        }
    }

    // =========================================================================
    // grad2_light: gradient w.r.t. light
    // =========================================================================

    for (int j = 0; j < 3; ++j) {
        T n_j = normal[j];
        T l_j = light[j];
        T h_j = h_normalized[j];

        T d_ndotl_dlj = n_j;
        T d_ndoth_dlj = dndoth_dl[j];
        T d_hdotv_dlj = dhdotv_dl[j];

        T d_D_dlj = dD_dndoth * d_ndoth_dlj;
        T d_G_dlj = dG_dndotl * n_j;
        T d_F_dlj = dF_dhdotv * d_hdotv_dlj;
        T d_inv_denom_dlj = -inv_denom * n_j / n_dot_l_clamped;

        T d_brdf_dlj = d_brdf_dndotl * n_j + d_brdf_dndoth * d_ndoth_dlj + d_brdf_dhdotv * d_hdotv_dlj;

        T d_d_brdf_dD_dlj = d_G_dlj * F * inv_denom + G * d_F_dlj * inv_denom + G * F * d_inv_denom_dlj;
        T d_d_brdf_dG_dlj = d_D_dlj * F * inv_denom + D * d_F_dlj * inv_denom + D * F * d_inv_denom_dlj;
        T d_d_brdf_dF_dlj = d_D_dlj * G * inv_denom + D * d_G_dlj * inv_denom + D * G * d_inv_denom_dlj;

        T d_dG_dndotl_dlj = G1_v * d2G1_l_dndotl2 * n_j;
        T d_dD_dndoth_dlj = d2D_dndoth2 * d_ndoth_dlj;
        T d_dF_dhdotv_dlj = d2F_dhdotv2 * d_hdotv_dlj;

        T d_d_brdf_dndotl_dlj = d_d_brdf_dG_dlj * dG_dndotl + d_brdf_dG * d_dG_dndotl_dlj
                                - d_brdf_dlj / n_dot_l_clamped + brdf / (n_dot_l_clamped * n_dot_l_clamped) * n_j;

        T d_d_brdf_dndotv_dlj = d_d_brdf_dG_dlj * dG_dndotv - d_brdf_dlj / n_dot_v_clamped;

        T d_d_brdf_dndoth_dlj = d_d_brdf_dD_dlj * dD_dndoth + d_brdf_dD * d_dD_dndoth_dlj;

        T d_d_brdf_dhdotv_dlj = d_d_brdf_dF_dlj * dF_dhdotv + d_brdf_dF * d_dF_dhdotv_dlj;

        grad2_light[j] += gg_f0 * grad_out * d_d_brdf_dF_dlj * dF_df0;

        T d_dD_dalpha_sq_dlj = d2D_dndoth_dalpha_sq * d_ndoth_dlj;
        T d_grad_roughness_dlj = grad_out * (d_d_brdf_dD_dlj * dD_dalpha_sq * dalpha_sq_droughness +
                                              d_brdf_dD * d_dD_dalpha_sq_dlj * dalpha_sq_droughness +
                                              d_d_brdf_dG_dlj * dG_droughness);
        grad2_light[j] += gg_roughness * d_grad_roughness_dlj;

        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_h_normalized_i_dlj = (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            T d_grad_normal_i_dlj = grad_out * (d_d_brdf_dndotl_dlj * light[i] + d_brdf_dndotl * delta_ij +
                                                 d_d_brdf_dndotv_dlj * view[i] +
                                                 d_d_brdf_dndoth_dlj * h_normalized[i] + d_brdf_dndoth * d_h_normalized_i_dlj);
            grad2_light[j] += gg_normal[i] * d_grad_normal_i_dlj;
        }

        T d_inv_h_len_dlj = -h_j * inv_h_len_sq;
        T d2ndoth_dv_i_dlj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_ndoth_times_hi_dlj = dndoth_dl[j] * h_normalized[i] + n_dot_h * (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            d2ndoth_dv_i_dlj[i] = (-d_ndoth_times_hi_dlj) * inv_h_len + (normal[i] - n_dot_h * h_normalized[i]) * d_inv_h_len_dlj;
        }

        T d2hdotv_dv_i_dlj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_h_i_dlj = (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            T d_hdotv_times_hi_dlj = dhdotv_dl[j] * h_normalized[i] + h_dot_v * d_h_i_dlj;
            T d_vi_minus_term_dlj = -d_hdotv_times_hi_dlj;
            d2hdotv_dv_i_dlj[i] = d_h_i_dlj + d_vi_minus_term_dlj * inv_h_len + (view[i] - h_dot_v * h_normalized[i]) * d_inv_h_len_dlj;
        }

        for (int i = 0; i < 3; ++i) {
            T d_grad_view_i_dlj = grad_out * (d_d_brdf_dndotv_dlj * normal[i] +
                                               d_d_brdf_dndoth_dlj * dndoth_dv[i] + d_brdf_dndoth * d2ndoth_dv_i_dlj[i] +
                                               d_d_brdf_dhdotv_dlj * dhdotv_dv[i] + d_brdf_dhdotv * d2hdotv_dv_i_dlj[i]);
            grad2_light[j] += gg_view[i] * d_grad_view_i_dlj;
        }

        T d2ndoth_dl_i_dlj[3];
        for (int i = 0; i < 3; ++i) {
            d2ndoth_dl_i_dlj[i] = d2ndoth_dv_i_dlj[i];
        }

        T d2hdotv_dl_i_dlj[3];
        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_h_i_dlj = (delta_ij - h_normalized[i] * h_j) * inv_h_len;
            T d_hdotv_times_hi_dlj = dhdotv_dl[j] * h_normalized[i] + h_dot_v * d_h_i_dlj;
            T d_vi_minus_term_dlj = -d_hdotv_times_hi_dlj;
            d2hdotv_dl_i_dlj[i] = d_vi_minus_term_dlj * inv_h_len + (view[i] - h_dot_v * h_normalized[i]) * d_inv_h_len_dlj;
        }

        for (int i = 0; i < 3; ++i) {
            T delta_ij = (i == j) ? T(1) : T(0);
            T d_grad_light_i_dlj = grad_out * (d_d_brdf_dndotl_dlj * normal[i] + d_brdf_dndotl * delta_ij +
                                                d_d_brdf_dndoth_dlj * dndoth_dl[i] + d_brdf_dndoth * d2ndoth_dl_i_dlj[i] +
                                                d_d_brdf_dhdotv_dlj * dhdotv_dl[i] + d_brdf_dhdotv * d2hdotv_dl_i_dlj[i]);
            grad2_light[j] += gg_light[i] * d_grad_light_i_dlj;
        }
    }
}

}  // anonymous namespace

/**
 * CPU implementation of Cook-Torrance specular BRDF.
 *
 * @param normal Surface normal vectors, shape (..., 3)
 * @param view View direction vectors, shape (..., 3)
 * @param light Light direction vectors, shape (..., 3)
 * @param roughness Surface roughness, shape (...) or scalar
 * @param f0 Fresnel reflectance at normal incidence, shape (...) or (..., 3)
 * @return BRDF values, shape (...) or (..., 3)
 */
inline at::Tensor cook_torrance(
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    TORCH_CHECK(normal.size(-1) == 3, "cook_torrance: normal must have last dimension 3");
    TORCH_CHECK(view.size(-1) == 3, "cook_torrance: view must have last dimension 3");
    TORCH_CHECK(light.size(-1) == 3, "cook_torrance: light must have last dimension 3");

    // Check if f0 is RGB (last dim = 3)
    bool f0_is_rgb = f0.dim() > 0 && f0.size(-1) == 3;

    // Get batch dimensions (excluding the last dim of 3 for vectors)
    auto normal_batch = normal.sizes().slice(0, normal.dim() - 1);
    auto view_batch = view.sizes().slice(0, view.dim() - 1);
    auto light_batch = light.sizes().slice(0, light.dim() - 1);
    auto roughness_batch = roughness.sizes();
    auto f0_batch = f0_is_rgb ? f0.sizes().slice(0, f0.dim() - 1) : f0.sizes();

    // Broadcast batch dimensions
    std::vector<int64_t> batch_shape;
    auto max_batch_dim = std::max({
        (int64_t)normal_batch.size(),
        (int64_t)view_batch.size(),
        (int64_t)light_batch.size(),
        (int64_t)roughness_batch.size(),
        (int64_t)f0_batch.size()
    });

    for (int64_t i = 0; i < max_batch_dim; ++i) {
        int64_t dim = 1;
        auto get_dim = [&](c10::IntArrayRef shape, int64_t offset) -> int64_t {
            int64_t idx = (int64_t)shape.size() - max_batch_dim + offset;
            return idx >= 0 ? shape[idx] : 1;
        };
        dim = std::max(dim, get_dim(normal_batch, i));
        dim = std::max(dim, get_dim(view_batch, i));
        dim = std::max(dim, get_dim(light_batch, i));
        dim = std::max(dim, get_dim(roughness_batch, i));
        dim = std::max(dim, get_dim(f0_batch, i));
        batch_shape.push_back(dim);
    }

    // Compute total number of elements
    int64_t numel = 1;
    for (auto d : batch_shape) {
        numel *= d;
    }

    // Expand tensors to broadcast shape and flatten
    std::vector<int64_t> vec_shape = batch_shape;
    vec_shape.push_back(3);

    at::Tensor normal_expanded = normal.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor view_expanded = view.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor light_expanded = light.expand(vec_shape).contiguous().view({numel, 3});

    // Optimize: avoid full expansion for scalar/broadcasted roughness and f0
    // Use stride=0 for scalars (read same value for all elements)
    bool roughness_is_scalar = roughness.numel() == 1;
    bool f0_is_scalar = !f0_is_rgb && f0.numel() == 1;

    at::Tensor roughness_data = roughness_is_scalar ? roughness.contiguous().view({1}) : roughness.expand(batch_shape).contiguous().view({numel});
    int64_t roughness_stride = roughness_is_scalar ? 0 : 1;

    at::Tensor f0_data;
    int64_t f0_stride;
    if (f0_is_rgb) {
        // For RGB f0, check if it's a single RGB value broadcasted
        bool f0_rgb_is_scalar = f0.numel() == 3;
        f0_data = f0_rgb_is_scalar ? f0.contiguous().view({1, 3}) : f0.expand(vec_shape).contiguous().view({numel, 3});
        f0_stride = f0_rgb_is_scalar ? 0 : 1;
    } else {
        f0_data = f0_is_scalar ? f0.contiguous().view({1}) : f0.expand(batch_shape).contiguous().view({numel});
        f0_stride = f0_is_scalar ? 0 : 1;
    }

    // Create output tensor
    at::Tensor output;
    if (f0_is_rgb) {
        output = at::empty({numel, 3}, normal.options());
    } else {
        output = at::empty({numel}, normal.options());
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        normal.scalar_type(),
        "cook_torrance_cpu",
        [&]() {
            const scalar_t* normal_ptr = normal_expanded.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_expanded.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_expanded.data_ptr<scalar_t>();
            const scalar_t* roughness_ptr = roughness_data.data_ptr<scalar_t>();
            const scalar_t* f0_ptr = f0_data.data_ptr<scalar_t>();
            scalar_t* out_ptr = output.data_ptr<scalar_t>();

            if (f0_is_rgb) {
                at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        const scalar_t* n = normal_ptr + idx * 3;
                        const scalar_t* v = view_ptr + idx * 3;
                        const scalar_t* l = light_ptr + idx * 3;
                        scalar_t r = roughness_ptr[idx * roughness_stride];
                        const scalar_t* f = f0_ptr + idx * f0_stride * 3;

                        // Compute BRDF for each color channel
                        for (int c = 0; c < 3; ++c) {
                            out_ptr[idx * 3 + c] = cook_torrance_scalar<scalar_t>(
                                n, v, l, r, f[c]
                            );
                        }
                    }
                });
            } else {
                at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        const scalar_t* n = normal_ptr + idx * 3;
                        const scalar_t* v = view_ptr + idx * 3;
                        const scalar_t* l = light_ptr + idx * 3;
                        scalar_t r = roughness_ptr[idx * roughness_stride];
                        scalar_t f = f0_ptr[idx * f0_stride];

                        out_ptr[idx] = cook_torrance_scalar<scalar_t>(
                            n, v, l, r, f
                        );
                    }
                });
            }
        }
    );

    // Reshape output to batch shape
    if (f0_is_rgb) {
        std::vector<int64_t> out_shape = batch_shape;
        out_shape.push_back(3);
        return output.view(out_shape);
    } else {
        return output.view(batch_shape);
    }
}

/**
 * Backward pass for Cook-Torrance BRDF.
 *
 * @return Tuple of (grad_normal, grad_view, grad_light, grad_roughness, grad_f0)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cook_torrance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    // Check if f0 is RGB
    bool f0_is_rgb = f0.dim() > 0 && f0.size(-1) == 3;

    // Get batch dimensions
    auto normal_batch = normal.sizes().slice(0, normal.dim() - 1);
    auto view_batch = view.sizes().slice(0, view.dim() - 1);
    auto light_batch = light.sizes().slice(0, light.dim() - 1);
    auto roughness_batch = roughness.sizes();
    auto f0_batch = f0_is_rgb ? f0.sizes().slice(0, f0.dim() - 1) : f0.sizes();

    // Broadcast batch dimensions
    std::vector<int64_t> batch_shape;
    auto max_batch_dim = std::max({
        (int64_t)normal_batch.size(),
        (int64_t)view_batch.size(),
        (int64_t)light_batch.size(),
        (int64_t)roughness_batch.size(),
        (int64_t)f0_batch.size()
    });

    for (int64_t i = 0; i < max_batch_dim; ++i) {
        int64_t dim = 1;
        auto get_dim = [&](c10::IntArrayRef shape, int64_t offset) -> int64_t {
            int64_t idx = (int64_t)shape.size() - max_batch_dim + offset;
            return idx >= 0 ? shape[idx] : 1;
        };
        dim = std::max(dim, get_dim(normal_batch, i));
        dim = std::max(dim, get_dim(view_batch, i));
        dim = std::max(dim, get_dim(light_batch, i));
        dim = std::max(dim, get_dim(roughness_batch, i));
        dim = std::max(dim, get_dim(f0_batch, i));
        batch_shape.push_back(dim);
    }

    int64_t numel = 1;
    for (auto d : batch_shape) {
        numel *= d;
    }

    // Expand tensors
    std::vector<int64_t> vec_shape = batch_shape;
    vec_shape.push_back(3);

    at::Tensor normal_expanded = normal.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor view_expanded = view.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor light_expanded = light.expand(vec_shape).contiguous().view({numel, 3});

    // Optimize: avoid full expansion for scalar/broadcasted roughness and f0
    bool roughness_is_scalar = roughness.numel() == 1;
    bool f0_is_scalar = !f0_is_rgb && f0.numel() == 1;

    at::Tensor roughness_data = roughness_is_scalar ? roughness.contiguous().view({1}) : roughness.expand(batch_shape).contiguous().view({numel});
    int64_t roughness_stride = roughness_is_scalar ? 0 : 1;

    at::Tensor f0_data;
    at::Tensor grad_expanded;
    int64_t f0_stride;
    if (f0_is_rgb) {
        bool f0_rgb_is_scalar = f0.numel() == 3;
        f0_data = f0_rgb_is_scalar ? f0.contiguous().view({1, 3}) : f0.expand(vec_shape).contiguous().view({numel, 3});
        f0_stride = f0_rgb_is_scalar ? 0 : 1;
        grad_expanded = grad_output.contiguous().view({numel, 3});
    } else {
        f0_data = f0_is_scalar ? f0.contiguous().view({1}) : f0.expand(batch_shape).contiguous().view({numel});
        f0_stride = f0_is_scalar ? 0 : 1;
        grad_expanded = grad_output.contiguous().view({numel});
    }

    // Create gradient tensors
    at::Tensor grad_normal = at::zeros({numel, 3}, normal.options());
    at::Tensor grad_view = at::zeros({numel, 3}, normal.options());
    at::Tensor grad_light = at::zeros({numel, 3}, normal.options());
    at::Tensor grad_roughness = at::zeros({numel}, normal.options());
    at::Tensor grad_f0;
    if (f0_is_rgb) {
        grad_f0 = at::zeros({numel, 3}, normal.options());
    } else {
        grad_f0 = at::zeros({numel}, normal.options());
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        normal.scalar_type(),
        "cook_torrance_backward_cpu",
        [&]() {
            const scalar_t* normal_ptr = normal_expanded.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_expanded.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_expanded.data_ptr<scalar_t>();
            const scalar_t* roughness_ptr = roughness_data.data_ptr<scalar_t>();
            const scalar_t* f0_ptr = f0_data.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_expanded.data_ptr<scalar_t>();

            scalar_t* grad_normal_ptr = grad_normal.data_ptr<scalar_t>();
            scalar_t* grad_view_ptr = grad_view.data_ptr<scalar_t>();
            scalar_t* grad_light_ptr = grad_light.data_ptr<scalar_t>();
            scalar_t* grad_roughness_ptr = grad_roughness.data_ptr<scalar_t>();
            scalar_t* grad_f0_ptr = grad_f0.data_ptr<scalar_t>();

            if (f0_is_rgb) {
                at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        const scalar_t* n = normal_ptr + idx * 3;
                        const scalar_t* v = view_ptr + idx * 3;
                        const scalar_t* l = light_ptr + idx * 3;
                        scalar_t r = roughness_ptr[idx * roughness_stride];
                        const scalar_t* f = f0_ptr + idx * f0_stride * 3;
                        const scalar_t* g = grad_ptr + idx * 3;

                        scalar_t* gn = grad_normal_ptr + idx * 3;
                        scalar_t* gv = grad_view_ptr + idx * 3;
                        scalar_t* gl = grad_light_ptr + idx * 3;
                        scalar_t* gr = grad_roughness_ptr + idx;
                        scalar_t* gf = grad_f0_ptr + idx * 3;

                        // Accumulate gradients from each channel
                        for (int c = 0; c < 3; ++c) {
                            scalar_t temp_gn[3], temp_gv[3], temp_gl[3];
                            scalar_t temp_gr, temp_gf;

                            cook_torrance_backward_scalar<scalar_t>(
                                g[c], n, v, l, r, f[c],
                                temp_gn, temp_gv, temp_gl, &temp_gr, &temp_gf
                            );

                            for (int i = 0; i < 3; ++i) {
                                gn[i] += temp_gn[i];
                                gv[i] += temp_gv[i];
                                gl[i] += temp_gl[i];
                            }
                            *gr += temp_gr;
                            gf[c] = temp_gf;
                        }
                    }
                });
            } else {
                at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        const scalar_t* n = normal_ptr + idx * 3;
                        const scalar_t* v = view_ptr + idx * 3;
                        const scalar_t* l = light_ptr + idx * 3;
                        scalar_t r = roughness_ptr[idx * roughness_stride];
                        scalar_t f = f0_ptr[idx * f0_stride];
                        scalar_t g = grad_ptr[idx];

                        scalar_t* gn = grad_normal_ptr + idx * 3;
                        scalar_t* gv = grad_view_ptr + idx * 3;
                        scalar_t* gl = grad_light_ptr + idx * 3;

                        cook_torrance_backward_scalar<scalar_t>(
                            g, n, v, l, r, f,
                            gn, gv, gl, grad_roughness_ptr + idx, grad_f0_ptr + idx
                        );
                    }
                });
            }
        }
    );

    // Reshape gradients back to original shapes
    return std::make_tuple(
        grad_normal.view(vec_shape),
        grad_view.view(vec_shape),
        grad_light.view(vec_shape),
        grad_roughness.view(batch_shape),
        f0_is_rgb ? grad_f0.view(vec_shape) : grad_f0.view(batch_shape)
    );
}

/**
 * Second-order backward pass for Cook-Torrance BRDF.
 *
 * @return Tuple of (grad_grad_output, grad2_normal, grad2_view, grad2_light, grad2_roughness, grad2_f0)
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> cook_torrance_backward_backward(
    const at::Tensor& gg_normal,
    const at::Tensor& gg_view,
    const at::Tensor& gg_light,
    const at::Tensor& gg_roughness,
    const at::Tensor& gg_f0,
    const at::Tensor& grad_output,
    const at::Tensor& normal,
    const at::Tensor& view,
    const at::Tensor& light,
    const at::Tensor& roughness,
    const at::Tensor& f0
) {
    // Check if f0 is RGB
    bool f0_is_rgb = f0.dim() > 0 && f0.size(-1) == 3;

    // Get batch dimensions
    auto normal_batch = normal.sizes().slice(0, normal.dim() - 1);
    auto view_batch = view.sizes().slice(0, view.dim() - 1);
    auto light_batch = light.sizes().slice(0, light.dim() - 1);
    auto roughness_batch = roughness.sizes();
    auto f0_batch = f0_is_rgb ? f0.sizes().slice(0, f0.dim() - 1) : f0.sizes();

    // Broadcast batch dimensions
    std::vector<int64_t> batch_shape;
    auto max_batch_dim = std::max({
        (int64_t)normal_batch.size(),
        (int64_t)view_batch.size(),
        (int64_t)light_batch.size(),
        (int64_t)roughness_batch.size(),
        (int64_t)f0_batch.size()
    });

    for (int64_t i = 0; i < max_batch_dim; ++i) {
        int64_t dim = 1;
        auto get_dim = [&](c10::IntArrayRef shape, int64_t offset) -> int64_t {
            int64_t idx = (int64_t)shape.size() - max_batch_dim + offset;
            return idx >= 0 ? shape[idx] : 1;
        };
        dim = std::max(dim, get_dim(normal_batch, i));
        dim = std::max(dim, get_dim(view_batch, i));
        dim = std::max(dim, get_dim(light_batch, i));
        dim = std::max(dim, get_dim(roughness_batch, i));
        dim = std::max(dim, get_dim(f0_batch, i));
        batch_shape.push_back(dim);
    }

    int64_t numel = 1;
    for (auto d : batch_shape) {
        numel *= d;
    }

    // Expand tensors
    std::vector<int64_t> vec_shape = batch_shape;
    vec_shape.push_back(3);

    at::Tensor normal_expanded = normal.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor view_expanded = view.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor light_expanded = light.expand(vec_shape).contiguous().view({numel, 3});

    at::Tensor gg_normal_expanded = gg_normal.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor gg_view_expanded = gg_view.expand(vec_shape).contiguous().view({numel, 3});
    at::Tensor gg_light_expanded = gg_light.expand(vec_shape).contiguous().view({numel, 3});

    // Optimize: avoid full expansion for scalar/broadcasted roughness and f0
    bool roughness_is_scalar = roughness.numel() == 1;
    bool gg_roughness_is_scalar = gg_roughness.numel() == 1;
    bool f0_is_scalar = !f0_is_rgb && f0.numel() == 1;

    at::Tensor roughness_data = roughness_is_scalar ? roughness.contiguous().view({1}) : roughness.expand(batch_shape).contiguous().view({numel});
    int64_t roughness_stride = roughness_is_scalar ? 0 : 1;

    at::Tensor gg_roughness_data = gg_roughness_is_scalar ? gg_roughness.contiguous().view({1}) : gg_roughness.expand(batch_shape).contiguous().view({numel});
    int64_t gg_roughness_stride = gg_roughness_is_scalar ? 0 : 1;

    at::Tensor f0_data;
    at::Tensor grad_expanded;
    at::Tensor gg_f0_data;
    int64_t f0_stride;
    int64_t gg_f0_stride;
    if (f0_is_rgb) {
        bool f0_rgb_is_scalar = f0.numel() == 3;
        bool gg_f0_rgb_is_scalar = gg_f0.numel() == 3;
        f0_data = f0_rgb_is_scalar ? f0.contiguous().view({1, 3}) : f0.expand(vec_shape).contiguous().view({numel, 3});
        f0_stride = f0_rgb_is_scalar ? 0 : 1;
        gg_f0_data = gg_f0_rgb_is_scalar ? gg_f0.contiguous().view({1, 3}) : gg_f0.expand(vec_shape).contiguous().view({numel, 3});
        gg_f0_stride = gg_f0_rgb_is_scalar ? 0 : 1;
        grad_expanded = grad_output.expand(vec_shape).contiguous().view({numel, 3});
    } else {
        bool gg_f0_is_scalar = gg_f0.numel() == 1;
        f0_data = f0_is_scalar ? f0.contiguous().view({1}) : f0.expand(batch_shape).contiguous().view({numel});
        f0_stride = f0_is_scalar ? 0 : 1;
        gg_f0_data = gg_f0_is_scalar ? gg_f0.contiguous().view({1}) : gg_f0.expand(batch_shape).contiguous().view({numel});
        gg_f0_stride = gg_f0_is_scalar ? 0 : 1;
        grad_expanded = grad_output.expand(batch_shape).contiguous().view({numel});
    }

    // Create output tensors
    at::Tensor grad_grad_output;
    if (f0_is_rgb) {
        grad_grad_output = at::zeros({numel, 3}, normal.options());
    } else {
        grad_grad_output = at::zeros({numel}, normal.options());
    }
    at::Tensor grad2_normal = at::zeros({numel, 3}, normal.options());
    at::Tensor grad2_view = at::zeros({numel, 3}, normal.options());
    at::Tensor grad2_light = at::zeros({numel, 3}, normal.options());
    at::Tensor grad2_roughness = at::zeros({numel}, normal.options());
    at::Tensor grad2_f0;
    if (f0_is_rgb) {
        grad2_f0 = at::zeros({numel, 3}, normal.options());
    } else {
        grad2_f0 = at::zeros({numel}, normal.options());
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        normal.scalar_type(),
        "cook_torrance_backward_backward_cpu",
        [&]() {
            const scalar_t* normal_ptr = normal_expanded.data_ptr<scalar_t>();
            const scalar_t* view_ptr = view_expanded.data_ptr<scalar_t>();
            const scalar_t* light_ptr = light_expanded.data_ptr<scalar_t>();
            const scalar_t* roughness_ptr = roughness_data.data_ptr<scalar_t>();
            const scalar_t* f0_ptr = f0_data.data_ptr<scalar_t>();
            const scalar_t* grad_ptr = grad_expanded.data_ptr<scalar_t>();

            const scalar_t* gg_normal_ptr = gg_normal_expanded.data_ptr<scalar_t>();
            const scalar_t* gg_view_ptr = gg_view_expanded.data_ptr<scalar_t>();
            const scalar_t* gg_light_ptr = gg_light_expanded.data_ptr<scalar_t>();
            const scalar_t* gg_roughness_ptr = gg_roughness_data.data_ptr<scalar_t>();
            const scalar_t* gg_f0_ptr = gg_f0_data.data_ptr<scalar_t>();

            scalar_t* grad_grad_output_ptr = grad_grad_output.data_ptr<scalar_t>();
            scalar_t* grad2_normal_ptr = grad2_normal.data_ptr<scalar_t>();
            scalar_t* grad2_view_ptr = grad2_view.data_ptr<scalar_t>();
            scalar_t* grad2_light_ptr = grad2_light.data_ptr<scalar_t>();
            scalar_t* grad2_roughness_ptr = grad2_roughness.data_ptr<scalar_t>();
            scalar_t* grad2_f0_ptr = grad2_f0.data_ptr<scalar_t>();

            if (f0_is_rgb) {
                at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        const scalar_t* n = normal_ptr + idx * 3;
                        const scalar_t* v = view_ptr + idx * 3;
                        const scalar_t* l = light_ptr + idx * 3;
                        scalar_t r = roughness_ptr[idx * roughness_stride];
                        const scalar_t* f = f0_ptr + idx * f0_stride * 3;

                        const scalar_t* gg_n = gg_normal_ptr + idx * 3;
                        const scalar_t* gg_v = gg_view_ptr + idx * 3;
                        const scalar_t* gg_l = gg_light_ptr + idx * 3;
                        scalar_t gg_r = gg_roughness_ptr[idx * gg_roughness_stride];
                        const scalar_t* gg_f = gg_f0_ptr + idx * gg_f0_stride * 3;
                        const scalar_t* g = grad_ptr + idx * 3;

                        scalar_t* ggo = grad_grad_output_ptr + idx * 3;
                        scalar_t* g2n = grad2_normal_ptr + idx * 3;
                        scalar_t* g2v = grad2_view_ptr + idx * 3;
                        scalar_t* g2l = grad2_light_ptr + idx * 3;
                        scalar_t* g2r = grad2_roughness_ptr + idx;
                        scalar_t* g2f = grad2_f0_ptr + idx * 3;

                        // Process each channel
                        for (int c = 0; c < 3; ++c) {
                            scalar_t temp_ggo, temp_g2n[3], temp_g2v[3], temp_g2l[3], temp_g2r, temp_g2f;

                            cook_torrance_backward_backward_scalar<scalar_t>(
                                gg_n[0], gg_n[1], gg_n[2],
                                gg_v[0], gg_v[1], gg_v[2],
                                gg_l[0], gg_l[1], gg_l[2],
                                gg_r,
                                gg_f[c],
                                g[c],
                                n, v, l, r, f[c],
                                &temp_ggo, temp_g2n, temp_g2v, temp_g2l, &temp_g2r, &temp_g2f
                            );

                            ggo[c] += temp_ggo;
                            for (int i = 0; i < 3; ++i) {
                                g2n[i] += temp_g2n[i];
                                g2v[i] += temp_g2v[i];
                                g2l[i] += temp_g2l[i];
                            }
                            *g2r += temp_g2r;
                            g2f[c] = temp_g2f;
                        }
                    }
                });
            } else {
                at::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
                    for (int64_t idx = begin; idx < end; ++idx) {
                        const scalar_t* n = normal_ptr + idx * 3;
                        const scalar_t* v = view_ptr + idx * 3;
                        const scalar_t* l = light_ptr + idx * 3;
                        scalar_t r = roughness_ptr[idx * roughness_stride];
                        scalar_t f = f0_ptr[idx * f0_stride];

                        const scalar_t* gg_n = gg_normal_ptr + idx * 3;
                        const scalar_t* gg_v = gg_view_ptr + idx * 3;
                        const scalar_t* gg_l = gg_light_ptr + idx * 3;
                        scalar_t gg_r = gg_roughness_ptr[idx * gg_roughness_stride];
                        scalar_t gg_f = gg_f0_ptr[idx * gg_f0_stride];
                        scalar_t g = grad_ptr[idx];

                        cook_torrance_backward_backward_scalar<scalar_t>(
                            gg_n[0], gg_n[1], gg_n[2],
                            gg_v[0], gg_v[1], gg_v[2],
                            gg_l[0], gg_l[1], gg_l[2],
                            gg_r,
                            gg_f,
                            g,
                            n, v, l, r, f,
                            grad_grad_output_ptr + idx,
                            grad2_normal_ptr + idx * 3,
                            grad2_view_ptr + idx * 3,
                            grad2_light_ptr + idx * 3,
                            grad2_roughness_ptr + idx,
                            grad2_f0_ptr + idx
                        );
                    }
                });
            }
        }
    );

    // Reshape outputs back to original shapes
    return std::make_tuple(
        f0_is_rgb ? grad_grad_output.view(vec_shape) : grad_grad_output.view(batch_shape),
        grad2_normal.view(vec_shape),
        grad2_view.view(vec_shape),
        grad2_light.view(vec_shape),
        grad2_roughness.view(batch_shape),
        f0_is_rgb ? grad2_f0.view(vec_shape) : grad2_f0.view(batch_shape)
    );
}

}  // namespace torchscience::cpu::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("cook_torrance", &torchscience::cpu::graphics::shading::cook_torrance);
    module.impl("cook_torrance_backward", &torchscience::cpu::graphics::shading::cook_torrance_backward);
    module.impl("cook_torrance_backward_backward", &torchscience::cpu::graphics::shading::cook_torrance_backward_backward);
}
