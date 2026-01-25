#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace torchscience::kernel::geometry::intersection {

/**
 * Backward pass for ray-triangle intersection.
 *
 * Uses the adjoint method applied to the Moller-Trumbore linear system.
 *
 * The forward pass solves M * [t, u, v]^T = s where:
 *   M = [-D | edge1 | edge2], s = O - V0
 *
 * Given upstream gradients g = [total_grad_t, grad_uv_u, grad_uv_v]^T,
 * the adjoint vector w = M^{-T} * g provides the key quantity for
 * computing all input gradients.
 *
 * Using the Moller-Trumbore quantities, the adjoint simplifies to:
 *   w = f * (total_grad_t * n + D x g1)
 * where:
 *   n = edge1 x edge2 (unnormalized face normal)
 *   g1 = grad_uv_u * edge2 - grad_uv_v * edge1
 *   f = 1/a, a = edge1 . (D x edge2)
 *
 * Then the input gradients are:
 *   grad_O = w + grad_hit_point
 *   grad_D = t * (w + grad_hit_point)
 *   grad_V0 = -(1 - u - v) * w
 *   grad_V1 = -u * w
 *   grad_V2 = -v * w
 *
 * Note: grad_normal is intentionally not propagated. The normalized surface
 * normal depends on edge1 x edge2 through a normalize() operation, which
 * would require additional chain rule through the normalization. This is
 * left as future work (see backward_backward kernel for second-order).
 *
 * @param grad_t Upstream gradient for intersection parameter t
 * @param grad_hit_x Upstream gradient for hit point x coordinate
 * @param grad_hit_y Upstream gradient for hit point y coordinate
 * @param grad_hit_z Upstream gradient for hit point z coordinate
 * @param grad_normal_x Upstream gradient for surface normal x coordinate
 * @param grad_normal_y Upstream gradient for surface normal y coordinate
 * @param grad_normal_z Upstream gradient for surface normal z coordinate
 * @param grad_uv_u Upstream gradient for barycentric coordinate u
 * @param grad_uv_v Upstream gradient for barycentric coordinate v
 * @param ray_origin_x Ray origin x coordinate (saved from forward)
 * @param ray_origin_y Ray origin y coordinate (saved from forward)
 * @param ray_origin_z Ray origin z coordinate (saved from forward)
 * @param ray_dir_x Ray direction x coordinate (saved from forward)
 * @param ray_dir_y Ray direction y coordinate (saved from forward)
 * @param ray_dir_z Ray direction z coordinate (saved from forward)
 * @param v0_x Triangle vertex V0 x coordinate (saved from forward)
 * @param v0_y Triangle vertex V0 y coordinate (saved from forward)
 * @param v0_z Triangle vertex V0 z coordinate (saved from forward)
 * @param v1_x Triangle vertex V1 x coordinate (saved from forward)
 * @param v1_y Triangle vertex V1 y coordinate (saved from forward)
 * @param v1_z Triangle vertex V1 z coordinate (saved from forward)
 * @param v2_x Triangle vertex V2 x coordinate (saved from forward)
 * @param v2_y Triangle vertex V2 y coordinate (saved from forward)
 * @param v2_z Triangle vertex V2 z coordinate (saved from forward)
 * @param t Intersection parameter t (saved from forward)
 * @param hit Hit flag (saved from forward)
 * @param grad_origin_x Output gradient for ray origin x
 * @param grad_origin_y Output gradient for ray origin y
 * @param grad_origin_z Output gradient for ray origin z
 * @param grad_dir_x Output gradient for ray direction x
 * @param grad_dir_y Output gradient for ray direction y
 * @param grad_dir_z Output gradient for ray direction z
 * @param grad_v0_x Output gradient for triangle vertex V0 x
 * @param grad_v0_y Output gradient for triangle vertex V0 y
 * @param grad_v0_z Output gradient for triangle vertex V0 z
 * @param grad_v1_x Output gradient for triangle vertex V1 x
 * @param grad_v1_y Output gradient for triangle vertex V1 y
 * @param grad_v1_z Output gradient for triangle vertex V1 z
 * @param grad_v2_x Output gradient for triangle vertex V2 x
 * @param grad_v2_y Output gradient for triangle vertex V2 y
 * @param grad_v2_z Output gradient for triangle vertex V2 z
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void ray_triangle_backward_kernel(
    T grad_t,
    T grad_hit_x,
    T grad_hit_y,
    T grad_hit_z,
    T grad_normal_x,
    T grad_normal_y,
    T grad_normal_z,
    T grad_uv_u,
    T grad_uv_v,
    T ray_origin_x,
    T ray_origin_y,
    T ray_origin_z,
    T ray_dir_x,
    T ray_dir_y,
    T ray_dir_z,
    T v0_x,
    T v0_y,
    T v0_z,
    T v1_x,
    T v1_y,
    T v1_z,
    T v2_x,
    T v2_y,
    T v2_z,
    T t,
    T hit,
    T& grad_origin_x,
    T& grad_origin_y,
    T& grad_origin_z,
    T& grad_dir_x,
    T& grad_dir_y,
    T& grad_dir_z,
    T& grad_v0_x,
    T& grad_v0_y,
    T& grad_v0_z,
    T& grad_v1_x,
    T& grad_v1_y,
    T& grad_v1_z,
    T& grad_v2_x,
    T& grad_v2_y,
    T& grad_v2_z) {
  // If no hit, all gradients are zero
  if (hit == T(0)) {
    grad_origin_x = T(0);
    grad_origin_y = T(0);
    grad_origin_z = T(0);
    grad_dir_x = T(0);
    grad_dir_y = T(0);
    grad_dir_z = T(0);
    grad_v0_x = T(0);
    grad_v0_y = T(0);
    grad_v0_z = T(0);
    grad_v1_x = T(0);
    grad_v1_y = T(0);
    grad_v1_z = T(0);
    grad_v2_x = T(0);
    grad_v2_y = T(0);
    grad_v2_z = T(0);
    return;
  }

  // Recompute forward quantities from saved inputs
  // Edge vectors
  T edge1_x = v1_x - v0_x;
  T edge1_y = v1_y - v0_y;
  T edge1_z = v1_z - v0_z;

  T edge2_x = v2_x - v0_x;
  T edge2_y = v2_y - v0_y;
  T edge2_z = v2_z - v0_z;

  // h = D x edge2
  T h_x = ray_dir_y * edge2_z - ray_dir_z * edge2_y;
  T h_y = ray_dir_z * edge2_x - ray_dir_x * edge2_z;
  T h_z = ray_dir_x * edge2_y - ray_dir_y * edge2_x;

  // a = edge1 . h (determinant)
  T a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;

  // f = 1/a
  T f = T(1) / a;

  // s = O - V0
  T s_x = ray_origin_x - v0_x;
  T s_y = ray_origin_y - v0_y;
  T s_z = ray_origin_z - v0_z;

  // Recompute u = f * (s . h)
  T u = f * (s_x * h_x + s_y * h_y + s_z * h_z);

  // q = s x edge1
  T q_x = s_y * edge1_z - s_z * edge1_y;
  T q_y = s_z * edge1_x - s_x * edge1_z;
  T q_z = s_x * edge1_y - s_y * edge1_x;

  // Recompute v = f * (D . q)
  T v = f * (ray_dir_x * q_x + ray_dir_y * q_y + ray_dir_z * q_z);

  // n = edge1 x edge2 (unnormalized face normal)
  T n_x = edge1_y * edge2_z - edge1_z * edge2_y;
  T n_y = edge1_z * edge2_x - edge1_x * edge2_z;
  T n_z = edge1_x * edge2_y - edge1_y * edge2_x;

  // Compute total_grad_t: gradient through t plus gradient through hit_point
  // hit_point = O + t * D, so d(hit_point)/dt = D
  T total_grad_t = grad_t + grad_hit_x * ray_dir_x + grad_hit_y * ray_dir_y +
      grad_hit_z * ray_dir_z;

  // Compute g1 = grad_uv_u * edge2 - grad_uv_v * edge1
  T g1_x = grad_uv_u * edge2_x - grad_uv_v * edge1_x;
  T g1_y = grad_uv_u * edge2_y - grad_uv_v * edge1_y;
  T g1_z = grad_uv_u * edge2_z - grad_uv_v * edge1_z;

  // Compute D x g1 (cross product)
  T dxg1_x = ray_dir_y * g1_z - ray_dir_z * g1_y;
  T dxg1_y = ray_dir_z * g1_x - ray_dir_x * g1_z;
  T dxg1_z = ray_dir_x * g1_y - ray_dir_y * g1_x;

  // Adjoint vector: w = f * (total_grad_t * n + D x g1)
  T w_x = f * (total_grad_t * n_x + dxg1_x);
  T w_y = f * (total_grad_t * n_y + dxg1_y);
  T w_z = f * (total_grad_t * n_z + dxg1_z);

  // Gradient w.r.t. ray origin (O)
  // From the linear system: grad_O = w (adjoint contribution)
  // From hit_point = O + t*D: grad_O += grad_hit_point (direct contribution)
  grad_origin_x = w_x + grad_hit_x;
  grad_origin_y = w_y + grad_hit_y;
  grad_origin_z = w_z + grad_hit_z;

  // Gradient w.r.t. ray direction (D)
  // From the linear system: column of M corresponding to D is -D, so
  //   grad_D = -(-t) * w = t * w (from d(M*x)/dD contribution)
  // From hit_point = O + t*D: grad_D += t * grad_hit_point
  grad_dir_x = t * (w_x + grad_hit_x);
  grad_dir_y = t * (w_y + grad_hit_y);
  grad_dir_z = t * (w_z + grad_hit_z);

  // Gradient w.r.t. triangle vertices using adjoint method for M*x = s.
  //
  // For a linear system M*x = s with loss L and upstream gradient g = dL/dx,
  // the adjoint vector w = M^{-T} g gives:
  //   dL = w^T ds - w^T (dM) x
  //
  // V0 appears in the RHS s = O - V0 and in matrix columns via
  // edge1 = V1-V0 (column 2) and edge2 = V2-V0 (column 3):
  //   From RHS (ds/dV0 = -I):        -w
  //   From edge1 (-w^T * (-dV0)*u):  +u*w
  //   From edge2 (-w^T * (-dV0)*v):  +v*w
  //   Total: grad_V0 = -(1 - u - v) * w
  //
  // V1 only appears through edge1 = V1 - V0 (column 2 of M):
  //   -w^T * (dV1)*u = -u*w
  //   Total: grad_V1 = -u * w
  //
  // V2 only appears through edge2 = V2 - V0 (column 3 of M):
  //   -w^T * (dV2)*v = -v*w
  //   Total: grad_V2 = -v * w
  //
  // Conservation check: grad_V0 + grad_V1 + grad_V2 = -w = -grad_O_from_system.
  // A uniform translation of the triangle is equivalent to shifting the origin
  // in the opposite direction, so the gradients must sum to -w. Verified:
  // -(1-u-v)*w + (-u*w) + (-v*w) = -(1-u-v+u+v)*w = -w.

  grad_v0_x = -(T(1) - u - v) * w_x;
  grad_v0_y = -(T(1) - u - v) * w_y;
  grad_v0_z = -(T(1) - u - v) * w_z;

  grad_v1_x = -u * w_x;
  grad_v1_y = -u * w_y;
  grad_v1_z = -u * w_z;

  grad_v2_x = -v * w_x;
  grad_v2_y = -v * w_y;
  grad_v2_z = -v * w_z;

  // Note: grad_normal (upstream gradient for the normalized surface normal)
  // is intentionally not propagated. Differentiating through the
  // normalization of edge1 x edge2 requires the Jacobian of the
  // normalize() operation, which is deferred to a future implementation.
}

}  // namespace torchscience::kernel::geometry::intersection
