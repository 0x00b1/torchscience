#pragma once

#include <cmath>

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace torchscience::kernel::geometry::intersection {

/**
 * Backward pass for ray-sphere intersection.
 *
 * Uses implicit differentiation of the quadratic at^2 + 2*half_b*t + c = 0.
 *
 * From the quadratic formula, differentiating implicitly:
 *   (2at + 2*half_b)*dt + t^2*da + 2t*d_half_b + dc = 0
 *   dt = -(t^2*da + 2t*d_half_b + dc) / (2*(at + half_b))
 *
 * Partial derivatives:
 *   da/dD = 2*D
 *   d_half_b/dO = D
 *   d_half_b/dD = (O - C)
 *   d_half_b/dC = -D
 *   dc/dO = 2*(O - C)
 *   dc/dC = -2*(O - C)
 *   dc/dr = -2*r
 *
 * @param grad_t Upstream gradient for intersection parameter t
 * @param grad_hit_x Upstream gradient for hit point x coordinate
 * @param grad_hit_y Upstream gradient for hit point y coordinate
 * @param grad_hit_z Upstream gradient for hit point z coordinate
 * @param grad_normal_x Upstream gradient for surface normal x coordinate
 * @param grad_normal_y Upstream gradient for surface normal y coordinate
 * @param grad_normal_z Upstream gradient for surface normal z coordinate
 * @param grad_uv_u Upstream gradient for UV coordinate u
 * @param grad_uv_v Upstream gradient for UV coordinate v
 * @param ray_origin_x Ray origin x coordinate (saved from forward)
 * @param ray_origin_y Ray origin y coordinate (saved from forward)
 * @param ray_origin_z Ray origin z coordinate (saved from forward)
 * @param ray_dir_x Ray direction x coordinate (saved from forward)
 * @param ray_dir_y Ray direction y coordinate (saved from forward)
 * @param ray_dir_z Ray direction z coordinate (saved from forward)
 * @param center_x Sphere center x coordinate (saved from forward)
 * @param center_y Sphere center y coordinate (saved from forward)
 * @param center_z Sphere center z coordinate (saved from forward)
 * @param radius Sphere radius (saved from forward)
 * @param t Intersection parameter t (saved from forward)
 * @param hit Hit flag (saved from forward)
 * @param grad_origin_x Output gradient for ray origin x
 * @param grad_origin_y Output gradient for ray origin y
 * @param grad_origin_z Output gradient for ray origin z
 * @param grad_dir_x Output gradient for ray direction x
 * @param grad_dir_y Output gradient for ray direction y
 * @param grad_dir_z Output gradient for ray direction z
 * @param grad_center_x Output gradient for sphere center x
 * @param grad_center_y Output gradient for sphere center y
 * @param grad_center_z Output gradient for sphere center z
 * @param grad_radius Output gradient for sphere radius
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void ray_sphere_backward_kernel(
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
    T center_x,
    T center_y,
    T center_z,
    T radius,
    T t,
    T hit,
    T& grad_origin_x,
    T& grad_origin_y,
    T& grad_origin_z,
    T& grad_dir_x,
    T& grad_dir_y,
    T& grad_dir_z,
    T& grad_center_x,
    T& grad_center_y,
    T& grad_center_z,
    T& grad_radius) {
  // If no hit, all gradients are zero
  if (hit == T(0)) {
    grad_origin_x = T(0);
    grad_origin_y = T(0);
    grad_origin_z = T(0);
    grad_dir_x = T(0);
    grad_dir_y = T(0);
    grad_dir_z = T(0);
    grad_center_x = T(0);
    grad_center_y = T(0);
    grad_center_z = T(0);
    grad_radius = T(0);
    return;
  }

  const T eps = T(1e-8);

  // Recompute forward values
  T oc_x = ray_origin_x - center_x;
  T oc_y = ray_origin_y - center_y;
  T oc_z = ray_origin_z - center_z;

  T a = ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y + ray_dir_z * ray_dir_z;
  T half_b = oc_x * ray_dir_x + oc_y * ray_dir_y + oc_z * ray_dir_z;

  // Denominator for dt: 2*(at + half_b)
  T denom = T(2) * (a * t + half_b) + eps;
  T inv_denom = T(1) / denom;

  // Gradient of t w.r.t. quadratic coefficients
  // dt/da = -t^2 / denom
  // dt/d_half_b = -2t / denom
  // dt/dc = -1 / denom
  T dt_da = -t * t * inv_denom;
  T dt_d_half_b = -T(2) * t * inv_denom;
  T dt_dc = -inv_denom;

  // Chain rule for quadratic coefficients:
  // da/dD = 2*D
  // d_half_b/dO = D, d_half_b/dD = oc, d_half_b/dC = -D
  // dc/dO = 2*oc, dc/dC = -2*oc, dc/dr = -2*r

  // Total gradient from t
  // hit_point = O + t*D, so grad from hit_point flows through t as well
  // Note: We ignore gradients through normal and UV outputs for simplicity.
  // Normal is derived from hit point (normal = (hit - center) / radius), and
  // UV is derived from spherical coordinates of normal (atan2, acos).
  // Full gradient propagation through these would require additional chain rule
  // steps through the trigonometric functions.
  T total_grad_t = grad_t;
  total_grad_t +=
      grad_hit_x * ray_dir_x + grad_hit_y * ray_dir_y + grad_hit_z * ray_dir_z;

  // Gradient w.r.t. ray origin (O)
  // dt/dO = dt/d_half_b * D + dt/dc * 2*oc
  T dt_do_x = dt_d_half_b * ray_dir_x + dt_dc * T(2) * oc_x;
  T dt_do_y = dt_d_half_b * ray_dir_y + dt_dc * T(2) * oc_y;
  T dt_do_z = dt_d_half_b * ray_dir_z + dt_dc * T(2) * oc_z;

  // Also direct contribution from hit_point = O + t*D: dhit/dO = I
  grad_origin_x = grad_hit_x + total_grad_t * dt_do_x;
  grad_origin_y = grad_hit_y + total_grad_t * dt_do_y;
  grad_origin_z = grad_hit_z + total_grad_t * dt_do_z;

  // Gradient w.r.t. ray direction (D)
  // dt/dD = dt/da * 2*D + dt/d_half_b * oc
  T dt_dd_x = dt_da * T(2) * ray_dir_x + dt_d_half_b * oc_x;
  T dt_dd_y = dt_da * T(2) * ray_dir_y + dt_d_half_b * oc_y;
  T dt_dd_z = dt_da * T(2) * ray_dir_z + dt_d_half_b * oc_z;

  // Also direct contribution from hit_point = O + t*D: dhit/dD = t*I
  grad_dir_x = t * grad_hit_x + total_grad_t * dt_dd_x;
  grad_dir_y = t * grad_hit_y + total_grad_t * dt_dd_y;
  grad_dir_z = t * grad_hit_z + total_grad_t * dt_dd_z;

  // Gradient w.r.t. center (C)
  // dt/dC = dt/d_half_b * (-D) + dt/dc * (-2*oc)
  T dt_dc_x = -dt_d_half_b * ray_dir_x - dt_dc * T(2) * oc_x;
  T dt_dc_y = -dt_d_half_b * ray_dir_y - dt_dc * T(2) * oc_y;
  T dt_dc_z = -dt_d_half_b * ray_dir_z - dt_dc * T(2) * oc_z;

  grad_center_x = total_grad_t * dt_dc_x;
  grad_center_y = total_grad_t * dt_dc_y;
  grad_center_z = total_grad_t * dt_dc_z;

  // Gradient w.r.t. radius
  // dt/dr = dt/dc * (-2*r)
  grad_radius = total_grad_t * dt_dc * (-T(2) * radius);
}

}  // namespace torchscience::kernel::geometry::intersection
