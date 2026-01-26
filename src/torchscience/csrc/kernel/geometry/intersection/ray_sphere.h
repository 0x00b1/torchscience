#pragma once

#include <cmath>
#include <limits>

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace torchscience::kernel::geometry::intersection {

/**
 * Ray-sphere intersection kernel.
 *
 * Solves the quadratic equation arising from substituting
 * the ray equation P(t) = O + t*D into the sphere equation |P - C|^2 = r^2.
 *
 * Quadratic: a*t^2 + b*t + c = 0
 * where:
 *   a = D . D
 *   half_b = D . (O - C)  (using half_b optimization)
 *   c = (O - C) . (O - C) - r^2
 *   discriminant = half_b^2 - a*c
 *   t = (-half_b - sqrt(discriminant)) / a  (near hit)
 *
 * @param ray_origin_x Ray origin x coordinate
 * @param ray_origin_y Ray origin y coordinate
 * @param ray_origin_z Ray origin z coordinate
 * @param ray_dir_x Ray direction x coordinate
 * @param ray_dir_y Ray direction y coordinate
 * @param ray_dir_z Ray direction z coordinate
 * @param center_x Sphere center x coordinate
 * @param center_y Sphere center y coordinate
 * @param center_z Sphere center z coordinate
 * @param radius Sphere radius
 * @param out_t Output intersection parameter t
 * @param out_hit_x Output hit point x coordinate
 * @param out_hit_y Output hit point y coordinate
 * @param out_hit_z Output hit point z coordinate
 * @param out_normal_x Output surface normal x coordinate
 * @param out_normal_y Output surface normal y coordinate
 * @param out_normal_z Output surface normal z coordinate
 * @param out_uv_u Output UV coordinate u (theta / 2*pi)
 * @param out_uv_v Output UV coordinate v (phi / pi)
 * @param out_hit Output hit flag (1 if hit, 0 otherwise)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void ray_sphere_kernel(
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
    T& out_t,
    T& out_hit_x,
    T& out_hit_y,
    T& out_hit_z,
    T& out_normal_x,
    T& out_normal_y,
    T& out_normal_z,
    T& out_uv_u,
    T& out_uv_v,
    T& out_hit) {
  const T eps = T(1e-8);
  const T inf = std::numeric_limits<T>::infinity();
  const T pi = T(3.14159265358979323846);

  // Vector from center to origin: oc = O - C
  T oc_x = ray_origin_x - center_x;
  T oc_y = ray_origin_y - center_y;
  T oc_z = ray_origin_z - center_z;

  // Quadratic coefficients (using half_b optimization)
  T a = ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y + ray_dir_z * ray_dir_z;
  T half_b = oc_x * ray_dir_x + oc_y * ray_dir_y + oc_z * ray_dir_z;
  T c = oc_x * oc_x + oc_y * oc_y + oc_z * oc_z - radius * radius;

  // Discriminant
  T discriminant = half_b * half_b - a * c;

  // Check for miss (discriminant < 0)
  if (discriminant < T(0)) {
    out_t = inf;
    out_hit_x = T(0);
    out_hit_y = T(0);
    out_hit_z = T(0);
    out_normal_x = T(0);
    out_normal_y = T(0);
    out_normal_z = T(0);
    out_uv_u = T(0);
    out_uv_v = T(0);
    out_hit = T(0);
    return;
  }

  // Compute t (prefer near hit)
  T sqrt_d = std::sqrt(discriminant);
  T t1 = (-half_b - sqrt_d) / a;  // near
  T t2 = (-half_b + sqrt_d) / a;  // far

  // Select valid t (>= 0)
  T t;
  if (t1 >= T(0)) {
    t = t1;
  } else if (t2 >= T(0)) {
    t = t2;
  } else {
    // Both behind ray
    out_t = inf;
    out_hit_x = T(0);
    out_hit_y = T(0);
    out_hit_z = T(0);
    out_normal_x = T(0);
    out_normal_y = T(0);
    out_normal_z = T(0);
    out_uv_u = T(0);
    out_uv_v = T(0);
    out_hit = T(0);
    return;
  }

  // Hit point
  T hit_x = ray_origin_x + t * ray_dir_x;
  T hit_y = ray_origin_y + t * ray_dir_y;
  T hit_z = ray_origin_z + t * ray_dir_z;

  // Outward normal: (P - C) / r
  T inv_r = T(1) / (radius + eps);
  T normal_x = (hit_x - center_x) * inv_r;
  T normal_y = (hit_y - center_y) * inv_r;
  T normal_z = (hit_z - center_z) * inv_r;

  // Check if ray hits front (outside) or back (inside) of sphere
  // Front face: direction opposes normal (D . N < 0)
  T d_dot_n =
      ray_dir_x * normal_x + ray_dir_y * normal_y + ray_dir_z * normal_z;
  if (d_dot_n > T(0)) {
    // Flip normal to face ray
    normal_x = -normal_x;
    normal_y = -normal_y;
    normal_z = -normal_z;
  }

  // UV: spherical coordinates normalized to [0, 1]
  // theta = atan2(y, x), phi = acos(z/r)
  // Note: use outward normal (before potential flip) for consistent UVs
  T outward_normal_x = (hit_x - center_x) * inv_r;
  T outward_normal_y = (hit_y - center_y) * inv_r;
  T outward_normal_z = (hit_z - center_z) * inv_r;
  T theta = std::atan2(outward_normal_y, outward_normal_x);  // [-pi, pi]
  T phi = std::acos(outward_normal_z);                       // [0, pi]
  T u = (theta + pi) / (T(2) * pi);                          // [0, 1]
  T v = phi / pi;                                            // [0, 1]

  out_t = t;
  out_hit_x = hit_x;
  out_hit_y = hit_y;
  out_hit_z = hit_z;
  out_normal_x = normal_x;
  out_normal_y = normal_y;
  out_normal_z = normal_z;
  out_uv_u = u;
  out_uv_v = v;
  out_hit = T(1);
}

}  // namespace torchscience::kernel::geometry::intersection
