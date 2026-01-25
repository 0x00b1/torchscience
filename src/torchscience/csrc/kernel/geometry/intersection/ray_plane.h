#pragma once

#include <cmath>
#include <limits>

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace torchscience::kernel::geometry::intersection {

/**
 * Ray-plane intersection kernel.
 *
 * Plane equation: n . x = d (where n is normal, d is offset)
 * Ray equation: p = o + t * dir
 *
 * Solving: n . (o + t * dir) = d
 *          t = (d - n . o) / (n . dir)
 *
 * @param ray_origin_x Ray origin x coordinate
 * @param ray_origin_y Ray origin y coordinate
 * @param ray_origin_z Ray origin z coordinate
 * @param ray_dir_x Ray direction x coordinate
 * @param ray_dir_y Ray direction y coordinate
 * @param ray_dir_z Ray direction z coordinate
 * @param plane_normal_x Plane normal x coordinate
 * @param plane_normal_y Plane normal y coordinate
 * @param plane_normal_z Plane normal z coordinate
 * @param plane_offset Plane offset (d in n . x = d)
 * @param out_t Output intersection parameter t
 * @param out_hit_x Output hit point x coordinate
 * @param out_hit_y Output hit point y coordinate
 * @param out_hit_z Output hit point z coordinate
 * @param out_normal_x Output surface normal x coordinate
 * @param out_normal_y Output surface normal y coordinate
 * @param out_normal_z Output surface normal z coordinate
 * @param out_uv_u Output UV coordinate u
 * @param out_uv_v Output UV coordinate v
 * @param out_hit Output hit flag (1 if hit, 0 otherwise)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE void ray_plane_kernel(
    T ray_origin_x,
    T ray_origin_y,
    T ray_origin_z,
    T ray_dir_x,
    T ray_dir_y,
    T ray_dir_z,
    T plane_normal_x,
    T plane_normal_y,
    T plane_normal_z,
    T plane_offset,
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
  const T t_min = T(0);
  const T t_max = inf;

  // Compute n . dir
  T n_dot_dir = plane_normal_x * ray_dir_x + plane_normal_y * ray_dir_y +
                plane_normal_z * ray_dir_z;

  // Check if ray is parallel to plane
  bool is_parallel = std::abs(n_dot_dir) < eps;

  if (is_parallel) {
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

  // Compute n . o
  T n_dot_origin = plane_normal_x * ray_origin_x +
                   plane_normal_y * ray_origin_y +
                   plane_normal_z * ray_origin_z;

  // t = (d - n . o) / (n . dir)
  T t = (plane_offset - n_dot_origin) / n_dot_dir;

  // Check t is in valid range
  if (t < t_min || t > t_max) {
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

  // Compute hit point
  T hit_x = ray_origin_x + t * ray_dir_x;
  T hit_y = ray_origin_y + t * ray_dir_y;
  T hit_z = ray_origin_z + t * ray_dir_z;

  // Normal is the plane normal (normalized)
  T normal_len = std::sqrt(plane_normal_x * plane_normal_x +
                           plane_normal_y * plane_normal_y +
                           plane_normal_z * plane_normal_z);
  T inv_len = T(1) / (normal_len + eps);
  T normal_x = plane_normal_x * inv_len;
  T normal_y = plane_normal_y * inv_len;
  T normal_z = plane_normal_z * inv_len;

  // Flip normal if ray is coming from behind
  if (n_dot_dir > 0) {
    normal_x = -normal_x;
    normal_y = -normal_y;
    normal_z = -normal_z;
  }

  // UV: project hit point onto plane's tangent frame
  // Use world x, y as UV (simple approximation)
  out_t = t;
  out_hit_x = hit_x;
  out_hit_y = hit_y;
  out_hit_z = hit_z;
  out_normal_x = normal_x;
  out_normal_y = normal_y;
  out_normal_z = normal_z;
  out_uv_u = hit_x;
  out_uv_v = hit_y;
  out_hit = T(1);
}

}  // namespace torchscience::kernel::geometry::intersection
