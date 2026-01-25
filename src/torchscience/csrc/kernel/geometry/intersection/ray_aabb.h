#pragma once

#include <cmath>
#include <limits>

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

namespace torchscience::kernel::geometry::intersection {

/**
 * Ray-AABB intersection kernel using the slab method.
 *
 * An axis-aligned bounding box (AABB) is defined by its minimum and maximum
 * corners. The slab method intersects the ray with each pair of parallel
 * planes (slabs) and computes the overlap of all three intervals.
 *
 * For each axis i:
 *   inv_d_i = 1 / direction_i
 *   t1_i = (box_min_i - origin_i) * inv_d_i
 *   t2_i = (box_max_i - origin_i) * inv_d_i
 *   if inv_d_i < 0: swap(t1_i, t2_i)
 *   t_near = max(t_near, t1_i)
 *   t_far = min(t_far, t2_i)
 *
 * Miss if t_near > t_far or t_far < 0.
 *
 * @param ray_origin_x Ray origin x coordinate
 * @param ray_origin_y Ray origin y coordinate
 * @param ray_origin_z Ray origin z coordinate
 * @param ray_dir_x Ray direction x coordinate
 * @param ray_dir_y Ray direction y coordinate
 * @param ray_dir_z Ray direction z coordinate
 * @param box_min_x AABB minimum corner x coordinate
 * @param box_min_y AABB minimum corner y coordinate
 * @param box_min_z AABB minimum corner z coordinate
 * @param box_max_x AABB maximum corner x coordinate
 * @param box_max_y AABB maximum corner y coordinate
 * @param box_max_z AABB maximum corner z coordinate
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
C10_HOST_DEVICE C10_ALWAYS_INLINE void ray_aabb_kernel(
    T ray_origin_x,
    T ray_origin_y,
    T ray_origin_z,
    T ray_dir_x,
    T ray_dir_y,
    T ray_dir_z,
    T box_min_x,
    T box_min_y,
    T box_min_z,
    T box_max_x,
    T box_max_y,
    T box_max_z,
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
  const T inf = std::numeric_limits<T>::infinity();
  const T eps = T(1e-8);

  // Store direction and box bounds in arrays for axis iteration
  T origin[3] = {ray_origin_x, ray_origin_y, ray_origin_z};
  T dir[3] = {ray_dir_x, ray_dir_y, ray_dir_z};
  T bmin[3] = {box_min_x, box_min_y, box_min_z};
  T bmax[3] = {box_max_x, box_max_y, box_max_z};

  T t_near = -inf;
  T t_far = inf;

  // Track which axis and which face (min or max) determined t_near
  int near_axis = 0;
  int near_face = 0;  // 0 = min face, 1 = max face

  for (int i = 0; i < 3; ++i) {
    T inv_d = T(1) / (dir[i] + (dir[i] == T(0) ? eps : T(0)));

    T t1 = (bmin[i] - origin[i]) * inv_d;
    T t2 = (bmax[i] - origin[i]) * inv_d;

    // Track whether we swap (determines which face is near)
    int face = 0;  // t1 corresponds to min face
    if (inv_d < T(0)) {
      // Swap t1 and t2
      T tmp = t1;
      t1 = t2;
      t2 = tmp;
      face = 1;  // After swap, t1 corresponds to max face
    }

    if (t1 > t_near) {
      t_near = t1;
      near_axis = i;
      // If we swapped, near slab came from max face; otherwise min face
      near_face = (inv_d < T(0)) ? 1 : 0;
    }

    if (t2 < t_far) {
      t_far = t2;
    }
  }

  // Check for miss
  if (t_near > t_far || t_far < T(0)) {
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

  // Select t: if t_near >= 0, use it; otherwise ray starts inside box, use
  // t_far
  T t;
  int hit_axis;
  int hit_face;

  if (t_near >= T(0)) {
    t = t_near;
    hit_axis = near_axis;
    hit_face = near_face;
  } else {
    // Ray origin is inside the box, use t_far as exit point.
    // Recompute which axis/face corresponds to t_far.
    t = t_far;
    hit_axis = 0;
    hit_face = 0;

    // Determine which axis produced t_far
    for (int i = 0; i < 3; ++i) {
      T inv_d = T(1) / (dir[i] + (dir[i] == T(0) ? eps : T(0)));

      T t1 = (bmin[i] - origin[i]) * inv_d;
      T t2 = (bmax[i] - origin[i]) * inv_d;

      if (inv_d < T(0)) {
        T tmp = t1;
        t1 = t2;
        t2 = tmp;
      }

      // t_far was determined by min of all t2 values
      T diff = t2 - t_far;
      if (diff < T(0))
        diff = -diff;
      if (diff < eps) {
        hit_axis = i;
        // t2 corresponds to max face if not swapped, min face if swapped
        hit_face = (inv_d < T(0)) ? 0 : 1;
        break;
      }
    }
  }

  // Hit point
  T hit_x = ray_origin_x + t * ray_dir_x;
  T hit_y = ray_origin_y + t * ray_dir_y;
  T hit_z = ray_origin_z + t * ray_dir_z;

  // Normal: axis-aligned face normal
  T normal_x = T(0);
  T normal_y = T(0);
  T normal_z = T(0);

  // The normal points outward from the box face that was hit.
  // hit_face == 0 means min face (normal points in negative axis direction)
  // hit_face == 1 means max face (normal points in positive axis direction)
  T sign = (hit_face == 0) ? T(-1) : T(1);

  if (hit_axis == 0) {
    normal_x = sign;
  } else if (hit_axis == 1) {
    normal_y = sign;
  } else {
    normal_z = sign;
  }

  // UV: parametric coordinates on the hit face
  // Use the two non-normal axes, mapped to [0,1] within the box extents.
  int u_axis = (hit_axis + 1) % 3;
  int v_axis = (hit_axis + 2) % 3;

  T hit_coords[3] = {hit_x, hit_y, hit_z};

  T u_range = bmax[u_axis] - bmin[u_axis];
  T v_range = bmax[v_axis] - bmin[v_axis];

  T u = (u_range > eps) ? (hit_coords[u_axis] - bmin[u_axis]) / u_range : T(0);
  T v = (v_range > eps) ? (hit_coords[v_axis] - bmin[v_axis]) / v_range : T(0);

  // Clamp to [0, 1]
  if (u < T(0))
    u = T(0);
  if (u > T(1))
    u = T(1);
  if (v < T(0))
    v = T(0);
  if (v > T(1))
    v = T(1);

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
