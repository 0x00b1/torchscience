#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace torchscience::kernel::geometry {

// ============================================================================
// 2D Convex Hull using Graham Scan
// ============================================================================

template <typename scalar_t>
struct ConvexHull2D {
  std::vector<int64_t> vertices;    // Vertex indices in CCW order
  std::vector<scalar_t> equations;  // Facet equations: (nx, ny, d) per edge
  scalar_t perimeter;               // Perimeter (area in N-1 dimension)
  scalar_t area;                    // Signed area

  // Cross product of vectors (p1-p0) and (p2-p0)
  static inline scalar_t cross(const scalar_t* pts, int64_t p0, int64_t p1,
                               int64_t p2) {
    scalar_t x0 = pts[p0 * 2], y0 = pts[p0 * 2 + 1];
    scalar_t x1 = pts[p1 * 2], y1 = pts[p1 * 2 + 1];
    scalar_t x2 = pts[p2 * 2], y2 = pts[p2 * 2 + 1];
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
  }

  void compute(const scalar_t* points, int64_t n) {
    if (n < 3) {
      vertices.clear();
      for (int64_t i = 0; i < n; ++i) vertices.push_back(i);
      perimeter = scalar_t(0);
      area = scalar_t(0);
      return;
    }

    // Find bottom-left point (pivot)
    int64_t pivot = 0;
    for (int64_t i = 1; i < n; ++i) {
      if (points[i * 2 + 1] < points[pivot * 2 + 1] ||
          (points[i * 2 + 1] == points[pivot * 2 + 1] &&
           points[i * 2] < points[pivot * 2])) {
        pivot = i;
      }
    }

    // Sort points by polar angle relative to pivot
    std::vector<int64_t> indices(n);
    for (int64_t i = 0; i < n; ++i) indices[i] = i;
    std::swap(indices[0], indices[pivot]);

    const scalar_t px = points[indices[0] * 2];
    const scalar_t py = points[indices[0] * 2 + 1];

    std::sort(indices.begin() + 1, indices.end(), [&](int64_t a, int64_t b) {
      scalar_t ax = points[a * 2] - px;
      scalar_t ay = points[a * 2 + 1] - py;
      scalar_t bx = points[b * 2] - px;
      scalar_t by = points[b * 2 + 1] - py;
      scalar_t cross_val = ax * by - ay * bx;
      if (std::abs(cross_val) < scalar_t(1e-10)) {
        // Collinear: sort by distance
        return (ax * ax + ay * ay) < (bx * bx + by * by);
      }
      return cross_val > 0;
    });

    // Graham scan
    vertices.clear();
    for (int64_t i = 0; i < n; ++i) {
      while (vertices.size() > 1 &&
             cross(points, vertices[vertices.size() - 2],
                   vertices[vertices.size() - 1], indices[i]) <= 0) {
        vertices.pop_back();
      }
      vertices.push_back(indices[i]);
    }

    // Compute area and perimeter
    int64_t m = static_cast<int64_t>(vertices.size());
    area = scalar_t(0);
    perimeter = scalar_t(0);
    equations.resize(m * 3);

    for (int64_t i = 0; i < m; ++i) {
      int64_t j = (i + 1) % m;
      int64_t vi = vertices[i];
      int64_t vj = vertices[j];

      scalar_t x0 = points[vi * 2], y0 = points[vi * 2 + 1];
      scalar_t x1 = points[vj * 2], y1 = points[vj * 2 + 1];

      // Shoelace formula
      area += x0 * y1 - x1 * y0;

      // Edge length
      scalar_t dx = x1 - x0, dy = y1 - y0;
      scalar_t len = std::sqrt(dx * dx + dy * dy);
      perimeter += len;

      // Edge equation: outward normal and offset
      if (len > scalar_t(1e-10)) {
        scalar_t nx = dy / len;  // Outward normal (perpendicular to edge)
        scalar_t ny = -dx / len;
        scalar_t d = -(nx * x0 + ny * y0);
        equations[i * 3 + 0] = nx;
        equations[i * 3 + 1] = ny;
        equations[i * 3 + 2] = d;
      } else {
        equations[i * 3 + 0] = scalar_t(0);
        equations[i * 3 + 1] = scalar_t(0);
        equations[i * 3 + 2] = scalar_t(0);
      }
    }
    area = std::abs(area) / scalar_t(2);
  }
};

// ============================================================================
// 3D Convex Hull using Quickhull
// ============================================================================

template <typename scalar_t>
struct ConvexHull3D {
  std::vector<int64_t> vertices;
  std::vector<int64_t> simplices;   // Triangular facets (3 vertices each)
  std::vector<int64_t> neighbors;   // Neighbor facet indices
  std::vector<scalar_t> equations;  // (nx, ny, nz, d) per facet
  scalar_t surface_area;
  scalar_t volume;

  // Facet structure for Quickhull
  struct Facet {
    int64_t v[3];           // Vertex indices (CCW when viewed from outside)
    scalar_t normal[3];     // Unit outward normal
    scalar_t offset;        // Plane offset (n.p + d = 0)
    std::vector<int64_t> outside_set;  // Points outside this facet
    bool active;

    void compute_plane(const scalar_t* pts) {
      scalar_t ax = pts[v[1] * 3 + 0] - pts[v[0] * 3 + 0];
      scalar_t ay = pts[v[1] * 3 + 1] - pts[v[0] * 3 + 1];
      scalar_t az = pts[v[1] * 3 + 2] - pts[v[0] * 3 + 2];
      scalar_t bx = pts[v[2] * 3 + 0] - pts[v[0] * 3 + 0];
      scalar_t by = pts[v[2] * 3 + 1] - pts[v[0] * 3 + 1];
      scalar_t bz = pts[v[2] * 3 + 2] - pts[v[0] * 3 + 2];

      normal[0] = ay * bz - az * by;
      normal[1] = az * bx - ax * bz;
      normal[2] = ax * by - ay * bx;

      scalar_t len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                               normal[2] * normal[2]);
      if (len > scalar_t(1e-12)) {
        normal[0] /= len;
        normal[1] /= len;
        normal[2] /= len;
      }

      offset = -(normal[0] * pts[v[0] * 3 + 0] + normal[1] * pts[v[0] * 3 + 1] +
                 normal[2] * pts[v[0] * 3 + 2]);
    }

    scalar_t signed_distance(const scalar_t* pt) const {
      return normal[0] * pt[0] + normal[1] * pt[1] + normal[2] * pt[2] + offset;
    }
  };

  void compute(const scalar_t* points, int64_t n) {
    if (n < 4) {
      vertices.clear();
      for (int64_t i = 0; i < n; ++i) vertices.push_back(i);
      surface_area = scalar_t(0);
      volume = scalar_t(0);
      return;
    }

    // Step 1: Find initial tetrahedron
    std::vector<int64_t> initial = find_initial_simplex(points, n);
    if (initial.size() < 4) {
      vertices.assign(initial.begin(), initial.end());
      surface_area = scalar_t(0);
      volume = scalar_t(0);
      return;
    }

    // Step 2: Create initial tetrahedron with 4 facets
    std::vector<Facet> facets;
    facets.reserve(n * 2);

    // Compute centroid for consistent outward orientation
    scalar_t cx = (points[initial[0] * 3] + points[initial[1] * 3] +
                   points[initial[2] * 3] + points[initial[3] * 3]) / 4;
    scalar_t cy = (points[initial[0] * 3 + 1] + points[initial[1] * 3 + 1] +
                   points[initial[2] * 3 + 1] + points[initial[3] * 3 + 1]) / 4;
    scalar_t cz = (points[initial[0] * 3 + 2] + points[initial[1] * 3 + 2] +
                   points[initial[2] * 3 + 2] + points[initial[3] * 3 + 2]) / 4;
    scalar_t centroid[3] = {cx, cy, cz};

    // 4 faces of tetrahedron
    int64_t tet_faces[4][3] = {
        {initial[0], initial[1], initial[2]},
        {initial[0], initial[2], initial[3]},
        {initial[0], initial[3], initial[1]},
        {initial[1], initial[3], initial[2]}};

    for (int i = 0; i < 4; ++i) {
      Facet f;
      f.v[0] = tet_faces[i][0];
      f.v[1] = tet_faces[i][1];
      f.v[2] = tet_faces[i][2];
      f.active = true;
      f.compute_plane(points);

      // Ensure normal points outward (away from centroid)
      if (f.signed_distance(centroid) > 0) {
        std::swap(f.v[1], f.v[2]);
        f.compute_plane(points);
      }
      facets.push_back(f);
    }

    // Step 3: Assign outside points to facets
    std::set<int64_t> hull_points(initial.begin(), initial.end());

    for (int64_t i = 0; i < n; ++i) {
      if (hull_points.count(i)) continue;

      scalar_t max_dist = scalar_t(0);
      int64_t best = -1;

      for (size_t f = 0; f < facets.size(); ++f) {
        if (!facets[f].active) continue;
        scalar_t d = facets[f].signed_distance(&points[i * 3]);
        if (d > max_dist) {
          max_dist = d;
          best = static_cast<int64_t>(f);
        }
      }

      if (best >= 0 && max_dist > scalar_t(1e-10)) {
        facets[best].outside_set.push_back(i);
      }
    }

    // Step 4: Iteratively expand hull
    bool progress = true;
    while (progress) {
      progress = false;

      // Find facet with non-empty outside set
      int64_t work_facet = -1;
      scalar_t max_dist = scalar_t(0);
      int64_t apex = -1;

      for (size_t f = 0; f < facets.size(); ++f) {
        if (!facets[f].active || facets[f].outside_set.empty()) continue;

        // Find furthest point in this facet's outside set
        for (int64_t pi : facets[f].outside_set) {
          scalar_t d = facets[f].signed_distance(&points[pi * 3]);
          if (d > max_dist) {
            max_dist = d;
            apex = pi;
            work_facet = static_cast<int64_t>(f);
          }
        }
      }

      if (work_facet < 0) break;  // No more work

      // Find ALL visible facets (direct check, no neighbor traversal)
      std::vector<int64_t> visible;
      for (size_t f = 0; f < facets.size(); ++f) {
        if (!facets[f].active) continue;
        if (facets[f].signed_distance(&points[apex * 3]) > scalar_t(1e-10)) {
          visible.push_back(static_cast<int64_t>(f));
        }
      }

      if (visible.empty()) continue;

      // Collect points from visible facets for redistribution
      std::vector<int64_t> orphan_points;
      for (int64_t vi : visible) {
        for (int64_t pi : facets[vi].outside_set) {
          if (pi != apex) orphan_points.push_back(pi);
        }
        facets[vi].outside_set.clear();
        facets[vi].active = false;
      }

      // Find horizon edges (edges of visible facets not shared with other visible)
      std::set<int64_t> visible_set(visible.begin(), visible.end());
      std::vector<std::pair<int64_t, int64_t>> horizon;

      for (int64_t vi : visible) {
        const Facet& vf = facets[vi];
        for (int e = 0; e < 3; ++e) {
          int64_t e1 = vf.v[e];
          int64_t e2 = vf.v[(e + 1) % 3];

          // Check if this edge is shared with another visible facet
          bool shared = false;
          for (int64_t vj : visible) {
            if (vj == vi) continue;
            const Facet& of = facets[vj];
            for (int oe = 0; oe < 3; ++oe) {
              int64_t oe1 = of.v[oe];
              int64_t oe2 = of.v[(oe + 1) % 3];
              // Edges match if they share same vertices (in either order)
              if ((e1 == oe1 && e2 == oe2) || (e1 == oe2 && e2 == oe1)) {
                shared = true;
                break;
              }
            }
            if (shared) break;
          }

          if (!shared) {
            // This edge is on the horizon - add in CCW order from outside
            horizon.emplace_back(e2, e1);  // Reverse to get correct winding
          }
        }
      }

      // Create new facets from apex to each horizon edge
      hull_points.insert(apex);
      size_t first_new = facets.size();

      for (const auto& edge : horizon) {
        Facet nf;
        nf.v[0] = apex;
        nf.v[1] = edge.first;
        nf.v[2] = edge.second;
        nf.active = true;
        nf.compute_plane(points);

        // Use centroid of current hull for orientation check
        scalar_t hcx = 0, hcy = 0, hcz = 0;
        for (int64_t hp : hull_points) {
          hcx += points[hp * 3];
          hcy += points[hp * 3 + 1];
          hcz += points[hp * 3 + 2];
        }
        hcx /= hull_points.size();
        hcy /= hull_points.size();
        hcz /= hull_points.size();
        scalar_t hc[3] = {hcx, hcy, hcz};

        if (nf.signed_distance(hc) > scalar_t(1e-10)) {
          std::swap(nf.v[1], nf.v[2]);
          nf.compute_plane(points);
        }

        facets.push_back(nf);
      }

      // Redistribute orphan points to new facets
      for (int64_t pi : orphan_points) {
        scalar_t best_dist = scalar_t(0);
        int64_t best = -1;

        for (size_t f = first_new; f < facets.size(); ++f) {
          scalar_t d = facets[f].signed_distance(&points[pi * 3]);
          if (d > best_dist) {
            best_dist = d;
            best = static_cast<int64_t>(f);
          }
        }

        if (best >= 0 && best_dist > scalar_t(1e-10)) {
          facets[best].outside_set.push_back(pi);
        }
      }

      progress = true;
    }

    // Step 5: Extract results
    std::set<int64_t> vertex_set;
    simplices.clear();
    equations.clear();
    neighbors.clear();

    for (const auto& f : facets) {
      if (!f.active) continue;
      vertex_set.insert(f.v[0]);
      vertex_set.insert(f.v[1]);
      vertex_set.insert(f.v[2]);
      simplices.push_back(f.v[0]);
      simplices.push_back(f.v[1]);
      simplices.push_back(f.v[2]);
      equations.push_back(f.normal[0]);
      equations.push_back(f.normal[1]);
      equations.push_back(f.normal[2]);
      equations.push_back(f.offset);
      neighbors.push_back(-1);  // Not tracking neighbors
      neighbors.push_back(-1);
      neighbors.push_back(-1);
    }

    vertices.assign(vertex_set.begin(), vertex_set.end());
    compute_metrics(points);
  }

 private:
  std::vector<int64_t> find_initial_simplex(const scalar_t* pts, int64_t n) {
    std::vector<int64_t> result;

    // Find extremal points
    int64_t min_x = 0, max_x = 0;
    for (int64_t i = 1; i < n; ++i) {
      if (pts[i * 3] < pts[min_x * 3]) min_x = i;
      if (pts[i * 3] > pts[max_x * 3]) max_x = i;
    }

    if (min_x == max_x) return result;
    result.push_back(min_x);
    result.push_back(max_x);

    // Find point furthest from line
    scalar_t max_dist = scalar_t(0);
    int64_t third = -1;
    for (int64_t i = 0; i < n; ++i) {
      if (i == min_x || i == max_x) continue;
      scalar_t dist = point_line_dist(&pts[i * 3], &pts[min_x * 3], &pts[max_x * 3]);
      if (dist > max_dist) {
        max_dist = dist;
        third = i;
      }
    }

    if (third < 0 || max_dist < scalar_t(1e-10)) return result;
    result.push_back(third);

    // Compute plane of first 3 points
    scalar_t ax = pts[max_x * 3] - pts[min_x * 3];
    scalar_t ay = pts[max_x * 3 + 1] - pts[min_x * 3 + 1];
    scalar_t az = pts[max_x * 3 + 2] - pts[min_x * 3 + 2];
    scalar_t bx = pts[third * 3] - pts[min_x * 3];
    scalar_t by = pts[third * 3 + 1] - pts[min_x * 3 + 1];
    scalar_t bz = pts[third * 3 + 2] - pts[min_x * 3 + 2];
    scalar_t nx = ay * bz - az * by;
    scalar_t ny = az * bx - ax * bz;
    scalar_t nz = ax * by - ay * bx;
    scalar_t len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > scalar_t(1e-12)) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
    scalar_t d = -(nx * pts[min_x * 3] + ny * pts[min_x * 3 + 1] + nz * pts[min_x * 3 + 2]);

    // Find point furthest from plane
    max_dist = scalar_t(0);
    int64_t fourth = -1;
    for (int64_t i = 0; i < n; ++i) {
      if (i == min_x || i == max_x || i == third) continue;
      scalar_t dist = std::abs(nx * pts[i * 3] + ny * pts[i * 3 + 1] + nz * pts[i * 3 + 2] + d);
      if (dist > max_dist) {
        max_dist = dist;
        fourth = i;
      }
    }

    if (fourth < 0 || max_dist < scalar_t(1e-10)) return result;
    result.push_back(fourth);

    return result;
  }

  scalar_t point_line_dist(const scalar_t* p, const scalar_t* a, const scalar_t* b) {
    scalar_t abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
    scalar_t apx = p[0] - a[0], apy = p[1] - a[1], apz = p[2] - a[2];
    scalar_t cx = apy * abz - apz * aby;
    scalar_t cy = apz * abx - apx * abz;
    scalar_t cz = apx * aby - apy * abx;
    scalar_t cross_len = std::sqrt(cx * cx + cy * cy + cz * cz);
    scalar_t ab_len = std::sqrt(abx * abx + aby * aby + abz * abz);
    if (ab_len < scalar_t(1e-12)) return scalar_t(0);
    return cross_len / ab_len;
  }

  void compute_metrics(const scalar_t* points) {
    surface_area = scalar_t(0);
    volume = scalar_t(0);

    int64_t n_facets = static_cast<int64_t>(simplices.size()) / 3;
    if (n_facets == 0) return;

    // Use origin as reference for volume (simpler and correct)
    for (int64_t i = 0; i < n_facets; ++i) {
      int64_t v0 = simplices[i * 3];
      int64_t v1 = simplices[i * 3 + 1];
      int64_t v2 = simplices[i * 3 + 2];

      scalar_t p0x = points[v0 * 3], p0y = points[v0 * 3 + 1], p0z = points[v0 * 3 + 2];
      scalar_t p1x = points[v1 * 3], p1y = points[v1 * 3 + 1], p1z = points[v1 * 3 + 2];
      scalar_t p2x = points[v2 * 3], p2y = points[v2 * 3 + 1], p2z = points[v2 * 3 + 2];

      // Cross product for triangle area and normal
      scalar_t ax = p1x - p0x, ay = p1y - p0y, az = p1z - p0z;
      scalar_t bx = p2x - p0x, by = p2y - p0y, bz = p2z - p0z;
      scalar_t cx = ay * bz - az * by;
      scalar_t cy = az * bx - ax * bz;
      scalar_t cz = ax * by - ay * bx;

      // Triangle area = |cross| / 2
      surface_area += std::sqrt(cx * cx + cy * cy + cz * cz) / scalar_t(2);

      // Signed volume of tetrahedron from origin = (p0 . (p1 x p2)) / 6
      // Using: p0 . ((p1-p0) x (p2-p0)) = p0 . cross, but simpler is:
      // V = (1/6) * p0 . (cross of p1-p0 and p2-p0)
      // Actually use direct formula: V = (1/6) * |a . (b x c)| where a,b,c from origin
      scalar_t tet_vol = (p0x * (p1y * p2z - p1z * p2y) -
                          p0y * (p1x * p2z - p1z * p2x) +
                          p0z * (p1x * p2y - p1y * p2x)) / scalar_t(6);
      volume += tet_vol;
    }

    volume = std::abs(volume);
  }
};

}  // namespace torchscience::kernel::geometry
