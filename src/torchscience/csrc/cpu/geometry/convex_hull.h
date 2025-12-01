#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../kernel/geometry/convex_hull.h"

namespace torchscience::cpu::geometry {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                  at::Tensor, at::Tensor, at::Tensor>
convex_hull(const at::Tensor& points) {
  TORCH_CHECK(points.dim() >= 2,
              "convex_hull: points must be at least 2D (n, d)");

  int64_t n_dims = points.size(-1);
  TORCH_CHECK(n_dims == 2 || n_dims == 3,
              "convex_hull: only 2D and 3D supported, got d=", n_dims);

  auto options = points.options();
  auto int_options = options.dtype(at::kLong);

  // Handle batched vs unbatched
  bool batched = points.dim() > 2;

  at::Tensor vertices, simplices, neighbors, equations, area, volume,
      n_vertices, n_facets;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, points.scalar_type(), "convex_hull_cpu", [&]() {
        if (batched) {
          // Flatten batch dimensions
          auto batch_shape = points.sizes().slice(0, points.dim() - 2);
          int64_t batch_size = 1;
          for (auto s : batch_shape) batch_size *= s;

          auto points_flat =
              points.reshape({batch_size, points.size(-2), n_dims}).contiguous();

          // Compute hulls in parallel
          std::vector<std::vector<int64_t>> all_vertices(batch_size);
          std::vector<std::vector<int64_t>> all_simplices(batch_size);
          std::vector<std::vector<int64_t>> all_neighbors(batch_size);
          std::vector<std::vector<scalar_t>> all_equations(batch_size);
          std::vector<scalar_t> all_areas(batch_size);
          std::vector<scalar_t> all_volumes(batch_size);

          at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
            for (int64_t b = begin; b < end; ++b) {
              auto pts = points_flat[b].contiguous();
              const scalar_t* pts_ptr = pts.data_ptr<scalar_t>();
              int64_t n = pts.size(0);

              if (n_dims == 2) {
                kernel::geometry::ConvexHull2D<scalar_t> hull;
                hull.compute(pts_ptr, n);
                all_vertices[b] = hull.vertices;
                // For 2D, simplices are edges (pairs)
                for (size_t i = 0; i < hull.vertices.size(); ++i) {
                  all_simplices[b].push_back(hull.vertices[i]);
                  all_simplices[b].push_back(
                      hull.vertices[(i + 1) % hull.vertices.size()]);
                }
                all_equations[b] = hull.equations;
                all_areas[b] = hull.perimeter;
                all_volumes[b] = hull.area;
              } else {
                kernel::geometry::ConvexHull3D<scalar_t> hull;
                hull.compute(pts_ptr, n);
                all_vertices[b] = hull.vertices;
                all_simplices[b] = hull.simplices;
                all_neighbors[b] = hull.neighbors;
                all_equations[b] = hull.equations;
                all_areas[b] = hull.surface_area;
                all_volumes[b] = hull.volume;
              }
            }
          });

          // Find max sizes for padding
          int64_t max_vertices = 0, max_facets = 0;
          for (int64_t b = 0; b < batch_size; ++b) {
            max_vertices = std::max(
                max_vertices, static_cast<int64_t>(all_vertices[b].size()));
            int64_t nf =
                n_dims == 2
                    ? static_cast<int64_t>(all_simplices[b].size()) / 2
                    : static_cast<int64_t>(all_simplices[b].size()) / 3;
            max_facets = std::max(max_facets, nf);
          }

          // Allocate output tensors
          vertices = at::full({batch_size, max_vertices}, -1, int_options);
          simplices =
              at::full({batch_size, max_facets, n_dims}, -1, int_options);
          neighbors =
              at::full({batch_size, max_facets, n_dims}, -1, int_options);
          equations =
              at::zeros({batch_size, max_facets, n_dims + 1}, options);
          area = at::zeros({batch_size}, options);
          volume = at::zeros({batch_size}, options);
          n_vertices = at::zeros({batch_size}, int_options);
          n_facets = at::zeros({batch_size}, int_options);

          // Copy data
          auto verts_acc = vertices.accessor<int64_t, 2>();
          auto simps_acc = simplices.accessor<int64_t, 3>();
          auto neigh_acc = neighbors.accessor<int64_t, 3>();
          auto eqs_acc = equations.accessor<scalar_t, 3>();
          auto area_acc = area.accessor<scalar_t, 1>();
          auto vol_acc = volume.accessor<scalar_t, 1>();
          auto nv_acc = n_vertices.accessor<int64_t, 1>();
          auto nf_acc = n_facets.accessor<int64_t, 1>();

          for (int64_t b = 0; b < batch_size; ++b) {
            nv_acc[b] = static_cast<int64_t>(all_vertices[b].size());
            for (size_t i = 0; i < all_vertices[b].size(); ++i) {
              verts_acc[b][i] = all_vertices[b][i];
            }

            int64_t stride = n_dims == 2 ? 2 : 3;
            int64_t nf =
                static_cast<int64_t>(all_simplices[b].size()) / stride;
            nf_acc[b] = nf;

            for (int64_t f = 0; f < nf; ++f) {
              for (int64_t d = 0; d < stride; ++d) {
                simps_acc[b][f][d] = all_simplices[b][f * stride + d];
              }
            }

            if (!all_neighbors[b].empty()) {
              for (int64_t f = 0; f < nf; ++f) {
                for (int64_t d = 0; d < stride; ++d) {
                  neigh_acc[b][f][d] = all_neighbors[b][f * stride + d];
                }
              }
            }

            int64_t eq_stride = n_dims + 1;
            for (int64_t f = 0;
                 f < nf &&
                 f * eq_stride < static_cast<int64_t>(all_equations[b].size());
                 ++f) {
              for (int64_t d = 0; d < eq_stride; ++d) {
                eqs_acc[b][f][d] = all_equations[b][f * eq_stride + d];
              }
            }

            area_acc[b] = all_areas[b];
            vol_acc[b] = all_volumes[b];
          }

          // Reshape back to original batch dims
          std::vector<int64_t> new_verts_shape(batch_shape.begin(),
                                               batch_shape.end());
          new_verts_shape.push_back(max_vertices);
          vertices = vertices.reshape(new_verts_shape);

          std::vector<int64_t> new_simps_shape(batch_shape.begin(),
                                               batch_shape.end());
          new_simps_shape.push_back(max_facets);
          new_simps_shape.push_back(n_dims);
          simplices = simplices.reshape(new_simps_shape);
          neighbors = neighbors.reshape(new_simps_shape);

          std::vector<int64_t> new_eqs_shape(batch_shape.begin(),
                                             batch_shape.end());
          new_eqs_shape.push_back(max_facets);
          new_eqs_shape.push_back(n_dims + 1);
          equations = equations.reshape(new_eqs_shape);

          std::vector<int64_t> new_scalar_shape(batch_shape.begin(),
                                                batch_shape.end());
          area = area.reshape(new_scalar_shape);
          volume = volume.reshape(new_scalar_shape);
          n_vertices = n_vertices.reshape(new_scalar_shape);
          n_facets = n_facets.reshape(new_scalar_shape);

        } else {
          // Unbatched
          auto pts = points.contiguous();
          const scalar_t* pts_ptr = pts.data_ptr<scalar_t>();
          int64_t n = pts.size(0);

          if (n_dims == 2) {
            kernel::geometry::ConvexHull2D<scalar_t> hull;
            hull.compute(pts_ptr, n);

            int64_t nv = static_cast<int64_t>(hull.vertices.size());
            int64_t nf = nv;  // In 2D, #facets = #vertices (edges)

            vertices = at::empty({nv}, int_options);
            simplices = at::empty({nf, 2}, int_options);
            neighbors = at::full({nf, 2}, -1, int_options);
            equations = at::empty({nf, 3}, options);
            area = at::scalar_tensor(hull.perimeter, options);
            volume = at::scalar_tensor(hull.area, options);
            n_vertices = at::scalar_tensor(nv, int_options);
            n_facets = at::scalar_tensor(nf, int_options);

            auto verts_ptr = vertices.data_ptr<int64_t>();
            auto simps_ptr = simplices.data_ptr<int64_t>();
            auto eqs_ptr = equations.data_ptr<scalar_t>();

            for (int64_t i = 0; i < nv; ++i) {
              verts_ptr[i] = hull.vertices[i];
              simps_ptr[i * 2] = hull.vertices[i];
              simps_ptr[i * 2 + 1] = hull.vertices[(i + 1) % nv];
            }

            for (int64_t i = 0; i < nf * 3; ++i) {
              eqs_ptr[i] = hull.equations[i];
            }
          } else {
            kernel::geometry::ConvexHull3D<scalar_t> hull;
            hull.compute(pts_ptr, n);

            int64_t nv = static_cast<int64_t>(hull.vertices.size());
            int64_t nf = static_cast<int64_t>(hull.simplices.size()) / 3;

            vertices = at::empty({nv}, int_options);
            simplices = at::empty({nf, 3}, int_options);
            neighbors = at::empty({nf, 3}, int_options);
            equations = at::empty({nf, 4}, options);
            area = at::scalar_tensor(hull.surface_area, options);
            volume = at::scalar_tensor(hull.volume, options);
            n_vertices = at::scalar_tensor(nv, int_options);
            n_facets = at::scalar_tensor(nf, int_options);

            std::memcpy(vertices.data_ptr<int64_t>(), hull.vertices.data(),
                        nv * sizeof(int64_t));
            std::memcpy(simplices.data_ptr<int64_t>(), hull.simplices.data(),
                        nf * 3 * sizeof(int64_t));
            std::memcpy(neighbors.data_ptr<int64_t>(), hull.neighbors.data(),
                        nf * 3 * sizeof(int64_t));
            std::memcpy(equations.data_ptr<scalar_t>(), hull.equations.data(),
                        nf * 4 * sizeof(scalar_t));
          }
        }
      });

  return std::make_tuple(vertices, simplices, neighbors, equations, area,
                         volume, n_vertices, n_facets);
}

}  // namespace torchscience::cpu::geometry

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("convex_hull", torchscience::cpu::geometry::convex_hull);
}
