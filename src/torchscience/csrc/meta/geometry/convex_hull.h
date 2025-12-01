#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                  at::Tensor, at::Tensor, at::Tensor>
convex_hull(const at::Tensor& points) {
  TORCH_CHECK(points.dim() >= 2,
              "convex_hull: points must be at least 2D (n, d)");

  auto options = points.options();
  auto int_options = options.dtype(at::kLong);

  // Handle batched vs unbatched
  bool batched = points.dim() > 2;
  int64_t n_dims = points.size(-1);
  int64_t n_points = points.size(-2);

  // Upper bounds for output sizes (worst case: all points on hull)
  int64_t max_vertices = n_points;
  // For n-dimensional convex hull, max facets is O(n^floor(d/2))
  // Use conservative upper bound: n * (d+1) for simplicity
  int64_t max_facets = n_points * (n_dims + 1);

  if (batched) {
    // Compute batch shape
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < points.dim() - 2; ++i) {
      batch_shape.push_back(points.size(i));
    }

    // Create batched output shapes
    std::vector<int64_t> vertices_shape = batch_shape;
    vertices_shape.push_back(max_vertices);

    std::vector<int64_t> simplices_shape = batch_shape;
    simplices_shape.push_back(max_facets);
    simplices_shape.push_back(n_dims);

    std::vector<int64_t> neighbors_shape = simplices_shape;

    std::vector<int64_t> equations_shape = batch_shape;
    equations_shape.push_back(max_facets);
    equations_shape.push_back(n_dims + 1);

    return std::make_tuple(at::empty(vertices_shape, int_options),
                           at::empty(simplices_shape, int_options),
                           at::empty(neighbors_shape, int_options),
                           at::empty(equations_shape, options),
                           at::empty(batch_shape, options),      // area
                           at::empty(batch_shape, options),      // volume
                           at::empty(batch_shape, int_options),  // n_vertices
                           at::empty(batch_shape, int_options)   // n_facets
    );
  } else {
    // Unbatched
    return std::make_tuple(
        at::empty({max_vertices}, int_options),
        at::empty({max_facets, n_dims}, int_options),
        at::empty({max_facets, n_dims}, int_options),
        at::empty({max_facets, n_dims + 1}, options),
        at::empty({}, options),      // area (scalar)
        at::empty({}, options),      // volume (scalar)
        at::empty({}, int_options),  // n_vertices
        at::empty({}, int_options)   // n_facets
    );
  }
}

}  // namespace torchscience::meta::geometry

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("convex_hull", torchscience::meta::geometry::convex_hull);
}
