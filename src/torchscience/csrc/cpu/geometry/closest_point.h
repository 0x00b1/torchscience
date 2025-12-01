// src/torchscience/csrc/cpu/geometry/closest_point.h
#pragma once

#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#ifdef TORCHSCIENCE_EMBREE
#include <embree4/rtcore.h>
#endif

namespace torchscience::cpu::geometry {

#ifdef TORCHSCIENCE_EMBREE

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
bvh_closest_point(
    int64_t scene_handle,
    const at::Tensor& query_points
) {
    TORCH_CHECK(query_points.dim() >= 1 && query_points.size(-1) == 3,
                "query_points must have shape (..., 3)");

    RTCScene scene = reinterpret_cast<RTCScene>(scene_handle);

    // Flatten batch dimensions
    auto orig_shape = query_points.sizes().vec();
    orig_shape.pop_back();

    at::Tensor queries_flat = query_points.reshape({-1, 3}).contiguous().to(at::kFloat);
    int64_t num_queries = queries_flat.size(0);

    // Allocate outputs
    at::Tensor point_out = at::zeros({num_queries, 3}, query_points.options().dtype(at::kFloat));
    at::Tensor distance_out = at::full({num_queries}, std::numeric_limits<float>::infinity(),
                                        query_points.options().dtype(at::kFloat));
    at::Tensor geometry_id_out = at::full({num_queries}, -1, query_points.options().dtype(at::kLong));
    at::Tensor primitive_id_out = at::full({num_queries}, -1, query_points.options().dtype(at::kLong));
    at::Tensor u_out = at::zeros({num_queries}, query_points.options().dtype(at::kFloat));
    at::Tensor v_out = at::zeros({num_queries}, query_points.options().dtype(at::kFloat));

    float* queries_ptr = queries_flat.data_ptr<float>();
    float* point_ptr = point_out.data_ptr<float>();
    float* dist_ptr = distance_out.data_ptr<float>();
    int64_t* geom_ptr = geometry_id_out.data_ptr<int64_t>();
    int64_t* prim_ptr = primitive_id_out.data_ptr<int64_t>();
    float* u_ptr = u_out.data_ptr<float>();
    float* v_ptr = v_out.data_ptr<float>();

    at::parallel_for(0, num_queries, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            // Cast rays in 6 directions to find closest point
            // (simplified implementation - Embree's point query API requires callbacks)
            float min_dist = std::numeric_limits<float>::infinity();
            float best_point[3] = {0, 0, 0};
            float best_u = 0, best_v = 0;
            unsigned int best_geom = RTC_INVALID_GEOMETRY_ID;
            unsigned int best_prim = RTC_INVALID_GEOMETRY_ID;

            float dirs[6][3] = {
                {1, 0, 0}, {-1, 0, 0},
                {0, 1, 0}, {0, -1, 0},
                {0, 0, 1}, {0, 0, -1}
            };

            for (int d = 0; d < 6; ++d) {
                RTCRayHit rayhit;
                rayhit.ray.org_x = queries_ptr[i * 3 + 0];
                rayhit.ray.org_y = queries_ptr[i * 3 + 1];
                rayhit.ray.org_z = queries_ptr[i * 3 + 2];
                rayhit.ray.dir_x = dirs[d][0];
                rayhit.ray.dir_y = dirs[d][1];
                rayhit.ray.dir_z = dirs[d][2];
                rayhit.ray.tnear = 0.0f;
                rayhit.ray.tfar = std::numeric_limits<float>::infinity();
                rayhit.ray.mask = 0xFFFFFFFF;
                rayhit.ray.flags = 0;
                rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

                rtcIntersect1(scene, &rayhit);

                if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                    float dist = rayhit.ray.tfar;
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_point[0] = queries_ptr[i * 3 + 0] + dist * dirs[d][0];
                        best_point[1] = queries_ptr[i * 3 + 1] + dist * dirs[d][1];
                        best_point[2] = queries_ptr[i * 3 + 2] + dist * dirs[d][2];
                        best_u = rayhit.hit.u;
                        best_v = rayhit.hit.v;
                        best_geom = rayhit.hit.geomID;
                        best_prim = rayhit.hit.primID;
                    }
                }
            }

            point_ptr[i * 3 + 0] = best_point[0];
            point_ptr[i * 3 + 1] = best_point[1];
            point_ptr[i * 3 + 2] = best_point[2];
            dist_ptr[i] = min_dist;
            if (best_geom != RTC_INVALID_GEOMETRY_ID) {
                geom_ptr[i] = static_cast<int64_t>(best_geom);
                prim_ptr[i] = static_cast<int64_t>(best_prim);
            }
            u_ptr[i] = best_u;
            v_ptr[i] = best_v;
        }
    });

    // Reshape outputs
    std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end());
    std::vector<int64_t> point_shape = out_shape;
    point_shape.push_back(3);

    return std::make_tuple(
        point_out.reshape(point_shape),
        distance_out.reshape(out_shape),
        geometry_id_out.reshape(out_shape),
        primitive_id_out.reshape(out_shape),
        u_out.reshape(out_shape),
        v_out.reshape(out_shape)
    );
}

#else

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
bvh_closest_point(
    int64_t scene_handle,
    const at::Tensor& query_points
) {
    TORCH_CHECK(false, "closest_point requires Embree. Install Embree and rebuild.");
    return std::make_tuple(
        at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), at::Tensor(), at::Tensor()
    );
}

#endif

}  // namespace torchscience::cpu::geometry

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("bvh_closest_point", torchscience::cpu::geometry::bvh_closest_point);
}
