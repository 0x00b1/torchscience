// src/torchscience/csrc/cpu/geometry/ray_intersect.h
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
bvh_ray_intersect(
    int64_t scene_handle,
    const at::Tensor& origins,
    const at::Tensor& directions
) {
    TORCH_CHECK(origins.dim() >= 1 && origins.size(-1) == 3,
                "origins must have shape (..., 3)");
    TORCH_CHECK(directions.dim() >= 1 && directions.size(-1) == 3,
                "directions must have shape (..., 3)");
    TORCH_CHECK(origins.sizes() == directions.sizes(),
                "origins and directions must have matching shapes");

    RTCScene scene = reinterpret_cast<RTCScene>(scene_handle);

    // Flatten batch dimensions
    auto orig_shape = origins.sizes().vec();
    orig_shape.pop_back();  // Remove last dim (3)

    at::Tensor origins_flat = origins.reshape({-1, 3}).contiguous().to(at::kFloat);
    at::Tensor directions_flat = directions.reshape({-1, 3}).contiguous().to(at::kFloat);

    int64_t num_rays = origins_flat.size(0);

    // Allocate outputs
    at::Tensor t_out = at::full({num_rays}, std::numeric_limits<float>::infinity(),
                                origins.options().dtype(at::kFloat));
    at::Tensor hit_out = at::zeros({num_rays}, origins.options().dtype(at::kBool));
    at::Tensor geometry_id_out = at::full({num_rays}, -1, origins.options().dtype(at::kLong));
    at::Tensor primitive_id_out = at::full({num_rays}, -1, origins.options().dtype(at::kLong));
    at::Tensor u_out = at::zeros({num_rays}, origins.options().dtype(at::kFloat));
    at::Tensor v_out = at::zeros({num_rays}, origins.options().dtype(at::kFloat));

    float* origins_ptr = origins_flat.data_ptr<float>();
    float* directions_ptr = directions_flat.data_ptr<float>();
    float* t_ptr = t_out.data_ptr<float>();
    bool* hit_ptr = hit_out.data_ptr<bool>();
    int64_t* geom_ptr = geometry_id_out.data_ptr<int64_t>();
    int64_t* prim_ptr = primitive_id_out.data_ptr<int64_t>();
    float* u_ptr = u_out.data_ptr<float>();
    float* v_ptr = v_out.data_ptr<float>();

    at::parallel_for(0, num_rays, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            RTCRayHit rayhit;

            // Initialize ray
            rayhit.ray.org_x = origins_ptr[i * 3 + 0];
            rayhit.ray.org_y = origins_ptr[i * 3 + 1];
            rayhit.ray.org_z = origins_ptr[i * 3 + 2];
            rayhit.ray.dir_x = directions_ptr[i * 3 + 0];
            rayhit.ray.dir_y = directions_ptr[i * 3 + 1];
            rayhit.ray.dir_z = directions_ptr[i * 3 + 2];
            rayhit.ray.tnear = 0.0f;
            rayhit.ray.tfar = std::numeric_limits<float>::infinity();
            rayhit.ray.mask = 0xFFFFFFFF;
            rayhit.ray.flags = 0;

            // Initialize hit
            rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;

            // Intersect
            rtcIntersect1(scene, &rayhit);

            // Store results
            if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                t_ptr[i] = rayhit.ray.tfar;
                hit_ptr[i] = true;
                geom_ptr[i] = static_cast<int64_t>(rayhit.hit.geomID);
                prim_ptr[i] = static_cast<int64_t>(rayhit.hit.primID);
                u_ptr[i] = rayhit.hit.u;
                v_ptr[i] = rayhit.hit.v;
            }
        }
    });

    // Reshape outputs to original batch shape
    std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end());

    return std::make_tuple(
        t_out.reshape(out_shape),
        hit_out.reshape(out_shape),
        geometry_id_out.reshape(out_shape),
        primitive_id_out.reshape(out_shape),
        u_out.reshape(out_shape),
        v_out.reshape(out_shape)
    );
}

#else  // !TORCHSCIENCE_EMBREE

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
bvh_ray_intersect(
    int64_t scene_handle,
    const at::Tensor& origins,
    const at::Tensor& directions
) {
    TORCH_CHECK(false, "ray_intersect requires Embree. Install Embree and rebuild.");
    return std::make_tuple(
        at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), at::Tensor(), at::Tensor()
    );
}

#endif  // TORCHSCIENCE_EMBREE

}  // namespace torchscience::cpu::geometry

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("bvh_ray_intersect", torchscience::cpu::geometry::bvh_ray_intersect);
}
