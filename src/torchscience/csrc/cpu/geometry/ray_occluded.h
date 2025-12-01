// src/torchscience/csrc/cpu/geometry/ray_occluded.h
#pragma once

#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#ifdef TORCHSCIENCE_EMBREE
#include <embree4/rtcore.h>
#endif

namespace torchscience::cpu::geometry {

#ifdef TORCHSCIENCE_EMBREE

inline at::Tensor bvh_ray_occluded(
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

    auto orig_shape = origins.sizes().vec();
    orig_shape.pop_back();

    at::Tensor origins_flat = origins.reshape({-1, 3}).contiguous().to(at::kFloat);
    at::Tensor directions_flat = directions.reshape({-1, 3}).contiguous().to(at::kFloat);

    int64_t num_rays = origins_flat.size(0);

    at::Tensor occluded_out = at::zeros({num_rays}, origins.options().dtype(at::kBool));

    float* origins_ptr = origins_flat.data_ptr<float>();
    float* directions_ptr = directions_flat.data_ptr<float>();
    bool* occluded_ptr = occluded_out.data_ptr<bool>();

    at::parallel_for(0, num_rays, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            RTCRay ray;
            ray.org_x = origins_ptr[i * 3 + 0];
            ray.org_y = origins_ptr[i * 3 + 1];
            ray.org_z = origins_ptr[i * 3 + 2];
            ray.dir_x = directions_ptr[i * 3 + 0];
            ray.dir_y = directions_ptr[i * 3 + 1];
            ray.dir_z = directions_ptr[i * 3 + 2];
            ray.tnear = 0.0f;
            ray.tfar = std::numeric_limits<float>::infinity();
            ray.mask = 0xFFFFFFFF;
            ray.flags = 0;

            rtcOccluded1(scene, &ray);

            // tfar is set to -inf if ray is occluded
            occluded_ptr[i] = (ray.tfar < 0.0f);
        }
    });

    return occluded_out.reshape(orig_shape);
}

#else

inline at::Tensor bvh_ray_occluded(
    int64_t scene_handle,
    const at::Tensor& origins,
    const at::Tensor& directions
) {
    TORCH_CHECK(false, "ray_occluded requires Embree. Install Embree and rebuild.");
    return at::Tensor();
}

#endif

}  // namespace torchscience::cpu::geometry

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("bvh_ray_occluded", torchscience::cpu::geometry::bvh_ray_occluded);
}
