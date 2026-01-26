// src/torchscience/csrc/cpu/space_partitioning/bvh.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

#ifdef TORCHSCIENCE_EMBREE
#include <embree4/rtcore.h>
#endif

#include "../../optix/destroy_callback.h"

namespace torchscience::cpu::space_partitioning {

#ifdef TORCHSCIENCE_EMBREE

namespace {

// Global Embree device - created once, reused
RTCDevice get_embree_device() {
    static RTCDevice device = rtcNewDevice(nullptr);
    return device;
}

}  // anonymous namespace

inline at::Tensor bvh_build(
    const at::Tensor& vertices,
    const at::Tensor& faces
) {
    TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3,
                "vertices must be (V, 3)");
    TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3,
                "faces must be (F, 3)");
    TORCH_CHECK(vertices.is_contiguous(), "vertices must be contiguous");
    TORCH_CHECK(faces.is_contiguous(), "faces must be contiguous");

    RTCDevice device = get_embree_device();
    RTCScene scene = rtcNewScene(device);

    // Create triangle geometry
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    int64_t num_vertices = vertices.size(0);
    int64_t num_faces = faces.size(0);

    // Set vertex buffer (Embree requires float)
    at::Tensor vertices_float = vertices.to(at::kFloat).contiguous();
    float* vertex_ptr = static_cast<float*>(
        rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                                RTC_FORMAT_FLOAT3, 3 * sizeof(float),
                                num_vertices)
    );
    std::memcpy(vertex_ptr, vertices_float.data_ptr<float>(),
                num_vertices * 3 * sizeof(float));

    // Set index buffer (Embree requires unsigned int)
    at::Tensor faces_uint = faces.to(at::kInt).contiguous();
    unsigned int* index_ptr = static_cast<unsigned int*>(
        rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
                                RTC_FORMAT_UINT3, 3 * sizeof(unsigned int),
                                num_faces)
    );
    std::memcpy(index_ptr, faces_uint.data_ptr<int>(),
                num_faces * 3 * sizeof(unsigned int));

    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    // Return scene handle as int64 tensor
    int64_t handle = reinterpret_cast<int64_t>(scene);
    return at::tensor({handle}, at::kLong);
}

inline void bvh_destroy(int64_t scene_handle) {
    if (scene_handle == 0) return;

    // Negative handles belong to OptiX backend
    if (scene_handle < 0) {
        if (torchscience::optix::g_scene_destroy) {
            torchscience::optix::g_scene_destroy(scene_handle);
        }
        return;
    }

    RTCScene scene = reinterpret_cast<RTCScene>(scene_handle);
    // Don't release device - it's shared/static
    rtcReleaseScene(scene);
}

#else  // !TORCHSCIENCE_EMBREE

inline at::Tensor bvh_build(
    const at::Tensor& vertices,
    const at::Tensor& faces
) {
    TORCH_CHECK(false, "BVH requires Embree. Install Embree and rebuild.");
    return at::Tensor();
}

inline void bvh_destroy(int64_t scene_handle) {
    if (scene_handle == 0) return;

    // Negative handles belong to OptiX backend
    if (scene_handle < 0) {
        if (torchscience::optix::g_scene_destroy) {
            torchscience::optix::g_scene_destroy(scene_handle);
        }
        return;
    }

    // No Embree available for positive handles
    (void)scene_handle;
}

#endif  // TORCHSCIENCE_EMBREE

}  // namespace torchscience::cpu::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("bvh_build", torchscience::cpu::space_partitioning::bvh_build);
    m.impl("bvh_destroy", torchscience::cpu::space_partitioning::bvh_destroy);
}
