// src/torchscience/csrc/optix/space_partitioning/bvh.h
// OptiX-accelerated BVH build and destroy
#pragma once

#ifdef TORCHSCIENCE_OPTIX

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>
#include <optix.h>

#include "../context.h"

namespace torchscience::optix::space_partitioning {

inline at::Tensor bvh_build(
    const at::Tensor& vertices,
    const at::Tensor& faces) {
  TORCH_CHECK(
      vertices.dim() == 2 && vertices.size(1) == 3,
      "bvh_build: vertices must be (V, 3), got ", vertices.sizes());
  TORCH_CHECK(
      faces.dim() == 2 && faces.size(1) == 3,
      "bvh_build: faces must be (F, 3), got ", faces.sizes());

  c10::cuda::CUDAGuard device_guard(vertices.device());

  auto& ctx = OptiXContext::instance();

  // Prepare vertex and index data on device
  at::Tensor verts_f32 = vertices.to(at::kFloat).contiguous();
  at::Tensor faces_i32 = faces.to(at::kInt).contiguous();

  int64_t num_vertices = verts_f32.size(0);
  int64_t num_triangles = faces_i32.size(0);

  // Upload vertex data to device buffer (OptiX needs persistent buffers)
  CUdeviceptr d_vertices = 0;
  size_t verts_bytes = num_vertices * 3 * sizeof(float);
  cudaMalloc(reinterpret_cast<void**>(&d_vertices), verts_bytes);
  cudaMemcpy(
      reinterpret_cast<void*>(d_vertices),
      verts_f32.data_ptr<float>(), verts_bytes,
      cudaMemcpyDeviceToDevice);

  // Upload index data
  CUdeviceptr d_indices = 0;
  size_t indices_bytes = num_triangles * 3 * sizeof(unsigned int);
  cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_bytes);
  cudaMemcpy(
      reinterpret_cast<void*>(d_indices),
      faces_i32.data_ptr<int>(), indices_bytes,
      cudaMemcpyDeviceToDevice);

  // Build input: triangle geometry
  OptixBuildInput build_input = {};
  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

  build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_input.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
  build_input.triangleArray.numVertices = static_cast<unsigned int>(num_vertices);
  build_input.triangleArray.vertexBuffers = &d_vertices;

  build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  build_input.triangleArray.indexStrideInBytes = 3 * sizeof(unsigned int);
  build_input.triangleArray.numIndexTriplets =
      static_cast<unsigned int>(num_triangles);
  build_input.triangleArray.indexBuffer = d_indices;

  unsigned int input_flags = OPTIX_GEOMETRY_FLAG_NONE;
  build_input.triangleArray.flags = &input_flags;
  build_input.triangleArray.numSbtRecords = 1;

  // Acceleration structure options
  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags =
      OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  // Query memory requirements
  OptixAccelBufferSizes buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(
      ctx.context(), &accel_options, &build_input, 1, &buffer_sizes));

  // Allocate temp buffer
  CUdeviceptr d_temp = 0;
  cudaMalloc(reinterpret_cast<void**>(&d_temp), buffer_sizes.tempSizeInBytes);

  // Allocate output buffer
  CUdeviceptr d_gas_output = 0;
  cudaMalloc(
      reinterpret_cast<void**>(&d_gas_output),
      buffer_sizes.outputSizeInBytes);

  // Allocate compacted size output
  CUdeviceptr d_compacted_size = 0;
  cudaMalloc(reinterpret_cast<void**>(&d_compacted_size), sizeof(size_t));

  OptixAccelEmitDesc emit_desc = {};
  emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit_desc.result = d_compacted_size;

  // Build GAS
  OptixTraversableHandle traversable = 0;
  OPTIX_CHECK(optixAccelBuild(
      ctx.context(),
      at::cuda::getCurrentCUDAStream(),
      &accel_options,
      &build_input, 1,
      d_temp, buffer_sizes.tempSizeInBytes,
      d_gas_output, buffer_sizes.outputSizeInBytes,
      &traversable,
      &emit_desc, 1));

  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

  // Compact the GAS
  size_t compacted_size = 0;
  cudaMemcpy(
      &compacted_size,
      reinterpret_cast<void*>(d_compacted_size),
      sizeof(size_t), cudaMemcpyDeviceToHost);

  CUdeviceptr d_gas_compacted = 0;
  if (compacted_size < buffer_sizes.outputSizeInBytes) {
    cudaMalloc(reinterpret_cast<void**>(&d_gas_compacted), compacted_size);
    OPTIX_CHECK(optixAccelCompact(
        ctx.context(),
        at::cuda::getCurrentCUDAStream(),
        traversable,
        d_gas_compacted, compacted_size,
        &traversable));
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

    // Free uncompacted buffer
    cudaFree(reinterpret_cast<void*>(d_gas_output));
    d_gas_output = d_gas_compacted;
  }

  // Free temp buffers
  cudaFree(reinterpret_cast<void*>(d_temp));
  cudaFree(reinterpret_cast<void*>(d_compacted_size));
  cudaFree(reinterpret_cast<void*>(d_vertices));
  cudaFree(reinterpret_cast<void*>(d_indices));

  // Register GAS resources
  GASResources resources;
  resources.d_gas_output = d_gas_output;
  resources.gas_output_size = compacted_size > 0 ? compacted_size
                                                  : buffer_sizes.outputSizeInBytes;
  resources.traversable = traversable;

  int64_t handle = ctx.register_gas(resources);

  // Return negative handle to distinguish from Embree handles
  return at::tensor({-handle}, at::kLong);
}

}  // namespace torchscience::optix::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
  m.impl(
      "bvh_build",
      torchscience::optix::space_partitioning::bvh_build);
}

#endif  // TORCHSCIENCE_OPTIX
