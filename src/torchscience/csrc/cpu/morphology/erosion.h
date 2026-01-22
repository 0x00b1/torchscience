#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../kernel/morphology/erosion.h"
#include "../../kernel/morphology/erosion_backward.h"

namespace torchscience::cpu::morphology {

/**
 * CPU implementation for N-dimensional erosion.
 *
 * @param input Input tensor of shape (*, spatial...)
 * @param structuring_element Structuring element tensor
 * @param origin Optional origin for the SE (default: center)
 * @param padding_mode 0=zeros, 1=reflect, 2=replicate, 3=circular
 * @return Eroded tensor with same shape as input
 */
inline at::Tensor erosion(
    const at::Tensor& input,
    const at::Tensor& structuring_element,
    c10::optional<at::IntArrayRef> origin,
    int64_t padding_mode
) {
    TORCH_CHECK(input.dim() >= structuring_element.dim(),
        "erosion: input must have at least as many dimensions as structuring_element");

    const int64_t ndim = structuring_element.dim();

    // Check origin if provided
    if (origin.has_value()) {
        TORCH_CHECK(origin->size() == static_cast<size_t>(ndim),
            "erosion: origin must have same number of dimensions as structuring_element");
    }

    auto output = at::empty_like(input);
    auto input_contig = input.contiguous();
    auto se_contig = structuring_element.contiguous();

    // Compute number of spatial elements
    int64_t num_spatial = 1;
    for (int64_t d = input.dim() - ndim; d < input.dim(); ++d) {
        num_spatial *= input.size(d);
    }

    // Compute number of batch elements
    int64_t num_batch = input.numel() / num_spatial;

    // Extract spatial shapes
    std::vector<int64_t> input_shape(ndim);
    std::vector<int64_t> se_shape(ndim);
    std::vector<int64_t> input_strides(ndim);
    std::vector<int64_t> se_strides(ndim);

    for (int64_t d = 0; d < ndim; ++d) {
        input_shape[d] = input.size(input.dim() - ndim + d);
        se_shape[d] = structuring_element.size(d);
        input_strides[d] = input_contig.stride(input.dim() - ndim + d);
        se_strides[d] = se_contig.stride(d);
    }

    // Process origin
    std::vector<int64_t> origin_vec;
    const int64_t* origin_ptr = nullptr;
    if (origin.has_value()) {
        origin_vec.assign(origin->begin(), origin->end());
        origin_ptr = origin_vec.data();
    }

    // Determine if flat or non-flat morphology
    // Flat: SE is boolean mask (all values 0 or 1)
    // Non-flat: SE contains weights
    bool is_flat = (structuring_element.dtype() == at::kBool);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        input.scalar_type(),
        "erosion_cpu",
        [&] {
            const scalar_t* input_ptr = input_contig.data_ptr<scalar_t>();
            scalar_t* output_ptr = output.data_ptr<scalar_t>();

            // For flat erosion with bool SE, se_ptr is nullptr
            const scalar_t* se_ptr = nullptr;
            const bool* se_mask_ptr = nullptr;

            at::Tensor se_float;
            if (is_flat) {
                se_mask_ptr = se_contig.data_ptr<bool>();
            } else {
                // Non-flat SE: use weights directly
                se_float = se_contig.to(input.dtype());
                se_ptr = se_float.data_ptr<scalar_t>();
            }

            const scalar_t padding_value = std::numeric_limits<scalar_t>::infinity();

            at::parallel_for(0, num_batch * num_spatial, 0, [&](int64_t begin, int64_t end) {
                for (int64_t idx = begin; idx < end; ++idx) {
                    int64_t batch_idx = idx / num_spatial;
                    int64_t spatial_idx = idx % num_spatial;

                    const scalar_t* batch_input = input_ptr + batch_idx * num_spatial;
                    scalar_t* batch_output = output_ptr + batch_idx * num_spatial;

                    batch_output[spatial_idx] = kernel::morphology::erosion_scalar(
                        batch_input,
                        se_ptr,
                        se_mask_ptr,
                        spatial_idx,
                        input_shape.data(),
                        se_shape.data(),
                        origin_ptr,
                        ndim,
                        input_strides.data(),
                        se_strides.data(),
                        padding_mode,
                        padding_value
                    );
                }
            });
        }
    );

    return output;
}

/**
 * CPU implementation for erosion backward pass.
 */
inline at::Tensor erosion_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& structuring_element,
    c10::optional<at::IntArrayRef> origin,
    int64_t padding_mode
) {
    const int64_t ndim = structuring_element.dim();

    auto grad_input = at::zeros_like(input);
    auto grad_output_contig = grad_output.contiguous();
    auto input_contig = input.contiguous();
    auto se_contig = structuring_element.contiguous();

    // Compute number of spatial elements
    int64_t num_spatial = 1;
    for (int64_t d = input.dim() - ndim; d < input.dim(); ++d) {
        num_spatial *= input.size(d);
    }

    int64_t num_batch = input.numel() / num_spatial;

    // Extract shapes
    std::vector<int64_t> input_shape(ndim);
    std::vector<int64_t> se_shape(ndim);
    std::vector<int64_t> input_strides(ndim);
    std::vector<int64_t> se_strides(ndim);

    for (int64_t d = 0; d < ndim; ++d) {
        input_shape[d] = input.size(input.dim() - ndim + d);
        se_shape[d] = structuring_element.size(d);
        input_strides[d] = input_contig.stride(input.dim() - ndim + d);
        se_strides[d] = se_contig.stride(d);
    }

    std::vector<int64_t> origin_vec;
    const int64_t* origin_ptr = nullptr;
    if (origin.has_value()) {
        origin_vec.assign(origin->begin(), origin->end());
        origin_ptr = origin_vec.data();
    }

    bool is_flat = (structuring_element.dtype() == at::kBool);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16,
        at::kHalf,
        input.scalar_type(),
        "erosion_backward_cpu",
        [&] {
            const scalar_t* grad_output_ptr = grad_output_contig.data_ptr<scalar_t>();
            const scalar_t* input_ptr = input_contig.data_ptr<scalar_t>();
            scalar_t* grad_input_ptr = grad_input.data_ptr<scalar_t>();

            const scalar_t* se_ptr = nullptr;
            const bool* se_mask_ptr = nullptr;

            at::Tensor se_float;
            if (is_flat) {
                se_mask_ptr = se_contig.data_ptr<bool>();
            } else {
                se_float = se_contig.to(input.dtype());
                se_ptr = se_float.data_ptr<scalar_t>();
            }

            const scalar_t padding_value = std::numeric_limits<scalar_t>::infinity();

            // Note: backward cannot be parallelized easily due to atomic writes
            // For now, process sequentially per batch
            for (int64_t batch_idx = 0; batch_idx < num_batch; ++batch_idx) {
                const scalar_t* batch_grad_output = grad_output_ptr + batch_idx * num_spatial;
                const scalar_t* batch_input = input_ptr + batch_idx * num_spatial;
                scalar_t* batch_grad_input = grad_input_ptr + batch_idx * num_spatial;

                for (int64_t spatial_idx = 0; spatial_idx < num_spatial; ++spatial_idx) {
                    kernel::morphology::erosion_backward_scalar(
                        batch_grad_output[spatial_idx],
                        batch_input,
                        se_ptr,
                        se_mask_ptr,
                        spatial_idx,
                        batch_grad_input,
                        input_shape.data(),
                        se_shape.data(),
                        origin_ptr,
                        ndim,
                        input_strides.data(),
                        se_strides.data(),
                        padding_mode,
                        padding_value
                    );
                }
            }
        }
    );

    return grad_input;
}

}  // namespace torchscience::cpu::morphology

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("erosion", &torchscience::cpu::morphology::erosion);
    m.impl("erosion_backward", &torchscience::cpu::morphology::erosion_backward);
}
