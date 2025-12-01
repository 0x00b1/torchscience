#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cub/cub.cuh>

namespace torchscience::cuda::descriptive {

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

/**
 * Warp-level reduction using shuffle.
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block-level reduction using shared memory.
 */
template <typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[lane] : T(0);

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

/**
 * Kernel to compute mean for a batch of 1D arrays.
 * One block per batch element.
 */
template <typename scalar_t>
__global__ void kurtosis_mean_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ means,
    int64_t reduce_size,
    int64_t batch_size
) {
    __shared__ scalar_t shared[32];  // For warp reduction

    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const scalar_t* batch_input = input + batch_idx * reduce_size;

    // Compute local sum
    scalar_t local_sum = 0;
    for (int64_t i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        local_sum += batch_input[i];
    }

    // Block-level reduction
    scalar_t total_sum = block_reduce_sum(local_sum, shared);

    if (threadIdx.x == 0) {
        means[batch_idx] = total_sum / scalar_t(reduce_size);
    }
}

/**
 * Kernel to compute m2 and m4 for a batch of 1D arrays.
 * Uses precomputed means.
 */
template <typename scalar_t>
__global__ void kurtosis_moments_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ means,
    scalar_t* __restrict__ m2_out,
    scalar_t* __restrict__ m4_out,
    int64_t reduce_size,
    int64_t batch_size
) {
    __shared__ scalar_t shared_m2[32];
    __shared__ scalar_t shared_m4[32];

    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const scalar_t* batch_input = input + batch_idx * reduce_size;
    scalar_t mean = means[batch_idx];

    // Compute local sums of (x - mean)^2 and (x - mean)^4
    scalar_t local_m2 = 0;
    scalar_t local_m4 = 0;
    for (int64_t i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        scalar_t d = batch_input[i] - mean;
        scalar_t d2 = d * d;
        local_m2 += d2;
        local_m4 += d2 * d2;
    }

    // Block-level reduction
    scalar_t total_m2 = block_reduce_sum(local_m2, shared_m2);
    __syncthreads();
    scalar_t total_m4 = block_reduce_sum(local_m4, shared_m4);

    if (threadIdx.x == 0) {
        m2_out[batch_idx] = total_m2 / scalar_t(reduce_size);
        m4_out[batch_idx] = total_m4 / scalar_t(reduce_size);
    }
}

/**
 * Kernel to compute final kurtosis from m2 and m4.
 */
template <typename scalar_t>
__global__ void kurtosis_finalize_kernel(
    const scalar_t* __restrict__ m2,
    const scalar_t* __restrict__ m4,
    scalar_t* __restrict__ output,
    int64_t batch_size,
    int64_t reduce_size,
    bool fisher,
    bool bias
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    scalar_t m2_val = m2[idx];
    scalar_t m4_val = m4[idx];

    // Handle zero variance
    if (m2_val == scalar_t(0)) {
        output[idx] = nan("");
        return;
    }

    // Handle edge cases for unbiased estimate
    if (!bias && reduce_size <= 3) {
        output[idx] = nan("");
        return;
    }

    scalar_t m2_sq = m2_val * m2_val;
    scalar_t g2 = m4_val / m2_sq;

    if (fisher) {
        g2 -= scalar_t(3);
    }

    if (!bias) {
        scalar_t n = scalar_t(reduce_size);
        g2 = ((n - scalar_t(1)) / ((n - scalar_t(2)) * (n - scalar_t(3)))) *
             ((n + scalar_t(1)) * g2 + scalar_t(6));
    }

    output[idx] = g2;
}

/**
 * Kernel to compute m3 for backward pass.
 */
template <typename scalar_t>
__global__ void kurtosis_m3_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ means,
    scalar_t* __restrict__ m3_out,
    int64_t reduce_size,
    int64_t batch_size
) {
    __shared__ scalar_t shared[32];

    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const scalar_t* batch_input = input + batch_idx * reduce_size;
    scalar_t mean = means[batch_idx];

    scalar_t local_m3 = 0;
    for (int64_t i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        scalar_t d = batch_input[i] - mean;
        local_m3 += d * d * d;
    }

    scalar_t total_m3 = block_reduce_sum(local_m3, shared);

    if (threadIdx.x == 0) {
        m3_out[batch_idx] = total_m3 / scalar_t(reduce_size);
    }
}

/**
 * Kernel to compute gradient of kurtosis.
 */
template <typename scalar_t>
__global__ void kurtosis_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ means,
    const scalar_t* __restrict__ m2,
    const scalar_t* __restrict__ m3,
    const scalar_t* __restrict__ m4,
    scalar_t* __restrict__ grad_input,
    int64_t reduce_size,
    int64_t batch_size,
    bool fisher,
    bool bias
) {
    int64_t batch_idx = blockIdx.x;
    int64_t elem_idx = threadIdx.x;

    if (batch_idx >= batch_size) return;

    scalar_t grad_out = grad_output[batch_idx];
    scalar_t mean = means[batch_idx];
    scalar_t m2_val = m2[batch_idx];
    scalar_t m3_val = m3[batch_idx];
    scalar_t m4_val = m4[batch_idx];

    // Handle zero variance
    if (m2_val == scalar_t(0)) {
        for (int64_t i = elem_idx; i < reduce_size; i += blockDim.x) {
            grad_input[batch_idx * reduce_size + i] = scalar_t(0);
        }
        return;
    }

    scalar_t m2_sq = m2_val * m2_val;
    scalar_t g2 = m4_val / m2_sq;
    scalar_t k_for_grad = fisher ? (g2 - scalar_t(3)) : g2;

    scalar_t n = scalar_t(reduce_size);
    scalar_t coeff = scalar_t(4) / (n * m2_sq);

    scalar_t dG2_dg2 = scalar_t(1);
    if (!bias) {
        dG2_dg2 = ((n - scalar_t(1)) * (n + scalar_t(1))) /
                  ((n - scalar_t(2)) * (n - scalar_t(3)));
    }

    const scalar_t* batch_input = input + batch_idx * reduce_size;
    scalar_t* batch_grad = grad_input + batch_idx * reduce_size;

    for (int64_t i = elem_idx; i < reduce_size; i += blockDim.x) {
        scalar_t d = batch_input[i] - mean;
        scalar_t d3 = d * d * d;
        scalar_t dg2_dxi = coeff * (d3 - m3_val - scalar_t(2) * k_for_grad * m2_val * d);

        if (!bias) {
            dg2_dxi *= dG2_dg2;
        }

        batch_grad[i] = grad_out * dg2_dxi;
    }
}

/**
 * Compute output shape after reduction.
 */
std::vector<int64_t> compute_output_shape(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim
) {
    std::vector<int64_t> output_shape;
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        if (keepdim) {
            output_shape.assign(ndim, 1);
        }
    } else {
        std::vector<bool> reduce_dim(ndim, false);
        for (int64_t d : *dim) {
            int64_t pos_d = d >= 0 ? d : d + ndim;
            reduce_dim[pos_d] = true;
        }

        for (int64_t i = 0; i < ndim; ++i) {
            if (reduce_dim[i]) {
                if (keepdim) {
                    output_shape.push_back(1);
                }
            } else {
                output_shape.push_back(input_sizes[i]);
            }
        }
    }

    return output_shape;
}

std::pair<int64_t, int64_t> compute_reduce_info(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim
) {
    auto input_sizes = input.sizes();
    int64_t ndim = input.dim();

    if (!dim.has_value() || dim->empty()) {
        return {input.numel(), 1};
    }

    std::vector<bool> reduce_dim(ndim, false);
    for (int64_t d : *dim) {
        int64_t pos_d = d >= 0 ? d : d + ndim;
        reduce_dim[pos_d] = true;
    }

    int64_t reduce_size = 1;
    int64_t batch_size = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        if (reduce_dim[i]) {
            reduce_size *= input_sizes[i];
        } else {
            batch_size *= input_sizes[i];
        }
    }

    return {reduce_size, batch_size};
}

}  // namespace

/**
 * CUDA implementation of kurtosis.
 */
at::Tensor kurtosis(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    TORCH_CHECK(input.is_cuda(), "kurtosis: input must be a CUDA tensor");
    TORCH_CHECK(input.numel() > 0, "kurtosis: input tensor must be non-empty");

    c10::cuda::CUDAGuard device_guard(input.device());

    auto output_shape = compute_output_shape(input, dim, keepdim);
    auto [reduce_size, batch_size] = compute_reduce_info(input, dim);

    at::Tensor input_contig = input.contiguous();

    auto options = input_contig.options();
    at::Tensor output = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    // Temporary tensors for intermediate results
    at::Tensor means = at::empty({batch_size}, options);
    at::Tensor m2 = at::empty({batch_size}, options);
    at::Tensor m4 = at::empty({batch_size}, options);

    // Permute and reshape if needed
    at::Tensor permuted_view;
    if (!dim.has_value() || dim->empty()) {
        permuted_view = input_contig.view({1, reduce_size});
    } else {
        int64_t ndim = input.dim();
        std::vector<bool> reduce_dim(ndim, false);
        for (int64_t d : *dim) {
            int64_t pos_d = d >= 0 ? d : d + ndim;
            reduce_dim[pos_d] = true;
        }

        std::vector<int64_t> permutation;
        for (int64_t i = 0; i < ndim; ++i) {
            if (!reduce_dim[i]) {
                permutation.push_back(i);
            }
        }
        for (int64_t i = 0; i < ndim; ++i) {
            if (reduce_dim[i]) {
                permutation.push_back(i);
            }
        }

        at::Tensor permuted = input_contig.permute(permutation).contiguous();
        permuted_view = permuted.view({batch_size, reduce_size});
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_contig.scalar_type(),
        "kurtosis_cuda",
        [&]() {
            const scalar_t* input_ptr = permuted_view.data_ptr<scalar_t>();
            scalar_t* means_ptr = means.data_ptr<scalar_t>();
            scalar_t* m2_ptr = m2.data_ptr<scalar_t>();
            scalar_t* m4_ptr = m4.data_ptr<scalar_t>();
            scalar_t* output_ptr = output.data_ptr<scalar_t>();

            // Launch mean kernel
            kurtosis_mean_kernel<scalar_t><<<batch_size, BLOCK_SIZE, 0, stream>>>(
                input_ptr, means_ptr, reduce_size, batch_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Launch moments kernel
            kurtosis_moments_kernel<scalar_t><<<batch_size, BLOCK_SIZE, 0, stream>>>(
                input_ptr, means_ptr, m2_ptr, m4_ptr, reduce_size, batch_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Launch finalize kernel
            int num_blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kurtosis_finalize_kernel<scalar_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                m2_ptr, m4_ptr, output_ptr, batch_size, reduce_size, fisher, bias
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    return output;
}

/**
 * CUDA backward pass for kurtosis.
 */
at::Tensor kurtosis_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    TORCH_CHECK(input.is_cuda(), "kurtosis_backward: input must be a CUDA tensor");

    c10::cuda::CUDAGuard device_guard(input.device());

    at::Tensor grad_input = at::zeros_like(input);
    at::Tensor input_contig = input.contiguous();

    auto [reduce_size, batch_size] = compute_reduce_info(input, dim);

    auto options = input_contig.options();
    at::Tensor means = at::empty({batch_size}, options);
    at::Tensor m2 = at::empty({batch_size}, options);
    at::Tensor m3 = at::empty({batch_size}, options);
    at::Tensor m4 = at::empty({batch_size}, options);

    // Permute and reshape
    at::Tensor permuted_view;
    std::vector<int64_t> permutation;
    if (!dim.has_value() || dim->empty()) {
        permuted_view = input_contig.view({1, reduce_size});
    } else {
        int64_t ndim = input.dim();
        std::vector<bool> reduce_dim(ndim, false);
        for (int64_t d : *dim) {
            int64_t pos_d = d >= 0 ? d : d + ndim;
            reduce_dim[pos_d] = true;
        }

        for (int64_t i = 0; i < ndim; ++i) {
            if (!reduce_dim[i]) {
                permutation.push_back(i);
            }
        }
        for (int64_t i = 0; i < ndim; ++i) {
            if (reduce_dim[i]) {
                permutation.push_back(i);
            }
        }

        at::Tensor permuted = input_contig.permute(permutation).contiguous();
        permuted_view = permuted.view({batch_size, reduce_size});
    }

    // Expand grad_output
    at::Tensor grad_output_expanded;
    if (!dim.has_value() || dim->empty()) {
        grad_output_expanded = grad_output.expand({1});
    } else {
        if (keepdim) {
            grad_output_expanded = grad_output.contiguous().view({batch_size});
        } else {
            int64_t ndim = input.dim();
            std::vector<bool> reduce_dim(ndim, false);
            for (int64_t d : *dim) {
                int64_t pos_d = d >= 0 ? d : d + ndim;
                reduce_dim[pos_d] = true;
            }

            at::Tensor temp = grad_output;
            for (int64_t i = 0; i < ndim; ++i) {
                if (reduce_dim[i]) {
                    temp = temp.unsqueeze(i);
                }
            }
            grad_output_expanded = temp.contiguous().view({batch_size});
        }
    }

    at::Tensor grad_permuted = at::zeros({batch_size, reduce_size}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_contig.scalar_type(),
        "kurtosis_backward_cuda",
        [&]() {
            const scalar_t* input_ptr = permuted_view.data_ptr<scalar_t>();
            const scalar_t* grad_out_ptr = grad_output_expanded.data_ptr<scalar_t>();
            scalar_t* means_ptr = means.data_ptr<scalar_t>();
            scalar_t* m2_ptr = m2.data_ptr<scalar_t>();
            scalar_t* m3_ptr = m3.data_ptr<scalar_t>();
            scalar_t* m4_ptr = m4.data_ptr<scalar_t>();
            scalar_t* grad_ptr = grad_permuted.data_ptr<scalar_t>();

            // Compute mean
            kurtosis_mean_kernel<scalar_t><<<batch_size, BLOCK_SIZE, 0, stream>>>(
                input_ptr, means_ptr, reduce_size, batch_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Compute moments
            kurtosis_moments_kernel<scalar_t><<<batch_size, BLOCK_SIZE, 0, stream>>>(
                input_ptr, means_ptr, m2_ptr, m4_ptr, reduce_size, batch_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Compute m3
            kurtosis_m3_kernel<scalar_t><<<batch_size, BLOCK_SIZE, 0, stream>>>(
                input_ptr, means_ptr, m3_ptr, reduce_size, batch_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Compute gradient
            kurtosis_backward_kernel<scalar_t><<<batch_size, BLOCK_SIZE, 0, stream>>>(
                grad_out_ptr, input_ptr, means_ptr, m2_ptr, m3_ptr, m4_ptr,
                grad_ptr, reduce_size, batch_size, fisher, bias
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );

    // Inverse permutation
    if (!dim.has_value() || dim->empty()) {
        grad_input = grad_permuted.view(input.sizes());
    } else {
        int64_t ndim = input.dim();
        std::vector<int64_t> inverse_perm(ndim);
        for (int64_t i = 0; i < ndim; ++i) {
            inverse_perm[permutation[i]] = i;
        }

        at::Tensor permuted = input_contig.permute(permutation).contiguous();
        grad_input = grad_permuted.view(permuted.sizes())
            .permute(inverse_perm)
            .contiguous();
    }

    return grad_input;
}

/**
 * CUDA double-backward pass for kurtosis.
 */
std::tuple<at::Tensor, at::Tensor> kurtosis_backward_backward(
    const at::Tensor& grad_grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    bool fisher,
    bool bias
) {
    // For now, return zeros (can be extended for full support)
    at::Tensor grad_grad_output = at::zeros_like(grad_output);
    at::Tensor new_grad_input = at::zeros_like(input);

    return std::make_tuple(grad_grad_output, new_grad_input);
}

}  // namespace torchscience::cuda::descriptive

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "kurtosis",
        &torchscience::cuda::descriptive::kurtosis
    );

    module.impl(
        "kurtosis_backward",
        &torchscience::cuda::descriptive::kurtosis_backward
    );

    module.impl(
        "kurtosis_backward_backward",
        &torchscience::cuda::descriptive::kurtosis_backward_backward
    );
}
