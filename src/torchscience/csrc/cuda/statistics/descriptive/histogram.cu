#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace torchscience::cuda::descriptive {

namespace {

constexpr int BLOCK_SIZE = 256;

/**
 * Device function for binary search to find bin index.
 *
 * For closed="right" (default): bin i contains (edge[i], edge[i+1]]
 * For closed="left": bin i contains [edge[i], edge[i+1])
 *
 * @param value The value to place in a bin
 * @param edges Array of (num_bins + 1) bin edges
 * @param num_bins Number of bins
 * @param closed_right True for (a, b] intervals, false for [a, b) intervals
 * @return Bin index (0 to num_bins-1), or -1 if value is below range,
 *         or num_bins if value is above range, or -2 for NaN
 */
template <typename T>
__device__ __forceinline__ int64_t find_bin_device(
    T value,
    const T* edges,
    int64_t num_bins,
    bool closed_right
) {
    // Handle NaN - return special value
    if (value != value) {  // NaN check
        return -2;  // Special marker for NaN
    }

    T left_edge = edges[0];
    T right_edge = edges[num_bins];

    // Check out of bounds
    if (closed_right) {
        // (a, b] - value at left edge goes to first bin only if it equals left_edge exactly
        if (value <= left_edge) {
            // For closed right, left boundary is special case
            if (value == left_edge) {
                return 0;  // Include left boundary in first bin
            }
            return -1;  // Below range
        }
        if (value > right_edge) {
            return num_bins;  // Above range
        }
    } else {
        // [a, b)
        if (value < left_edge) {
            return -1;  // Below range
        }
        if (value >= right_edge) {
            // For closed left, right boundary is special case
            if (value == right_edge) {
                return num_bins - 1;  // Include right boundary in last bin
            }
            return num_bins;  // Above range
        }
    }

    // Binary search to find the bin
    int64_t lo = 0;
    int64_t hi = num_bins;

    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;

        if (closed_right) {
            // Looking for bin where edge[mid] < value <= edge[mid+1]
            if (edges[mid + 1] < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        } else {
            // Looking for bin where edge[mid] <= value < edge[mid+1]
            if (edges[mid + 1] <= value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
    }

    return lo;
}

/**
 * CUDA kernel for histogram binning using atomic adds.
 *
 * Each thread processes one element and atomically adds to the appropriate bin.
 *
 * @param input Input data array
 * @param weights Optional weights array (nullptr for unweighted)
 * @param edges Bin edges array
 * @param counts Output histogram counts (modified via atomicAdd)
 * @param total Output total count (modified via atomicAdd)
 * @param n Number of input elements
 * @param num_bins Number of bins
 * @param closed_right True for (a, b] intervals
 * @param clamp_out_of_bounds True to clamp out-of-bounds values to edge bins
 */
template <typename scalar_t>
__global__ void histogram_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ edges,
    scalar_t* __restrict__ counts,
    scalar_t* __restrict__ total,
    int64_t n,
    int64_t num_bins,
    bool closed_right,
    bool clamp_out_of_bounds
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    scalar_t value = input[idx];
    scalar_t weight = weights ? weights[idx] : scalar_t(1);

    int64_t bin_idx = find_bin_device(value, edges, num_bins, closed_right);

    // Handle special cases
    if (bin_idx == -2) {
        // NaN value - ignore
        return;
    }

    if (bin_idx == -1) {
        // Below range
        if (clamp_out_of_bounds) {
            bin_idx = 0;
        } else {
            return;  // Ignore
        }
    } else if (bin_idx == num_bins) {
        // Above range
        if (clamp_out_of_bounds) {
            bin_idx = num_bins - 1;
        } else {
            return;  // Ignore
        }
    }

    // Atomic add to bin count
    atomicAdd(&counts[bin_idx], weight);
    atomicAdd(total, weight);
}

/**
 * CUDA kernel for density normalization.
 *
 * @param counts Histogram counts (modified in place)
 * @param num_bins Number of bins
 * @param total_count Total count (or sum of weights)
 * @param bin_width Width of each bin (for equal-width histograms)
 */
template <typename scalar_t>
__global__ void normalize_density_kernel(
    scalar_t* __restrict__ counts,
    int64_t num_bins,
    scalar_t total_count,
    scalar_t bin_width
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_bins) return;

    // Handle edge cases
    if (total_count <= scalar_t(0) || bin_width <= scalar_t(0)) {
        counts[idx] = scalar_t(0);
        return;
    }

    scalar_t normalizer = total_count * bin_width;
    counts[idx] /= normalizer;
}

/**
 * CUDA kernel for density normalization with variable bin widths.
 *
 * @param counts Histogram counts (modified in place)
 * @param edges Bin edges array
 * @param num_bins Number of bins
 * @param total_count Total count (or sum of weights)
 */
template <typename scalar_t>
__global__ void normalize_density_variable_width_kernel(
    scalar_t* __restrict__ counts,
    const scalar_t* __restrict__ edges,
    int64_t num_bins,
    scalar_t total_count
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_bins) return;

    // Handle edge case
    if (total_count <= scalar_t(0)) {
        counts[idx] = scalar_t(0);
        return;
    }

    scalar_t bin_width = edges[idx + 1] - edges[idx];
    if (bin_width > scalar_t(0)) {
        counts[idx] /= (total_count * bin_width);
    } else {
        counts[idx] = scalar_t(0);
    }
}

/**
 * Compute bin edges on CPU (helper function).
 */
template <typename T>
void compute_bin_edges_cpu(
    T* edges,
    T data_min,
    T data_max,
    int64_t num_bins
) {
    T bin_width = (data_max - data_min) / static_cast<T>(num_bins);

    for (int64_t i = 0; i <= num_bins; ++i) {
        edges[i] = data_min + static_cast<T>(i) * bin_width;
    }

    // Ensure the last edge is exactly data_max to avoid floating point issues
    edges[num_bins] = data_max;
}

/**
 * Compute data range on CPU (helper function).
 */
template <typename T>
bool compute_data_range_cpu(
    const T* data,
    int64_t n,
    T* out_min,
    T* out_max
) {
    T min_val = std::numeric_limits<T>::infinity();
    T max_val = -std::numeric_limits<T>::infinity();
    bool found_valid = false;

    for (int64_t i = 0; i < n; ++i) {
        T val = data[i];
        // Skip NaN values
        if (val != val) {
            continue;
        }
        found_valid = true;
        if (val < min_val) {
            min_val = val;
        }
        if (val > max_val) {
            max_val = val;
        }
    }

    if (!found_valid) {
        *out_min = std::numeric_limits<T>::quiet_NaN();
        *out_max = std::numeric_limits<T>::quiet_NaN();
        return false;
    }

    *out_min = min_val;
    *out_max = max_val;
    return true;
}

}  // namespace

/**
 * CUDA implementation of histogram with integer bin count.
 *
 * Computes a 1D histogram of the input data.
 *
 * @param input Input tensor (any shape, will be flattened)
 * @param bins Number of equal-width bins
 * @param range Optional (min, max) range for binning
 * @param weight Optional weight tensor (same numel as input)
 * @param density If true, normalize to probability density
 * @param closed "left" for [a,b) intervals, "right" for (a,b] intervals
 * @param out_of_bounds "clamp", "ignore", or "error"
 * @return Tuple of (counts, bin_edges)
 */
std::tuple<at::Tensor, at::Tensor> histogram(
    const at::Tensor& input,
    int64_t bins,
    std::optional<c10::ArrayRef<double>> range,
    const std::optional<at::Tensor>& weight,
    bool density,
    c10::string_view closed,
    c10::string_view out_of_bounds
) {
    TORCH_CHECK(input.is_cuda(), "histogram: input must be a CUDA tensor");
    TORCH_CHECK(input.numel() > 0, "histogram: input tensor must be non-empty");
    TORCH_CHECK(bins > 0, "histogram: bins must be positive");

    // Validate weight tensor if provided
    if (weight.has_value()) {
        TORCH_CHECK(
            weight->numel() == input.numel(),
            "histogram: weight tensor must have same number of elements as input, "
            "got weight numel=", weight->numel(), ", input numel=", input.numel()
        );
    }

    c10::cuda::CUDAGuard device_guard(input.device());

    // Parse closed string
    bool closed_right = (closed == "right" || closed == "Right" || closed == "RIGHT");

    // Parse out_of_bounds string
    bool clamp = (out_of_bounds == "clamp" || out_of_bounds == "Clamp" || out_of_bounds == "CLAMP");
    bool error_mode = (out_of_bounds == "error" || out_of_bounds == "Error" || out_of_bounds == "ERROR");

    // Flatten input for 1D processing
    at::Tensor input_flat = input.flatten().contiguous();
    at::Tensor weight_flat;
    if (weight.has_value()) {
        weight_flat = weight->flatten().contiguous().to(input_flat.scalar_type());
    }

    // Determine output dtype - compute in float for Half/BFloat16
    at::ScalarType compute_dtype = input.scalar_type();
    if (compute_dtype == at::kHalf || compute_dtype == at::kBFloat16) {
        compute_dtype = at::kFloat;
    }

    // Create output tensors
    auto options = input.options().dtype(compute_dtype);
    at::Tensor counts = at::zeros({bins}, options);
    at::Tensor edges = at::empty({bins + 1}, options);
    at::Tensor total_tensor = at::zeros({1}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_flat.scalar_type(),
        "histogram_cuda",
        [&]() {
            using compute_t = std::conditional_t<
                std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>,
                float,
                scalar_t
            >;

            int64_t n = input_flat.numel();

            // Copy data to CPU to compute range (this is necessary for determining
            // data min/max unless range is explicitly provided)
            at::Tensor input_cpu = input_flat.cpu();
            const scalar_t* input_cpu_ptr = input_cpu.data_ptr<scalar_t>();

            // Convert to compute type if needed
            std::vector<compute_t> input_compute_vec;
            const compute_t* data_ptr;

            if constexpr (std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
                input_compute_vec.resize(n);
                for (int64_t i = 0; i < n; ++i) {
                    input_compute_vec[i] = static_cast<compute_t>(input_cpu_ptr[i]);
                }
                data_ptr = input_compute_vec.data();
            } else {
                data_ptr = reinterpret_cast<const compute_t*>(input_cpu_ptr);
            }

            // Determine range
            compute_t data_min, data_max;
            if (range.has_value() && range->size() == 2) {
                data_min = static_cast<compute_t>((*range)[0]);
                data_max = static_cast<compute_t>((*range)[1]);
            } else {
                // Compute range from data
                bool valid = compute_data_range_cpu<compute_t>(
                    data_ptr, n, &data_min, &data_max
                );
                TORCH_CHECK(valid, "histogram: all input values are NaN");

                // Handle case where all values are the same
                if (data_min == data_max) {
                    data_min = data_min - compute_t(0.5);
                    data_max = data_max + compute_t(0.5);
                }
            }

            TORCH_CHECK(
                data_min < data_max,
                "histogram: range minimum must be less than maximum, got min=",
                data_min, ", max=", data_max
            );

            // Check for out-of-bounds values in error mode
            if (error_mode) {
                for (int64_t i = 0; i < n; ++i) {
                    compute_t val = data_ptr[i];
                    // Skip NaN
                    if (val != val) continue;

                    bool out_of_range = (val < data_min || val > data_max);
                    TORCH_CHECK(
                        !out_of_range,
                        "histogram: value ", val, " is outside the specified range [",
                        data_min, ", ", data_max, "]"
                    );
                }
            }

            // Compute bin edges on CPU
            std::vector<compute_t> edges_cpu(bins + 1);
            compute_bin_edges_cpu<compute_t>(
                edges_cpu.data(), data_min, data_max, bins
            );

            // Copy edges to GPU
            compute_t* edges_ptr = edges.data_ptr<compute_t>();
            cudaMemcpyAsync(
                edges_ptr,
                edges_cpu.data(),
                (bins + 1) * sizeof(compute_t),
                cudaMemcpyHostToDevice,
                stream
            );

            // Get data pointers for kernel
            compute_t* counts_ptr = counts.data_ptr<compute_t>();
            compute_t* total_ptr = total_tensor.data_ptr<compute_t>();

            // Handle weights
            const compute_t* weights_ptr = nullptr;
            at::Tensor weight_converted;
            if (weight.has_value()) {
                if constexpr (std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
                    weight_converted = weight_flat.to(at::kFloat);
                    weights_ptr = weight_converted.data_ptr<compute_t>();
                } else {
                    weights_ptr = weight_flat.data_ptr<compute_t>();
                }
            }

            // Get input data pointer (converted to compute type on GPU)
            const compute_t* input_gpu_ptr;
            at::Tensor input_converted;
            if constexpr (std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
                input_converted = input_flat.to(at::kFloat);
                input_gpu_ptr = input_converted.data_ptr<compute_t>();
            } else {
                input_gpu_ptr = input_flat.data_ptr<compute_t>();
            }

            // Launch histogram kernel
            int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            histogram_kernel<compute_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input_gpu_ptr,
                weights_ptr,
                edges_ptr,
                counts_ptr,
                total_ptr,
                n,
                bins,
                closed_right,
                clamp
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Apply density normalization if requested
            if (density) {
                // Synchronize to get total count
                cudaStreamSynchronize(stream);
                compute_t total_count = total_tensor.cpu().item<compute_t>();

                compute_t bin_width = (data_max - data_min) / static_cast<compute_t>(bins);

                int num_blocks_norm = (bins + BLOCK_SIZE - 1) / BLOCK_SIZE;
                normalize_density_kernel<compute_t><<<num_blocks_norm, BLOCK_SIZE, 0, stream>>>(
                    counts_ptr,
                    bins,
                    total_count,
                    bin_width
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    );

    // Convert output to input dtype if input was Half/BFloat16
    if (input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16) {
        counts = counts.to(input.scalar_type());
        edges = edges.to(input.scalar_type());
    }

    return std::make_tuple(counts, edges);
}

/**
 * CUDA implementation of histogram with explicit bin edges.
 *
 * Computes a 1D histogram of the input data using pre-specified bin edges.
 *
 * @param input Input tensor (any shape, will be flattened)
 * @param bins Tensor of bin edges (must be monotonically increasing)
 * @param weight Optional weight tensor (same numel as input)
 * @param density If true, normalize to probability density
 * @param closed "left" for [a,b) intervals, "right" for (a,b] intervals
 * @param out_of_bounds "clamp", "ignore", or "error"
 * @return Tuple of (counts, bin_edges)
 */
std::tuple<at::Tensor, at::Tensor> histogram_edges(
    const at::Tensor& input,
    const at::Tensor& bins,
    const std::optional<at::Tensor>& weight,
    bool density,
    c10::string_view closed,
    c10::string_view out_of_bounds
) {
    TORCH_CHECK(input.is_cuda(), "histogram: input must be a CUDA tensor");
    TORCH_CHECK(input.numel() > 0, "histogram: input tensor must be non-empty");
    TORCH_CHECK(bins.dim() == 1, "histogram: bins must be 1-dimensional");
    TORCH_CHECK(bins.numel() >= 2, "histogram: bins must have at least 2 edges");

    int64_t num_bins = bins.numel() - 1;

    // Validate weight tensor if provided
    if (weight.has_value()) {
        TORCH_CHECK(
            weight->numel() == input.numel(),
            "histogram: weight tensor must have same number of elements as input, "
            "got weight numel=", weight->numel(), ", input numel=", input.numel()
        );
    }

    c10::cuda::CUDAGuard device_guard(input.device());

    // Parse closed string
    bool closed_right = (closed == "right" || closed == "Right" || closed == "RIGHT");

    // Parse out_of_bounds string
    bool clamp = (out_of_bounds == "clamp" || out_of_bounds == "Clamp" || out_of_bounds == "CLAMP");
    bool error_mode = (out_of_bounds == "error" || out_of_bounds == "Error" || out_of_bounds == "ERROR");

    // Flatten input for 1D processing
    at::Tensor input_flat = input.flatten().contiguous();
    at::Tensor bins_contig = bins.contiguous();
    at::Tensor weight_flat;
    if (weight.has_value()) {
        weight_flat = weight->flatten().contiguous().to(input_flat.scalar_type());
    }

    // Determine output dtype - compute in float for Half/BFloat16
    at::ScalarType compute_dtype = input.scalar_type();
    if (compute_dtype == at::kHalf || compute_dtype == at::kBFloat16) {
        compute_dtype = at::kFloat;
    }

    // Create output tensors
    auto options = input.options().dtype(compute_dtype);
    at::Tensor counts = at::zeros({num_bins}, options);
    at::Tensor edges = bins_contig.to(compute_dtype).clone();
    at::Tensor total_tensor = at::zeros({1}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_flat.scalar_type(),
        "histogram_edges_cuda",
        [&]() {
            using compute_t = std::conditional_t<
                std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>,
                float,
                scalar_t
            >;

            int64_t n = input_flat.numel();

            // Get edges pointer
            compute_t* edges_ptr = edges.data_ptr<compute_t>();

            // Copy edges to CPU for validation
            at::Tensor edges_cpu = edges.cpu();
            compute_t* edges_cpu_ptr = edges_cpu.data_ptr<compute_t>();

            // Verify edges are monotonically increasing
            for (int64_t i = 0; i < num_bins; ++i) {
                TORCH_CHECK(
                    edges_cpu_ptr[i] < edges_cpu_ptr[i + 1],
                    "histogram: bin edges must be monotonically increasing, "
                    "got edges[", i, "]=", edges_cpu_ptr[i], " >= edges[", i + 1, "]=", edges_cpu_ptr[i + 1]
                );
            }

            compute_t data_min = edges_cpu_ptr[0];
            compute_t data_max = edges_cpu_ptr[num_bins];

            // Check for out-of-bounds values in error mode
            if (error_mode) {
                at::Tensor input_cpu = input_flat.cpu();
                const scalar_t* input_cpu_ptr = input_cpu.data_ptr<scalar_t>();

                for (int64_t i = 0; i < n; ++i) {
                    compute_t val = static_cast<compute_t>(input_cpu_ptr[i]);
                    // Skip NaN
                    if (val != val) continue;

                    bool out_of_range = (val < data_min || val > data_max);
                    TORCH_CHECK(
                        !out_of_range,
                        "histogram: value ", val, " is outside the bin edge range [",
                        data_min, ", ", data_max, "]"
                    );
                }
            }

            // Get data pointers for kernel
            compute_t* counts_ptr = counts.data_ptr<compute_t>();
            compute_t* total_ptr = total_tensor.data_ptr<compute_t>();

            // Handle weights
            const compute_t* weights_ptr = nullptr;
            at::Tensor weight_converted;
            if (weight.has_value()) {
                if constexpr (std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
                    weight_converted = weight_flat.to(at::kFloat);
                    weights_ptr = weight_converted.data_ptr<compute_t>();
                } else {
                    weights_ptr = weight_flat.data_ptr<compute_t>();
                }
            }

            // Get input data pointer (converted to compute type on GPU)
            const compute_t* input_gpu_ptr;
            at::Tensor input_converted;
            if constexpr (std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
                input_converted = input_flat.to(at::kFloat);
                input_gpu_ptr = input_converted.data_ptr<compute_t>();
            } else {
                input_gpu_ptr = input_flat.data_ptr<compute_t>();
            }

            // Launch histogram kernel
            int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            histogram_kernel<compute_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                input_gpu_ptr,
                weights_ptr,
                edges_ptr,
                counts_ptr,
                total_ptr,
                n,
                num_bins,
                closed_right,
                clamp
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Apply density normalization if requested (variable bin widths)
            if (density) {
                // Synchronize to get total count
                cudaStreamSynchronize(stream);
                compute_t total_count = total_tensor.cpu().item<compute_t>();

                int num_blocks_norm = (num_bins + BLOCK_SIZE - 1) / BLOCK_SIZE;
                normalize_density_variable_width_kernel<compute_t><<<num_blocks_norm, BLOCK_SIZE, 0, stream>>>(
                    counts_ptr,
                    edges_ptr,
                    num_bins,
                    total_count
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    );

    // Convert output to input dtype if input was Half/BFloat16
    if (input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16) {
        counts = counts.to(input.scalar_type());
        edges = edges.to(input.scalar_type());
    }

    return std::make_tuple(counts, edges);
}

}  // namespace torchscience::cuda::descriptive

TORCH_LIBRARY_IMPL(torchscience, CUDA, module) {
    module.impl(
        "histogram",
        &torchscience::cuda::descriptive::histogram
    );

    module.impl(
        "histogram_edges",
        &torchscience::cuda::descriptive::histogram_edges
    );
}
