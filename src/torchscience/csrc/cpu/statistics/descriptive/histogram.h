#pragma once

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::descriptive {

namespace {

// ============================================================================
// Bin Edge Computation
// ============================================================================

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void compute_bin_edges(
    T* edges,
    T data_min,
    T data_max,
    int64_t num_bins
) {
    T bin_width = (data_max - data_min) / static_cast<T>(num_bins);

    for (int64_t i = 0; i <= num_bins; ++i) {
        edges[i] = data_min + static_cast<T>(i) * bin_width;
    }

    edges[num_bins] = data_max;
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T compute_bin_width(T data_min, T data_max, int64_t num_bins) {
    return (data_max - data_min) / static_cast<T>(num_bins);
}

// ============================================================================
// Bin Finding
// ============================================================================

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
int64_t find_bin(
    T value,
    const T* edges,
    int64_t num_bins,
    bool closed_right
) {
    if (value != value) {
        return -2;
    }

    T left_edge = edges[0];
    T right_edge = edges[num_bins];

    if (closed_right) {
        if (value <= left_edge) {
            if (value == left_edge) {
                return 0;
            }
            return -1;
        }
        if (value > right_edge) {
            return num_bins;
        }
    } else {
        if (value < left_edge) {
            return -1;
        }
        if (value >= right_edge) {
            if (value == right_edge) {
                return num_bins - 1;
            }
            return num_bins;
        }
    }

    int64_t lo = 0;
    int64_t hi = num_bins;

    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;

        if (closed_right) {
            if (edges[mid + 1] < value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        } else {
            if (edges[mid + 1] <= value) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
    }

    return lo;
}

// ============================================================================
// Main 1D Histogram Computation
// ============================================================================

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T histogram_1d(
    const T* data,
    int64_t n,
    T* counts,
    const T* edges,
    int64_t num_bins,
    const T* weights,
    bool closed_right,
    bool clamp_out_of_bounds
) {
    for (int64_t i = 0; i < num_bins; ++i) {
        counts[i] = T(0);
    }

    T total = T(0);

    for (int64_t i = 0; i < n; ++i) {
        T value = data[i];
        T weight = weights ? weights[i] : T(1);

        int64_t bin_idx = find_bin(value, edges, num_bins, closed_right);

        if (bin_idx == -2) {
            continue;
        }

        if (bin_idx == -1) {
            if (clamp_out_of_bounds) {
                bin_idx = 0;
            } else {
                continue;
            }
        } else if (bin_idx == num_bins) {
            if (clamp_out_of_bounds) {
                bin_idx = num_bins - 1;
            } else {
                continue;
            }
        }

        counts[bin_idx] += weight;
        total += weight;
    }

    return total;
}

// ============================================================================
// Density Normalization
// ============================================================================

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void normalize_density(
    T* counts,
    int64_t num_bins,
    T total_count,
    T bin_width
) {
    if (total_count <= T(0) || bin_width <= T(0)) {
        for (int64_t i = 0; i < num_bins; ++i) {
            counts[i] = T(0);
        }
        return;
    }

    T normalizer = total_count * bin_width;

    for (int64_t i = 0; i < num_bins; ++i) {
        counts[i] /= normalizer;
    }
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void normalize_density_variable_width(
    T* counts,
    const T* edges,
    int64_t num_bins,
    T total_count
) {
    if (total_count <= T(0)) {
        for (int64_t i = 0; i < num_bins; ++i) {
            counts[i] = T(0);
        }
        return;
    }

    for (int64_t i = 0; i < num_bins; ++i) {
        T bin_width = edges[i + 1] - edges[i];
        if (bin_width > T(0)) {
            counts[i] /= (total_count * bin_width);
        } else {
            counts[i] = T(0);
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
bool compute_data_range(
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

}  // anonymous namespace

/**
 * CPU implementation of histogram with integer bin count.
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
inline std::tuple<at::Tensor, at::Tensor> histogram(
    const at::Tensor& input,
    int64_t bins,
    std::optional<c10::ArrayRef<double>> range,
    const std::optional<at::Tensor>& weight,
    bool density,
    c10::string_view closed,
    c10::string_view out_of_bounds
) {
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

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        input_flat.scalar_type(),
        "histogram_cpu",
        [&]() {
            using compute_t = std::conditional_t<
                std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>,
                float,
                scalar_t
            >;

            const scalar_t* input_ptr = input_flat.data_ptr<scalar_t>();
            int64_t n = input_flat.numel();

            // Convert input to compute type if needed
            std::vector<compute_t> input_compute;
            std::vector<compute_t> weight_compute;
            const compute_t* data_ptr;
            const compute_t* weights_ptr = nullptr;

            if constexpr (std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
                input_compute.resize(n);
                for (int64_t i = 0; i < n; ++i) {
                    input_compute[i] = static_cast<compute_t>(input_ptr[i]);
                }
                data_ptr = input_compute.data();

                if (weight.has_value()) {
                    const scalar_t* weight_ptr = weight_flat.data_ptr<scalar_t>();
                    weight_compute.resize(n);
                    for (int64_t i = 0; i < n; ++i) {
                        weight_compute[i] = static_cast<compute_t>(weight_ptr[i]);
                    }
                    weights_ptr = weight_compute.data();
                }
            } else {
                data_ptr = reinterpret_cast<const compute_t*>(input_ptr);
                if (weight.has_value()) {
                    weights_ptr = weight_flat.data_ptr<compute_t>();
                }
            }

            // Determine range
            compute_t data_min, data_max;
            if (range.has_value() && range->size() == 2) {
                data_min = static_cast<compute_t>((*range)[0]);
                data_max = static_cast<compute_t>((*range)[1]);
            } else {
                // Compute range from data
                bool valid = compute_data_range<compute_t>(
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

                    bool out_of_range = false;
                    if (closed_right) {
                        out_of_range = (val < data_min || val > data_max);
                    } else {
                        out_of_range = (val < data_min || val > data_max);
                    }
                    TORCH_CHECK(
                        !out_of_range,
                        "histogram: value ", val, " is outside the specified range [",
                        data_min, ", ", data_max, "]"
                    );
                }
            }

            // Compute bin edges
            compute_t* edges_ptr = edges.data_ptr<compute_t>();
            compute_bin_edges<compute_t>(
                edges_ptr, data_min, data_max, bins
            );

            // Compute histogram
            compute_t* counts_ptr = counts.data_ptr<compute_t>();
            compute_t total = histogram_1d<compute_t>(
                data_ptr,
                n,
                counts_ptr,
                edges_ptr,
                bins,
                weights_ptr,
                closed_right,
                clamp  // clamp if not error mode and clamp specified
            );

            // Apply density normalization if requested
            if (density) {
                compute_t bin_width = compute_bin_width<compute_t>(
                    data_min, data_max, bins
                );
                normalize_density<compute_t>(
                    counts_ptr, bins, total, bin_width
                );
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
 * CPU implementation of histogram with explicit bin edges.
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
inline std::tuple<at::Tensor, at::Tensor> histogram_edges(
    const at::Tensor& input,
    const at::Tensor& bins,
    const std::optional<at::Tensor>& weight,
    bool density,
    c10::string_view closed,
    c10::string_view out_of_bounds
) {
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

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        input_flat.scalar_type(),
        "histogram_edges_cpu",
        [&]() {
            using compute_t = std::conditional_t<
                std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>,
                float,
                scalar_t
            >;

            const scalar_t* input_ptr = input_flat.data_ptr<scalar_t>();
            int64_t n = input_flat.numel();

            // Convert input to compute type if needed
            std::vector<compute_t> input_compute;
            std::vector<compute_t> weight_compute;
            const compute_t* data_ptr;
            const compute_t* weights_ptr = nullptr;

            if constexpr (std::is_same_v<scalar_t, at::Half> || std::is_same_v<scalar_t, at::BFloat16>) {
                input_compute.resize(n);
                for (int64_t i = 0; i < n; ++i) {
                    input_compute[i] = static_cast<compute_t>(input_ptr[i]);
                }
                data_ptr = input_compute.data();

                if (weight.has_value()) {
                    const scalar_t* weight_ptr = weight_flat.data_ptr<scalar_t>();
                    weight_compute.resize(n);
                    for (int64_t i = 0; i < n; ++i) {
                        weight_compute[i] = static_cast<compute_t>(weight_ptr[i]);
                    }
                    weights_ptr = weight_compute.data();
                }
            } else {
                data_ptr = reinterpret_cast<const compute_t*>(input_ptr);
                if (weight.has_value()) {
                    weights_ptr = weight_flat.data_ptr<compute_t>();
                }
            }

            // Get edges pointer
            compute_t* edges_ptr = edges.data_ptr<compute_t>();

            // Verify edges are monotonically increasing
            for (int64_t i = 0; i < num_bins; ++i) {
                TORCH_CHECK(
                    edges_ptr[i] < edges_ptr[i + 1],
                    "histogram: bin edges must be monotonically increasing, "
                    "got edges[", i, "]=", edges_ptr[i], " >= edges[", i + 1, "]=", edges_ptr[i + 1]
                );
            }

            compute_t data_min = edges_ptr[0];
            compute_t data_max = edges_ptr[num_bins];

            // Check for out-of-bounds values in error mode
            if (error_mode) {
                for (int64_t i = 0; i < n; ++i) {
                    compute_t val = data_ptr[i];
                    // Skip NaN
                    if (val != val) continue;

                    bool out_of_range = false;
                    if (closed_right) {
                        out_of_range = (val < data_min || val > data_max);
                    } else {
                        out_of_range = (val < data_min || val > data_max);
                    }
                    TORCH_CHECK(
                        !out_of_range,
                        "histogram: value ", val, " is outside the bin edge range [",
                        data_min, ", ", data_max, "]"
                    );
                }
            }

            // Compute histogram
            compute_t* counts_ptr = counts.data_ptr<compute_t>();
            compute_t total = histogram_1d<compute_t>(
                data_ptr,
                n,
                counts_ptr,
                edges_ptr,
                num_bins,
                weights_ptr,
                closed_right,
                clamp
            );

            // Apply density normalization if requested (variable bin widths)
            if (density) {
                normalize_density_variable_width<compute_t>(
                    counts_ptr, edges_ptr, num_bins, total
                );
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

}  // namespace torchscience::cpu::descriptive

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl(
        "histogram",
        &torchscience::cpu::descriptive::histogram
    );

    module.impl(
        "histogram_edges",
        &torchscience::cpu::descriptive::histogram_edges
    );
}
