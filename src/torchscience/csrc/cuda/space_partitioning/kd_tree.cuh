// src/torchscience/csrc/cuda/space_partitioning/kd_tree.cuh
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace torchscience::cuda::space_partitioning {

// Parallel k-d tree construction using radix-based approach
// Algorithm: Sort points by Morton code, then build tree from sorted order
// This enables GPU-parallel construction unlike recursive CPU approach

namespace {

// D-dimensional Morton code computation
// For D dimensions, each coordinate gets floor(64/D) bits
// Bits are interleaved: dim0_bit0, dim1_bit0, ..., dimD-1_bit0, dim0_bit1, ...
template <typename scalar_t>
__device__ uint64_t morton_encode_nd(
    const scalar_t* point,
    const scalar_t* mins,
    const scalar_t* ranges,
    int64_t d
) {
    // Bits per dimension (e.g., d=3 -> 21 bits, d=4 -> 16 bits, d=8 -> 8 bits)
    int bits_per_dim = 64 / d;
    uint64_t max_val = (1ULL << bits_per_dim) - 1;

    uint64_t code = 0;

    for (int64_t dim = 0; dim < d; ++dim) {
        // Normalize to [0, 1]
        float range = static_cast<float>(ranges[dim]);
        float normalized = (range > 1e-10f)
            ? (static_cast<float>(point[dim]) - static_cast<float>(mins[dim])) / range
            : 0.0f;

        // Quantize to bits_per_dim bits
        uint64_t quantized = static_cast<uint64_t>(
            fminf(fmaxf(normalized * static_cast<float>(max_val), 0.0f),
                  static_cast<float>(max_val))
        );

        // Interleave bits: spread bits of this coordinate across the code
        // Bit i of this coordinate goes to position (i * d + dim)
        for (int bit = 0; bit < bits_per_dim; ++bit) {
            if (quantized & (1ULL << bit)) {
                code |= (1ULL << (bit * d + dim));
            }
        }
    }

    return code;
}

// Kernel to compute Morton codes for all points (D-dimensional)
template <typename scalar_t>
__global__ void compute_morton_codes_kernel(
    const scalar_t* __restrict__ points,
    uint64_t* __restrict__ morton_codes,
    int64_t* __restrict__ indices,
    const scalar_t* __restrict__ mins,
    const scalar_t* __restrict__ ranges,
    int64_t n, int64_t d
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    morton_codes[idx] = morton_encode_nd(&points[idx * d], mins, ranges, d);
    indices[idx] = idx;
}

// Delta function for Karras algorithm - finds position of highest differing bit
__device__ int delta(const uint64_t* __restrict__ morton_codes, int64_t n, int64_t i, int64_t j) {
    if (j < 0 || j >= n) return -1;
    if (morton_codes[i] == morton_codes[j]) {
        // Tie-breaker using indices when codes are equal
        return 64 + __clzll(i ^ j);
    }
    return __clzll(morton_codes[i] ^ morton_codes[j]);
}

// Determine range of keys covered by internal node i (Karras Algorithm 2)
__device__ void determine_range(
    const uint64_t* __restrict__ morton_codes,
    int64_t n,
    int64_t i,
    int64_t* first,
    int64_t* last
) {
    // Determine direction of the range
    int d_left = delta(morton_codes, n, i, i - 1);
    int d_right = delta(morton_codes, n, i, i + 1);
    int d = (d_right > d_left) ? 1 : -1;
    int d_min = (d == 1) ? d_left : d_right;

    // Find upper bound for range length
    int64_t l_max = 2;
    while (delta(morton_codes, n, i, i + l_max * d) > d_min) {
        l_max *= 2;
    }

    // Binary search for actual range length
    int64_t l = 0;
    for (int64_t t = l_max / 2; t >= 1; t /= 2) {
        if (delta(morton_codes, n, i, i + (l + t) * d) > d_min) {
            l = l + t;
        }
    }

    int64_t j = i + l * d;
    *first = (d == 1) ? i : j;
    *last = (d == 1) ? j : i;
}

// Find split position within range (Karras Algorithm 3)
__device__ int64_t find_split(
    const uint64_t* __restrict__ morton_codes,
    int64_t n,
    int64_t first,
    int64_t last
) {
    uint64_t first_code = morton_codes[first];
    uint64_t last_code = morton_codes[last];

    if (first_code == last_code) {
        return (first + last) / 2;
    }

    int common_prefix = __clzll(first_code ^ last_code);

    // Binary search for split position
    int64_t split = first;
    int64_t step = last - first;

    do {
        step = (step + 1) / 2;
        int64_t new_split = split + step;

        if (new_split < last) {
            uint64_t split_code = morton_codes[new_split];
            int split_prefix = __clzll(first_code ^ split_code);
            if (split_prefix > common_prefix) {
                split = new_split;
            }
        }
    } while (step > 1);

    return split;
}

// Kernel to build internal nodes in parallel (Karras Algorithm 1)
// Generalized for D dimensions
template <typename scalar_t>
__global__ void build_radix_tree_kernel(
    const uint64_t* __restrict__ morton_codes,
    const scalar_t* __restrict__ points,
    const int64_t* __restrict__ sorted_indices,
    int64_t* __restrict__ split_dim,
    scalar_t* __restrict__ split_val,
    int64_t* __restrict__ left,
    int64_t* __restrict__ right,
    int64_t n,
    int64_t d
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;  // n-1 internal nodes for n leaves

    // Determine range covered by this internal node
    int64_t first, last;
    determine_range(morton_codes, n, i, &first, &last);

    // Find split position
    int64_t split_pos = find_split(morton_codes, n, first, last);

    // Determine split dimension from Morton code bit position
    uint64_t first_code = morton_codes[first];
    uint64_t last_code = morton_codes[last];
    int common_prefix = __clzll(first_code ^ last_code);

    // Morton codes interleave bits: dim0_bit0, dim1_bit0, ..., dimD-1_bit0, dim0_bit1, ...
    // The split dimension is determined by which bit differs first
    int bit_pos = 63 - common_prefix;  // Position of first differing bit
    int64_t dim = bit_pos % d;  // Which dimension (correctly handles any D)

    // Get median point at split position for split value
    int64_t median_idx = sorted_indices[split_pos];
    scalar_t split_v = points[median_idx * d + dim];

    split_dim[i] = dim;
    split_val[i] = split_v;

    // Set children: internal nodes index from 0..n-2, leaves from n-1..2n-2
    // Left child
    if (first == split_pos) {
        left[i] = n - 1 + split_pos;  // Leaf node
    } else {
        left[i] = split_pos;  // Internal node
    }

    // Right child
    if (last == split_pos + 1) {
        right[i] = n - 1 + split_pos + 1;  // Leaf node
    } else {
        right[i] = split_pos + 1;  // Internal node (covers split_pos+1..last)
    }
}

// Kernel to coalesce adjacent leaves up to leaf_size (LBVH optimization)
// This respects the leaf_size parameter by grouping Morton-adjacent points
__global__ void coalesce_leaves_kernel(
    const int64_t* __restrict__ left,
    const int64_t* __restrict__ right,
    int64_t* __restrict__ coalesced_left,
    int64_t* __restrict__ coalesced_right,
    int64_t* __restrict__ leaf_starts,
    int64_t* __restrict__ leaf_counts,
    int64_t* __restrict__ new_leaf_id,  // Maps old leaf -> new coalesced leaf
    int64_t n_internal,
    int64_t n_leaves,
    int64_t leaf_size
) {
    // Coalescing is done by grouping consecutive leaves in Morton order
    // Each thread handles one coalesced leaf group
    int64_t coalesced_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_coalesced = (n_leaves + leaf_size - 1) / leaf_size;

    if (coalesced_idx >= n_coalesced) return;

    int64_t start_leaf = coalesced_idx * leaf_size;
    int64_t end_leaf = min(start_leaf + leaf_size, n_leaves);
    int64_t count = end_leaf - start_leaf;

    // Update leaf info
    leaf_starts[coalesced_idx] = start_leaf;
    leaf_counts[coalesced_idx] = count;

    // Map old leaves to new coalesced leaf
    for (int64_t i = start_leaf; i < end_leaf; ++i) {
        new_leaf_id[i] = n_internal + coalesced_idx;  // New leaf node index
    }
}

// Kernel to update parent pointers after leaf coalescing
__global__ void update_parent_pointers_kernel(
    int64_t* __restrict__ left,
    int64_t* __restrict__ right,
    const int64_t* __restrict__ new_leaf_id,
    int64_t n_internal,
    int64_t n_leaves
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_internal) return;

    // Update left child if it's a leaf
    if (left[i] >= n_internal && left[i] < n_internal + n_leaves) {
        int64_t old_leaf_idx = left[i] - n_internal;
        left[i] = new_leaf_id[old_leaf_idx];
    }

    // Update right child if it's a leaf
    if (right[i] >= n_internal && right[i] < n_internal + n_leaves) {
        int64_t old_leaf_idx = right[i] - n_internal;
        right[i] = new_leaf_id[old_leaf_idx];
    }
}

}  // anonymous namespace

// Helper: build single tree on GPU (internal use)
// Generalized for D dimensions with LBVH leaf coalescing
template <typename scalar_t>
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
kd_tree_build_single(
    const at::Tensor& points_2d,  // (n, d)
    int64_t leaf_size,
    cudaStream_t stream
) {
    int64_t n = points_2d.size(0);
    int64_t d = points_2d.size(1);

    // Compute D-dimensional bounding box
    at::Tensor mins = std::get<0>(points_2d.min(0));  // (d,)
    at::Tensor maxs = std::get<0>(points_2d.max(0));  // (d,)
    at::Tensor ranges = maxs - mins;                   // (d,)

    // Ensure contiguous for kernel access
    mins = mins.contiguous();
    ranges = ranges.contiguous();

    // Compute Morton codes
    at::Tensor morton_codes = at::empty({n}, points_2d.options().dtype(at::kLong));
    at::Tensor indices = at::empty({n}, points_2d.options().dtype(at::kLong));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    compute_morton_codes_kernel<<<blocks, threads, 0, stream>>>(
        points_2d.data_ptr<scalar_t>(),
        reinterpret_cast<uint64_t*>(morton_codes.data_ptr<int64_t>()),
        indices.data_ptr<int64_t>(),
        mins.data_ptr<scalar_t>(),
        ranges.data_ptr<scalar_t>(),
        n, d
    );

    // Sort by Morton codes
    at::Tensor sorted_order = std::get<1>(morton_codes.sort());
    at::Tensor sorted_morton = morton_codes.index_select(0, sorted_order);
    indices = indices.index_select(0, sorted_order);

    // Radix tree: n-1 internal nodes + n leaf nodes (before coalescing)
    int64_t n_internal = n - 1;
    int64_t n_leaves_original = n;

    // After coalescing: ceil(n / leaf_size) leaves
    int64_t n_leaves_coalesced = (n + leaf_size - 1) / leaf_size;
    int64_t n_nodes = n_internal + n_leaves_coalesced;

    at::Tensor split_dim_t = at::full({n_nodes}, -1, points_2d.options().dtype(at::kLong));
    at::Tensor split_val_t = at::zeros({n_nodes}, points_2d.options());
    at::Tensor left_t = at::full({n_nodes}, -1, points_2d.options().dtype(at::kLong));
    at::Tensor right_t = at::full({n_nodes}, -1, points_2d.options().dtype(at::kLong));
    at::Tensor leaf_starts_t = at::empty({n_leaves_coalesced}, points_2d.options().dtype(at::kLong));
    at::Tensor leaf_counts_t = at::empty({n_leaves_coalesced}, points_2d.options().dtype(at::kLong));

    if (n_internal > 0) {
        // Step 1: Build radix tree with original 1-point leaves
        at::Tensor left_orig = at::full({n_internal + n_leaves_original}, -1, points_2d.options().dtype(at::kLong));
        at::Tensor right_orig = at::full({n_internal + n_leaves_original}, -1, points_2d.options().dtype(at::kLong));

        build_radix_tree_kernel<scalar_t><<<(n_internal + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<uint64_t*>(sorted_morton.data_ptr<int64_t>()),
            points_2d.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            split_dim_t.data_ptr<int64_t>(),
            split_val_t.data_ptr<scalar_t>(),
            left_orig.data_ptr<int64_t>(),
            right_orig.data_ptr<int64_t>(),
            n, d
        );

        // Step 2: Coalesce leaves up to leaf_size
        at::Tensor new_leaf_id = at::empty({n_leaves_original}, points_2d.options().dtype(at::kLong));

        coalesce_leaves_kernel<<<(n_leaves_coalesced + 255) / 256, 256, 0, stream>>>(
            left_orig.data_ptr<int64_t>(),
            right_orig.data_ptr<int64_t>(),
            left_t.data_ptr<int64_t>(),
            right_t.data_ptr<int64_t>(),
            leaf_starts_t.data_ptr<int64_t>(),
            leaf_counts_t.data_ptr<int64_t>(),
            new_leaf_id.data_ptr<int64_t>(),
            n_internal,
            n_leaves_original,
            leaf_size
        );

        // Step 3: Update parent pointers to reference coalesced leaves
        // Copy internal node structure
        left_t.slice(0, 0, n_internal).copy_(left_orig.slice(0, 0, n_internal));
        right_t.slice(0, 0, n_internal).copy_(right_orig.slice(0, 0, n_internal));

        update_parent_pointers_kernel<<<(n_internal + 255) / 256, 256, 0, stream>>>(
            left_t.data_ptr<int64_t>(),
            right_t.data_ptr<int64_t>(),
            new_leaf_id.data_ptr<int64_t>(),
            n_internal,
            n_leaves_original
        );
    } else {
        // Single leaf case
        leaf_starts_t.fill_(0);
        leaf_counts_t.fill_(n);
    }

    return std::make_tuple(split_dim_t, split_val_t, left_t, right_t, indices, leaf_starts_t, leaf_counts_t);
}

// Batched CUDA tree build - builds all trees then pads to max size
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
kd_tree_build_batched(
    const at::Tensor& points,
    int64_t leaf_size
) {
    TORCH_CHECK(points.dim() == 3, "kd_tree_build_batched: points must be 3D (B, n, d)");
    TORCH_CHECK(leaf_size > 0, "kd_tree_build_batched: leaf_size must be > 0");
    TORCH_CHECK(points.is_cuda(), "kd_tree_build_batched CUDA: points must be on CUDA device");

    const at::cuda::CUDAGuard device_guard(points.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int64_t B = points.size(0);
    int64_t n = points.size(1);
    int64_t d = points.size(2);

    if (B == 0 || n == 0) {
        return std::make_tuple(
            points.clone(),
            at::empty({B, 0}, points.options().dtype(at::kLong)),
            at::empty({B, 0}, points.options()),
            at::empty({B, 0}, points.options().dtype(at::kLong)),
            at::empty({B, 0}, points.options().dtype(at::kLong)),
            at::empty({B, n}, points.options().dtype(at::kLong)),
            at::empty({B, 0}, points.options().dtype(at::kLong)),
            at::empty({B, 0}, points.options().dtype(at::kLong))
        );
    }

    at::Tensor points_contig = points.contiguous();

    // With LBVH coalescing: n-1 internal nodes + ceil(n/leaf_size) leaves
    int64_t n_internal = n - 1;
    int64_t n_leaves = (n + leaf_size - 1) / leaf_size;
    int64_t n_nodes = n_internal + n_leaves;

    // Pre-allocate output tensors
    at::Tensor split_dim_out = at::full({B, n_nodes}, -1, points.options().dtype(at::kLong));
    at::Tensor split_val_out = at::zeros({B, n_nodes}, points.options());
    at::Tensor left_out = at::full({B, n_nodes}, -1, points.options().dtype(at::kLong));
    at::Tensor right_out = at::full({B, n_nodes}, -1, points.options().dtype(at::kLong));
    at::Tensor indices_out = at::empty({B, n}, points.options().dtype(at::kLong));
    at::Tensor leaf_starts_out = at::empty({B, n_leaves}, points.options().dtype(at::kLong));
    at::Tensor leaf_counts_out = at::empty({B, n_leaves}, points.options().dtype(at::kLong));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        points.scalar_type(),
        "kd_tree_build_batched_cuda",
        [&]() {
            // Build trees sequentially on GPU (each tree build is parallel internally)
            // TODO(perf): Use CUDA stream pool for concurrent tree builds when B > 1.
            // Each tree build is independent and could run on separate streams,
            // enabling overlap of Morton code computation, sorting, and tree construction.
            // Priority: medium (current implementation is already GPU-parallel within each tree)
            for (int64_t b = 0; b < B; ++b) {
                auto result = kd_tree_build_single<scalar_t>(points_contig[b], leaf_size, stream);

                // Copy results to output tensors
                split_dim_out[b].copy_(std::get<0>(result));
                split_val_out[b].copy_(std::get<1>(result));
                left_out[b].copy_(std::get<2>(result));
                right_out[b].copy_(std::get<3>(result));
                indices_out[b].copy_(std::get<4>(result));
                leaf_starts_out[b].copy_(std::get<5>(result));
                leaf_counts_out[b].copy_(std::get<6>(result));
            }
        }
    );

    return std::make_tuple(
        points_contig,
        split_dim_out,
        split_val_out,
        left_out,
        right_out,
        indices_out,
        leaf_starts_out,
        leaf_counts_out
    );
}

}  // namespace torchscience::cuda::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
    m.impl("kd_tree_build_batched", torchscience::cuda::space_partitioning::kd_tree_build_batched);
}
