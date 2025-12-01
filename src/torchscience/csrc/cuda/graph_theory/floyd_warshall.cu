// src/torchscience/csrc/cuda/graph_theory/floyd_warshall.cu
#include <cmath>
#include <limits>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace torchscience::cuda::graph_theory {

namespace {

constexpr int BLOCK_SIZE = 32;

template <typename scalar_t>
__global__ void floyd_warshall_init_kernel(
    scalar_t* __restrict__ dist,
    int64_t* __restrict__ pred,
    int64_t N,
    int64_t batch_offset,
    bool directed
) {
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N) return;

    const scalar_t inf = std::numeric_limits<scalar_t>::infinity();
    int64_t idx = batch_offset + i * N + j;

    // Symmetrize if undirected: each thread reads both (i,j) and (j,i),
    // computes the min, and writes ONLY to its own position (i,j).
    // This eliminates the race condition where multiple threads could
    // write to the same location when N > BLOCK_SIZE.
    if (!directed) {
        scalar_t val_ij = dist[idx];
        scalar_t val_ji = dist[batch_offset + j * N + i];
        dist[idx] = min(val_ij, val_ji);
    }

    // Initialize predecessors
    if (i == j) {
        dist[idx] = scalar_t(0);
        pred[idx] = -1;
    } else if (dist[idx] < inf) {
        pred[idx] = i;
    } else {
        pred[idx] = -1;
    }
}

template <typename scalar_t>
__global__ void floyd_warshall_phase1_kernel(
    scalar_t* __restrict__ dist,
    int64_t* __restrict__ pred,
    int64_t N,
    int64_t k_block,
    int64_t batch_offset
) {
    __shared__ scalar_t s_dist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int64_t s_pred[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int64_t i = k_block * BLOCK_SIZE + ty;
    int64_t j = k_block * BLOCK_SIZE + tx;

    const scalar_t inf = std::numeric_limits<scalar_t>::infinity();

    // Load tile into shared memory
    if (i < N && j < N) {
        s_dist[ty][tx] = dist[batch_offset + i * N + j];
        s_pred[ty][tx] = pred[batch_offset + i * N + j];
    } else {
        s_dist[ty][tx] = inf;
        s_pred[ty][tx] = -1;
    }
    __syncthreads();

    // Process all k within this block
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        scalar_t dist_ik = s_dist[ty][k];
        scalar_t dist_kj = s_dist[k][tx];

        if (dist_ik < inf && dist_kj < inf) {
            scalar_t new_dist = dist_ik + dist_kj;
            if (new_dist < s_dist[ty][tx]) {
                s_dist[ty][tx] = new_dist;
                s_pred[ty][tx] = s_pred[k][tx];
            }
        }
        __syncthreads();
    }

    // Write back
    if (i < N && j < N) {
        dist[batch_offset + i * N + j] = s_dist[ty][tx];
        pred[batch_offset + i * N + j] = s_pred[ty][tx];
    }
}

template <typename scalar_t>
__global__ void floyd_warshall_phase2_row_kernel(
    scalar_t* __restrict__ dist,
    int64_t* __restrict__ pred,
    int64_t N,
    int64_t k_block,
    int64_t batch_offset
) {
    __shared__ scalar_t s_pivot[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t s_current[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int64_t s_pred_current[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int64_t block_j = blockIdx.x;

    // Skip diagonal block
    if (block_j >= k_block) block_j++;

    int64_t i = k_block * BLOCK_SIZE + ty;
    int64_t j = block_j * BLOCK_SIZE + tx;
    int64_t pivot_j = k_block * BLOCK_SIZE + tx;

    const scalar_t inf = std::numeric_limits<scalar_t>::infinity();

    // Load pivot tile (k, k)
    if (i < N && pivot_j < N) {
        s_pivot[ty][tx] = dist[batch_offset + i * N + pivot_j];
    } else {
        s_pivot[ty][tx] = inf;
    }

    // Load current tile (k, block_j)
    if (i < N && j < N) {
        s_current[ty][tx] = dist[batch_offset + i * N + j];
        s_pred_current[ty][tx] = pred[batch_offset + i * N + j];
    } else {
        s_current[ty][tx] = inf;
        s_pred_current[ty][tx] = -1;
    }
    __syncthreads();

    // Process
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        scalar_t dist_ik = s_pivot[ty][k];
        scalar_t dist_kj = s_current[k][tx];

        if (dist_ik < inf && dist_kj < inf) {
            scalar_t new_dist = dist_ik + dist_kj;
            if (new_dist < s_current[ty][tx]) {
                s_current[ty][tx] = new_dist;
                s_pred_current[ty][tx] = s_pred_current[k][tx];
            }
        }
        __syncthreads();
    }

    // Write back
    if (i < N && j < N) {
        dist[batch_offset + i * N + j] = s_current[ty][tx];
        pred[batch_offset + i * N + j] = s_pred_current[ty][tx];
    }
}

template <typename scalar_t>
__global__ void floyd_warshall_phase2_col_kernel(
    scalar_t* __restrict__ dist,
    int64_t* __restrict__ pred,
    int64_t N,
    int64_t k_block,
    int64_t batch_offset
) {
    __shared__ scalar_t s_pivot[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t s_current[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int64_t s_pred_pivot[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int64_t s_pred_current[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int64_t block_i = blockIdx.x;

    // Skip diagonal block
    if (block_i >= k_block) block_i++;

    int64_t i = block_i * BLOCK_SIZE + ty;
    int64_t j = k_block * BLOCK_SIZE + tx;
    int64_t pivot_i = k_block * BLOCK_SIZE + ty;

    const scalar_t inf = std::numeric_limits<scalar_t>::infinity();

    // Load pivot tile (k, k)
    if (pivot_i < N && j < N) {
        s_pivot[ty][tx] = dist[batch_offset + pivot_i * N + j];
        s_pred_pivot[ty][tx] = pred[batch_offset + pivot_i * N + j];
    } else {
        s_pivot[ty][tx] = inf;
        s_pred_pivot[ty][tx] = -1;
    }

    // Load current tile (block_i, k)
    if (i < N && j < N) {
        s_current[ty][tx] = dist[batch_offset + i * N + j];
        s_pred_current[ty][tx] = pred[batch_offset + i * N + j];
    } else {
        s_current[ty][tx] = inf;
        s_pred_current[ty][tx] = -1;
    }
    __syncthreads();

    // Process
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        scalar_t dist_ik = s_current[ty][k];
        scalar_t dist_kj = s_pivot[k][tx];

        if (dist_ik < inf && dist_kj < inf) {
            scalar_t new_dist = dist_ik + dist_kj;
            if (new_dist < s_current[ty][tx]) {
                s_current[ty][tx] = new_dist;
                s_pred_current[ty][tx] = s_pred_pivot[k][tx];
            }
        }
        __syncthreads();
    }

    // Write back
    if (i < N && j < N) {
        dist[batch_offset + i * N + j] = s_current[ty][tx];
        pred[batch_offset + i * N + j] = s_pred_current[ty][tx];
    }
}

template <typename scalar_t>
__global__ void floyd_warshall_phase3_kernel(
    scalar_t* __restrict__ dist,
    int64_t* __restrict__ pred,
    int64_t N,
    int64_t k_block,
    int64_t batch_offset
) {
    __shared__ scalar_t s_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t s_col[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int64_t s_pred_row[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int64_t block_i = blockIdx.y;
    int64_t block_j = blockIdx.x;

    // Skip row-k and column-k tiles
    if (block_i >= k_block) block_i++;
    if (block_j >= k_block) block_j++;

    int64_t i = block_i * BLOCK_SIZE + ty;
    int64_t j = block_j * BLOCK_SIZE + tx;

    const scalar_t inf = std::numeric_limits<scalar_t>::infinity();

    // Load row-k tile (k_block, block_j)
    int64_t row_i = k_block * BLOCK_SIZE + ty;
    int64_t row_j = block_j * BLOCK_SIZE + tx;
    if (row_i < N && row_j < N) {
        s_row[ty][tx] = dist[batch_offset + row_i * N + row_j];
        s_pred_row[ty][tx] = pred[batch_offset + row_i * N + row_j];
    } else {
        s_row[ty][tx] = inf;
        s_pred_row[ty][tx] = -1;
    }

    // Load column-k tile (block_i, k_block)
    int64_t col_i = block_i * BLOCK_SIZE + ty;
    int64_t col_j = k_block * BLOCK_SIZE + tx;
    if (col_i < N && col_j < N) {
        s_col[ty][tx] = dist[batch_offset + col_i * N + col_j];
    } else {
        s_col[ty][tx] = inf;
    }

    // Load current distance
    scalar_t cur_dist = inf;
    int64_t cur_pred = -1;
    if (i < N && j < N) {
        cur_dist = dist[batch_offset + i * N + j];
        cur_pred = pred[batch_offset + i * N + j];
    }
    __syncthreads();

    // Process
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        scalar_t dist_ik = s_col[ty][k];
        scalar_t dist_kj = s_row[k][tx];

        if (dist_ik < inf && dist_kj < inf) {
            scalar_t new_dist = dist_ik + dist_kj;
            if (new_dist < cur_dist) {
                cur_dist = new_dist;
                cur_pred = s_pred_row[k][tx];
            }
        }
    }

    // Write back
    if (i < N && j < N) {
        dist[batch_offset + i * N + j] = cur_dist;
        pred[batch_offset + i * N + j] = cur_pred;
    }
}

template <typename scalar_t>
__global__ void floyd_warshall_check_negative_kernel(
    const scalar_t* __restrict__ dist,
    int64_t N,
    int64_t batch_offset,
    bool* has_negative
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        if (dist[batch_offset + i * N + i] < scalar_t(0)) {
            *has_negative = true;
        }
    }
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor, bool> floyd_warshall(
    const at::Tensor& input,
    bool directed
) {
    TORCH_CHECK(
        input.dim() >= 2,
        "floyd_warshall: input must be at least 2D, got ", input.dim(), "D"
    );
    TORCH_CHECK(
        input.size(-1) == input.size(-2),
        "floyd_warshall: last two dimensions must be equal, got ",
        input.size(-2), " x ", input.size(-1)
    );
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()),
        "floyd_warshall: input must be floating-point, got ", input.scalar_type()
    );

    c10::cuda::CUDAGuard guard(input.device());

    // Handle sparse input
    at::Tensor dense_input = input.is_sparse() ? input.to_dense() : input;

    // Make contiguous copy
    at::Tensor distances = dense_input.clone().contiguous();

    int64_t N = distances.size(-1);
    int64_t batch_size = distances.numel() / (N * N);

    at::Tensor predecessors = at::empty(
        distances.sizes(),
        distances.options().dtype(at::kLong)
    ).contiguous();

    // Handle empty graph
    if (N == 0) {
        return std::make_tuple(distances, predecessors, false);
    }

    // Allocate flag for negative cycle detection
    at::Tensor has_negative_tensor = at::zeros({1}, distances.options().dtype(at::kBool));
    bool* has_negative_ptr = has_negative_tensor.data_ptr<bool>();

    int64_t num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        distances.scalar_type(),
        "floyd_warshall_cuda",
        [&] {
            scalar_t* dist_ptr = distances.data_ptr<scalar_t>();
            int64_t* pred_ptr = predecessors.data_ptr<int64_t>();

            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid_init((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                          (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

            cudaStream_t stream = at::cuda::getCurrentCUDAStream();

            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t batch_offset = b * N * N;

                // Initialize
                floyd_warshall_init_kernel<scalar_t><<<grid_init, block, 0, stream>>>(
                    dist_ptr, pred_ptr, N, batch_offset, directed
                );

                // Main loop over block iterations
                for (int64_t k = 0; k < num_blocks; ++k) {
                    // Phase 1: Diagonal tile
                    floyd_warshall_phase1_kernel<scalar_t><<<1, block, 0, stream>>>(
                        dist_ptr, pred_ptr, N, k, batch_offset
                    );

                    if (num_blocks > 1) {
                        // Phase 2: Row and column tiles
                        dim3 grid_phase2(num_blocks - 1);
                        floyd_warshall_phase2_row_kernel<scalar_t><<<grid_phase2, block, 0, stream>>>(
                            dist_ptr, pred_ptr, N, k, batch_offset
                        );
                        floyd_warshall_phase2_col_kernel<scalar_t><<<grid_phase2, block, 0, stream>>>(
                            dist_ptr, pred_ptr, N, k, batch_offset
                        );

                        // Phase 3: Remaining tiles
                        if (num_blocks > 1) {
                            dim3 grid_phase3(num_blocks - 1, num_blocks - 1);
                            floyd_warshall_phase3_kernel<scalar_t><<<grid_phase3, block, 0, stream>>>(
                                dist_ptr, pred_ptr, N, k, batch_offset
                            );
                        }
                    }
                }

                // Check for negative cycles
                dim3 grid_check((N + 255) / 256);
                floyd_warshall_check_negative_kernel<scalar_t><<<grid_check, 256, 0, stream>>>(
                    dist_ptr, N, batch_offset, has_negative_ptr
                );
            }
        }
    );

    // Sync and check result
    C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    bool has_negative_cycle = has_negative_tensor.item<bool>();

    return std::make_tuple(distances, predecessors, has_negative_cycle);
}

}  // namespace torchscience::cuda::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
    m.impl("floyd_warshall", &torchscience::cuda::graph_theory::floyd_warshall);
}
