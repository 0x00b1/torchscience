#pragma once

#include "operators.cuh"
#include "../../../operators/special_functions.def"

// Arity-based dispatch helper for Sparse COO CUDA
#define REGISTER_SPARSE_COO_CUDA_ARITY_1(m, name) REGISTER_SPARSE_COO_CUDA_UNARY(m, name)
#define REGISTER_SPARSE_COO_CUDA_ARITY_2(m, name) REGISTER_SPARSE_COO_CUDA_BINARY(m, name)
#define REGISTER_SPARSE_COO_CUDA_ARITY_3(m, name) REGISTER_SPARSE_COO_CUDA_TERNARY(m, name)
#define REGISTER_SPARSE_COO_CUDA_ARITY_4(m, name) REGISTER_SPARSE_COO_CUDA_QUATERNARY(m, name)

#define REGISTER_SPARSE_COO_CUDA_DISPATCH(m, name, arity) \
    REGISTER_SPARSE_COO_CUDA_ARITY_##arity(m, name)

TORCH_LIBRARY_IMPL(torchscience, SparseCUDA, m_sparse_coo_cuda_special_functions) {
    #define REGISTER_OP(name, arity, impl) REGISTER_SPARSE_COO_CUDA_DISPATCH(m_sparse_coo_cuda_special_functions, name, arity);
    TORCHSCIENCE_SPECIAL_FUNCTIONS(REGISTER_OP)
    #undef REGISTER_OP
}
