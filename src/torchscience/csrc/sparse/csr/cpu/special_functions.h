#pragma once

#include "operators.h"

TORCH_LIBRARY_IMPL(torchscience, SparseCsrCPU, m) {
    REGISTER_SPARSE_CSR_CPU_UNARY(m, gamma);
    REGISTER_SPARSE_CSR_CPU_BINARY(m, beta);
    REGISTER_SPARSE_CSR_CPU_BINARY(m, chebyshev_polynomial_t);
    REGISTER_SPARSE_CSR_CPU_TERNARY(m, incomplete_beta);
    REGISTER_SPARSE_CSR_CPU_QUATERNARY(m, hypergeometric_2_f_1);
}
