// src/torchscience/csrc/cpu/test/sum_squares.h
#pragma once

#include "../../kernel/test/sum_squares.h"
#include "../reduction_macros.h"

// Use the reduction macro to generate CPU implementation
// Kernel is in torchscience::kernel::test namespace
TORCHSCIENCE_CPU_DIM_REDUCTION_UNARY_OPERATOR(test, sum_squares, input)
