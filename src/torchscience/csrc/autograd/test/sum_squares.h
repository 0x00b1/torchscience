// src/torchscience/csrc/autograd/test/sum_squares.h
#pragma once

#include "../reduction_macros.h"

TORCHSCIENCE_AUTOGRAD_DIM_REDUCTION_UNARY_OPERATOR(
    test,
    sum_squares,
    SumSquares,
    input
)
