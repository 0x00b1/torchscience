// src/torchscience/csrc/autocast/test/sum_squares.h
#pragma once

#include "../reduction_macros.h"

TORCHSCIENCE_AUTOCAST_DIM_REDUCTION_UNARY_OPERATOR(
    test,
    sum_squares,
    input
)
