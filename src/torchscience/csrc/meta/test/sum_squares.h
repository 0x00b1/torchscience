// src/torchscience/csrc/meta/test/sum_squares.h
#pragma once

#include "../reduction_macros.h"

// Use the reduction macro to generate Meta implementation
TORCHSCIENCE_META_DIM_REDUCTION_UNARY_OPERATOR(test, sum_squares, input)
