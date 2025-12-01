#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../kernel/coding/morton.h"

namespace torchscience::cpu::coding {

inline at::Tensor morton_encode(const at::Tensor& coordinates) {
    TORCH_CHECK(
        coordinates.dim() >= 1,
        "morton_encode: coordinates must be at least 1D, got ", coordinates.dim(), "D"
    );

    int64_t n_dims = coordinates.size(-1);
    TORCH_CHECK(
        n_dims == 2 || n_dims == 3,
        "morton_encode: last dimension must be 2 or 3, got ", n_dims
    );

    TORCH_CHECK(
        coordinates.scalar_type() == at::kLong || coordinates.scalar_type() == at::kInt,
        "morton_encode: coordinates must be integer type (int32 or int64), got ",
        coordinates.scalar_type()
    );

    // Compute output shape (remove last dimension)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < coordinates.dim() - 1; ++i) {
        output_shape.push_back(coordinates.size(i));
    }
    if (output_shape.empty()) {
        output_shape.push_back(1);
    }

    // Create output tensor
    at::Tensor output = at::empty(
        output_shape,
        coordinates.options().dtype(at::kLong)
    );

    // Make input contiguous
    at::Tensor coords = coordinates.contiguous();

    // Compute total number of points
    int64_t n_points = 1;
    for (int64_t i = 0; i < coords.dim() - 1; ++i) {
        n_points *= coords.size(i);
    }

    // Flatten for processing
    at::Tensor coords_flat = coords.view({n_points, n_dims});
    at::Tensor output_flat = output.view({n_points});

    // Process based on input dtype
    if (coords.scalar_type() == at::kLong) {
        const int64_t* coords_ptr = coords_flat.data_ptr<int64_t>();
        int64_t* output_ptr = output_flat.data_ptr<int64_t>();

        at::parallel_for(0, n_points, 1, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                if (n_dims == 3) {
                    int64_t x = coords_ptr[i * 3];
                    int64_t y = coords_ptr[i * 3 + 1];
                    int64_t z = coords_ptr[i * 3 + 2];
                    output_ptr[i] = kernel::coding::morton_encode_3d(x, y, z);
                } else {
                    int64_t x = coords_ptr[i * 2];
                    int64_t y = coords_ptr[i * 2 + 1];
                    output_ptr[i] = kernel::coding::morton_encode_2d(x, y);
                }
            }
        });
    } else {
        // int32 input
        const int32_t* coords_ptr = coords_flat.data_ptr<int32_t>();
        int64_t* output_ptr = output_flat.data_ptr<int64_t>();

        at::parallel_for(0, n_points, 1, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                if (n_dims == 3) {
                    int64_t x = static_cast<int64_t>(coords_ptr[i * 3]);
                    int64_t y = static_cast<int64_t>(coords_ptr[i * 3 + 1]);
                    int64_t z = static_cast<int64_t>(coords_ptr[i * 3 + 2]);
                    output_ptr[i] = kernel::coding::morton_encode_3d(x, y, z);
                } else {
                    int64_t x = static_cast<int64_t>(coords_ptr[i * 2]);
                    int64_t y = static_cast<int64_t>(coords_ptr[i * 2 + 1]);
                    output_ptr[i] = kernel::coding::morton_encode_2d(x, y);
                }
            }
        });
    }

    // Reshape output if input was scalar-like
    if (coordinates.dim() == 1) {
        return output.squeeze(0);
    }
    return output;
}

inline at::Tensor morton_decode(const at::Tensor& codes, int64_t dimensions) {
    TORCH_CHECK(
        dimensions == 2 || dimensions == 3,
        "morton_decode: dimensions must be 2 or 3, got ", dimensions
    );

    TORCH_CHECK(
        codes.scalar_type() == at::kLong,
        "morton_decode: codes must be int64, got ", codes.scalar_type()
    );

    // Compute output shape (add last dimension)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < codes.dim(); ++i) {
        output_shape.push_back(codes.size(i));
    }
    output_shape.push_back(dimensions);

    // Handle scalar input
    bool scalar_input = (codes.dim() == 0);
    at::Tensor codes_work = scalar_input ? codes.unsqueeze(0) : codes;
    if (scalar_input) {
        output_shape = {dimensions};
    }

    // Create output tensor
    at::Tensor output = at::empty(
        output_shape,
        codes.options().dtype(at::kLong)
    );

    // Make input contiguous
    at::Tensor codes_cont = codes_work.contiguous();

    // Compute total number of codes
    int64_t n_codes = codes_cont.numel();

    // Flatten for processing
    at::Tensor codes_flat = codes_cont.view({n_codes});
    at::Tensor output_flat = output.view({n_codes, dimensions});

    const int64_t* codes_ptr = codes_flat.data_ptr<int64_t>();
    int64_t* output_ptr = output_flat.data_ptr<int64_t>();

    at::parallel_for(0, n_codes, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            if (dimensions == 3) {
                int64_t x, y, z;
                kernel::coding::morton_decode_3d(codes_ptr[i], x, y, z);
                output_ptr[i * 3] = x;
                output_ptr[i * 3 + 1] = y;
                output_ptr[i * 3 + 2] = z;
            } else {
                int64_t x, y;
                kernel::coding::morton_decode_2d(codes_ptr[i], x, y);
                output_ptr[i * 2] = x;
                output_ptr[i * 2 + 1] = y;
            }
        }
    });

    return output;
}

}  // namespace torchscience::cpu::coding

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("morton_encode", &torchscience::cpu::coding::morton_encode);
    m.impl("morton_decode", &torchscience::cpu::coding::morton_decode);
}
