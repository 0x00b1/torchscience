#pragma once

#include <cmath>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::graphics::texture_mapping {

namespace {

// Cube mapping kernel
// Returns: face (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z), u, v
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
void cube_mapping_kernel(
    const T* direction,
    int64_t* face,
    T* u,
    T* v
) {
    T x = direction[0];
    T y = direction[1];
    T z = direction[2];

    T abs_x = std::abs(x);
    T abs_y = std::abs(y);
    T abs_z = std::abs(z);

    T max_axis;
    T uc, vc;  // Unnormalized texture coordinates

    if (abs_x >= abs_y && abs_x >= abs_z) {
        // X is dominant axis
        max_axis = abs_x;
        if (x > 0) {
            // +X face (face 0)
            *face = 0;
            uc = -z;
            vc = -y;
        } else {
            // -X face (face 1)
            *face = 1;
            uc = z;
            vc = -y;
        }
    } else if (abs_y >= abs_x && abs_y >= abs_z) {
        // Y is dominant axis
        max_axis = abs_y;
        if (y > 0) {
            // +Y face (face 2)
            *face = 2;
            uc = x;
            vc = z;
        } else {
            // -Y face (face 3)
            *face = 3;
            uc = x;
            vc = -z;
        }
    } else {
        // Z is dominant axis
        max_axis = abs_z;
        if (z > 0) {
            // +Z face (face 4)
            *face = 4;
            uc = x;
            vc = -y;
        } else {
            // -Z face (face 5)
            *face = 5;
            uc = -x;
            vc = -y;
        }
    }

    // Convert to [0, 1] UV coordinates
    // uc, vc are in range [-max_axis, max_axis]
    // Map to [0, 1]: (val / max_axis + 1) / 2
    *u = (uc / max_axis + T(1)) * T(0.5);
    *v = (vc / max_axis + T(1)) * T(0.5);
}

}  // namespace

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> cube_mapping(
    const at::Tensor& direction
) {
    TORCH_CHECK(direction.size(-1) == 3, "cube_mapping: direction must have last dimension 3");

    auto direction_contig = direction.contiguous();

    // Compute output shape (batch dimensions)
    std::vector<int64_t> batch_shape;
    for (int64_t i = 0; i < direction.dim() - 1; ++i) {
        batch_shape.push_back(direction.size(i));
    }

    auto face = at::empty(batch_shape, direction.options().dtype(at::kLong));
    auto u = at::empty(batch_shape, direction.options());
    auto v = at::empty(batch_shape, direction.options());

    int64_t num_elements = face.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        direction.scalar_type(), "cube_mapping_cpu", [&] {
            const scalar_t* direction_ptr = direction_contig.data_ptr<scalar_t>();
            int64_t* face_ptr = face.data_ptr<int64_t>();
            scalar_t* u_ptr = u.data_ptr<scalar_t>();
            scalar_t* v_ptr = v.data_ptr<scalar_t>();

            at::parallel_for(0, num_elements, 1024, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    cube_mapping_kernel(
                        direction_ptr + i * 3,
                        face_ptr + i,
                        u_ptr + i,
                        v_ptr + i
                    );
                }
            });
        }
    );

    return std::make_tuple(face, u, v);
}

}  // namespace torchscience::cpu::graphics::texture_mapping

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("cube_mapping", &torchscience::cpu::graphics::texture_mapping::cube_mapping);
}
