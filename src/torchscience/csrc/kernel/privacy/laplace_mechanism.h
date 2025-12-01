#pragma once

namespace torchscience::kernel::privacy {

template <typename scalar_t>
inline scalar_t laplace_mechanism_forward(scalar_t x, scalar_t noise, scalar_t b) {
    return x + b * noise;
}

}  // namespace torchscience::kernel::privacy
