#pragma once

namespace torchscience::kernel::privacy {

template <typename scalar_t>
inline scalar_t gaussian_mechanism_forward(scalar_t x, scalar_t noise, scalar_t sigma) {
    return x + sigma * noise;
}

}  // namespace torchscience::kernel::privacy
