#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

// Additive Secret Sharing - split shape inference
// Returns a tensor of shape [n, secret_len]
at::Tensor additive_split(
    const at::Tensor& secret,
    const at::Tensor& randomness,
    int64_t n
) {
    TORCH_CHECK(secret.dim() == 1, "additive_split: secret must be 1D");
    TORCH_CHECK(secret.dtype() == at::kByte, "additive_split: secret must be uint8");
    TORCH_CHECK(n >= 2, "additive_split: n must be >= 2");

    int64_t secret_len = secret.size(0);
    TORCH_CHECK(randomness.numel() == (n - 1) * secret_len,
                "additive_split: randomness must have (n-1) * secret_len bytes");

    return at::empty({n, secret_len}, secret.options());
}

// Additive Secret Sharing - reconstruct shape inference
// Returns a tensor of shape [secret_len]
at::Tensor additive_reconstruct(
    const at::Tensor& shares
) {
    TORCH_CHECK(shares.dim() == 2, "additive_reconstruct: shares must be 2D [n, secret_len]");
    TORCH_CHECK(shares.dtype() == at::kByte, "additive_reconstruct: shares must be uint8");
    TORCH_CHECK(shares.size(0) >= 2, "additive_reconstruct: must have at least 2 shares");

    int64_t secret_len = shares.size(1);

    return at::empty({secret_len}, shares.options());
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("additive_split", &additive_split);
    m.impl("additive_reconstruct", &additive_reconstruct);
}

}  // namespace torchscience::meta::encryption
