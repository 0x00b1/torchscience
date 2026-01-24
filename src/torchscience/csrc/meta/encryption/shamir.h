#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

// Shamir's Secret Sharing - split shape inference
// Returns a tensor of shape [n, secret_len]
at::Tensor shamir_split(
    const at::Tensor& secret,
    const at::Tensor& randomness,
    int64_t n,
    int64_t k
) {
    TORCH_CHECK(secret.dim() == 1, "shamir_split: secret must be 1D");
    TORCH_CHECK(secret.dtype() == at::kByte, "shamir_split: secret must be uint8");
    TORCH_CHECK(k >= 2, "shamir_split: threshold k must be >= 2");
    TORCH_CHECK(n >= k, "shamir_split: n must be >= k");
    TORCH_CHECK(n <= 255, "shamir_split: n must be <= 255");

    int64_t secret_len = secret.size(0);
    TORCH_CHECK(randomness.numel() == (k - 1) * secret_len,
                "shamir_split: randomness must have (k-1) * secret_len bytes");

    return at::empty({n, secret_len}, secret.options());
}

// Shamir's Secret Sharing - reconstruct shape inference
// Returns a tensor of shape [secret_len]
at::Tensor shamir_reconstruct(
    const at::Tensor& shares,
    const at::Tensor& indices
) {
    TORCH_CHECK(shares.dim() == 2, "shamir_reconstruct: shares must be 2D [k, secret_len]");
    TORCH_CHECK(shares.dtype() == at::kByte, "shamir_reconstruct: shares must be uint8");
    TORCH_CHECK(indices.dim() == 1, "shamir_reconstruct: indices must be 1D");
    TORCH_CHECK(indices.size(0) == shares.size(0), "shamir_reconstruct: indices must match number of shares");

    int64_t secret_len = shares.size(1);

    return at::empty({secret_len}, shares.options());
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("shamir_split", &shamir_split);
    m.impl("shamir_reconstruct", &shamir_reconstruct);
}

}  // namespace torchscience::meta::encryption
