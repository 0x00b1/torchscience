#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/shamir.h"

namespace torchscience::cpu::encryption {

// Shamir's Secret Sharing - split secret into n shares with threshold k
// secret: 1D tensor of bytes (the secret to split)
// randomness: random bytes for polynomial coefficients, size = (k-1) * secret_len
// n: number of shares to generate
// k: threshold - minimum shares needed to reconstruct
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

    auto secret_contig = secret.contiguous();
    auto random_contig = randomness.contiguous();
    auto shares = at::empty({n, secret_len}, secret.options());

    kernel::encryption::shamir_split_kernel(
        secret_contig.data_ptr<uint8_t>(),
        shares.data_ptr<uint8_t>(),
        random_contig.data_ptr<uint8_t>(),
        secret_len, n, k
    );

    return shares;
}

// Shamir's Secret Sharing - reconstruct secret from k shares
// shares: 2D tensor [k, secret_len] of share values
// indices: 1D tensor of share indices (1-indexed x-coordinates)
at::Tensor shamir_reconstruct(
    const at::Tensor& shares,
    const at::Tensor& indices
) {
    TORCH_CHECK(shares.dim() == 2, "shamir_reconstruct: shares must be 2D [k, secret_len]");
    TORCH_CHECK(shares.dtype() == at::kByte, "shamir_reconstruct: shares must be uint8");
    TORCH_CHECK(indices.dim() == 1, "shamir_reconstruct: indices must be 1D");
    TORCH_CHECK(indices.size(0) == shares.size(0), "shamir_reconstruct: indices must match number of shares");

    int64_t k = shares.size(0);
    int64_t secret_len = shares.size(1);

    auto shares_contig = shares.contiguous();
    auto indices_byte = indices.to(at::kByte).contiguous();
    auto output = at::empty({secret_len}, shares.options());

    kernel::encryption::shamir_reconstruct_kernel(
        shares_contig.data_ptr<uint8_t>(),
        indices_byte.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        secret_len, k
    );

    return output;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("shamir_split", &shamir_split);
    m.impl("shamir_reconstruct", &shamir_reconstruct);
}

}  // namespace torchscience::cpu::encryption
