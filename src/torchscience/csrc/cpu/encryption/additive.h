#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/additive.h"

namespace torchscience::cpu::encryption {

// Additive Secret Sharing - split secret into n shares (n-of-n scheme)
// secret: 1D tensor of bytes (the secret to split)
// randomness: random bytes for first n-1 shares, size = (n-1) * secret_len
// n: number of shares to generate (all required for reconstruction)
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

    auto secret_contig = secret.contiguous();
    auto random_contig = randomness.contiguous();
    auto shares = at::empty({n, secret_len}, secret.options());

    kernel::encryption::additive_split_kernel(
        secret_contig.data_ptr<uint8_t>(),
        shares.data_ptr<uint8_t>(),
        random_contig.data_ptr<uint8_t>(),
        secret_len, n
    );

    return shares;
}

// Additive Secret Sharing - reconstruct secret from all n shares
// shares: 2D tensor [n, secret_len] of all share values
at::Tensor additive_reconstruct(
    const at::Tensor& shares
) {
    TORCH_CHECK(shares.dim() == 2, "additive_reconstruct: shares must be 2D [n, secret_len]");
    TORCH_CHECK(shares.dtype() == at::kByte, "additive_reconstruct: shares must be uint8");
    TORCH_CHECK(shares.size(0) >= 2, "additive_reconstruct: must have at least 2 shares");

    int64_t n = shares.size(0);
    int64_t secret_len = shares.size(1);

    auto shares_contig = shares.contiguous();
    auto output = at::empty({secret_len}, shares.options());

    kernel::encryption::additive_reconstruct_kernel(
        shares_contig.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        secret_len, n
    );

    return output;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("additive_split", &additive_split);
    m.impl("additive_reconstruct", &additive_reconstruct);
}

}  // namespace torchscience::cpu::encryption
