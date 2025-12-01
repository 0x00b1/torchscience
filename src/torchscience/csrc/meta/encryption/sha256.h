#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

at::Tensor sha256(const at::Tensor& data) {
    TORCH_CHECK(data.dtype() == at::kByte, "sha256: data must be uint8");
    return at::empty({32}, data.options());
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) { m.impl("sha256", &sha256); }

}  // namespace torchscience::meta::encryption
