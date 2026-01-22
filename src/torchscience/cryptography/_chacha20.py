import torch
from torch import Tensor


def chacha20(
    key: Tensor,
    nonce: Tensor,
    num_bytes: int,
    counter: int = 0,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.uint8,
) -> Tensor:
    """Generate pseudorandom bytes using ChaCha20 stream cipher."""
    return torch.ops.torchscience.chacha20(key, nonce, num_bytes, counter)
