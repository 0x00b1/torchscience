import torch
from torch import Tensor

from torchscience.cryptography._sha256 import sha256


def hmac_sha256(key: Tensor, data: Tensor) -> Tensor:
    """Compute HMAC-SHA256.

    Parameters
    ----------
    key : Tensor
        Key as (k,) uint8 tensor. Any length.
    data : Tensor
        Message as (n,) uint8 tensor.

    Returns
    -------
    Tensor
        (32,) uint8 authentication tag.
    """
    block_size = 64

    # If key > block_size, hash it
    if key.size(0) > block_size:
        key = sha256(key)

    # Pad key to block_size
    if key.size(0) < block_size:
        key = torch.nn.functional.pad(key, (0, block_size - key.size(0)))

    # Inner and outer padding
    ipad = torch.full(
        (block_size,), 0x36, dtype=torch.uint8, device=key.device
    )
    opad = torch.full(
        (block_size,), 0x5C, dtype=torch.uint8, device=key.device
    )

    inner_key = key ^ ipad
    outer_key = key ^ opad

    # Inner hash: H(inner_key || data)
    inner_input = torch.cat([inner_key, data])
    inner_hash = sha256(inner_input)

    # Outer hash: H(outer_key || inner_hash)
    outer_input = torch.cat([outer_key, inner_hash])
    return sha256(outer_input)
