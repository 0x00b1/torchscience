import torch
from torch import Tensor


def poly1305(data: Tensor, key: Tensor) -> Tensor:
    """Compute Poly1305 authentication tag.

    Parameters
    ----------
    data : Tensor
        (n,) uint8 tensor - message to authenticate.
    key : Tensor
        (32,) uint8 tensor - one-time key (r || s).

    Returns
    -------
    Tensor
        (16,) uint8 tensor - authentication tag.
    """
    return torch.ops.torchscience.poly1305(data, key)
