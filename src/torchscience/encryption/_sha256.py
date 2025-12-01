import torch
from torch import Tensor


def sha256(data: Tensor) -> Tensor:
    """Compute SHA-256 hash.

    Parameters
    ----------
    data : Tensor
        Input bytes as (..., n) uint8 tensor.

    Returns
    -------
    Tensor
        (..., 32) uint8 tensor containing the 256-bit hash.
    """
    return torch.ops.torchscience.sha256(data)
