import torch
from torch import Tensor


def sha3_256(data: Tensor) -> Tensor:
    """Compute SHA3-256 hash.

    Parameters
    ----------
    data : Tensor
        Input bytes as (n,) uint8 tensor.

    Returns
    -------
    Tensor
        (32,) uint8 tensor containing the 256-bit hash.
    """
    return torch.ops.torchscience.sha3_256(data)


def sha3_512(data: Tensor) -> Tensor:
    """Compute SHA3-512 hash.

    Parameters
    ----------
    data : Tensor
        Input bytes as (n,) uint8 tensor.

    Returns
    -------
    Tensor
        (64,) uint8 tensor containing the 512-bit hash.
    """
    return torch.ops.torchscience.sha3_512(data)


def keccak256(data: Tensor) -> Tensor:
    """Compute Keccak-256 hash (Ethereum variant).

    Parameters
    ----------
    data : Tensor
        Input bytes as (n,) uint8 tensor.

    Returns
    -------
    Tensor
        (32,) uint8 tensor containing the 256-bit hash.

    Notes
    -----
    This uses the original Keccak padding (0x01) rather than SHA3 padding (0x06).
    This is the hash function used by Ethereum.
    """
    return torch.ops.torchscience.keccak256(data)
