import torch
from torch import Tensor


def x25519(scalar: Tensor, point: Tensor) -> Tensor:
    """X25519 scalar multiplication.

    Computes scalar * point on Curve25519.

    Parameters
    ----------
    scalar : Tensor
        (32,) uint8 tensor - scalar value (will be clamped internally).
    point : Tensor
        (32,) uint8 tensor - x-coordinate of point.

    Returns
    -------
    Tensor
        (32,) uint8 tensor - x-coordinate of result.
    """
    return torch.ops.torchscience.x25519(scalar, point)


def x25519_base(scalar: Tensor) -> Tensor:
    """X25519 base point multiplication.

    Computes scalar * G where G is the Curve25519 base point.

    Parameters
    ----------
    scalar : Tensor
        (32,) uint8 tensor - scalar value (will be clamped internally).

    Returns
    -------
    Tensor
        (32,) uint8 tensor - public key (x-coordinate).
    """
    return torch.ops.torchscience.x25519_base(scalar)


def x25519_keypair(seed: Tensor) -> tuple[Tensor, Tensor]:
    """Generate X25519 keypair from seed.

    Parameters
    ----------
    seed : Tensor
        (32,) uint8 tensor - random seed.

    Returns
    -------
    tuple[Tensor, Tensor]
        (private_key, public_key) - each 32 bytes.
    """
    return torch.ops.torchscience.x25519_keypair(seed)
