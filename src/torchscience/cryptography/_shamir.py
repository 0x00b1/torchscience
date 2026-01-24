import torch
from torch import Tensor


def shamir_split(
    secret: Tensor,
    n: int,
    k: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Split a secret into n shares using Shamir's Secret Sharing.

    Uses GF(2^8) arithmetic. Any k shares can reconstruct the secret,
    while k-1 shares reveal no information about it.

    Parameters
    ----------
    secret : Tensor
        Secret bytes as a 1D uint8 tensor.
    n : int
        Number of shares to generate.
    k : int
        Threshold - minimum shares needed to reconstruct.
    generator : torch.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Tensor
        Shares as a 2D uint8 tensor of shape (n, len(secret)).
        Share i corresponds to x-coordinate (i+1).
    """
    if k < 2:
        raise ValueError("Threshold k must be at least 2")
    if n < k:
        raise ValueError("n must be at least k")

    secret_len = secret.size(0)
    randomness = torch.randint(
        0,
        256,
        ((k - 1) * secret_len,),
        dtype=torch.uint8,
        device=secret.device,
        generator=generator,
    )

    return torch.ops.torchscience.shamir_split(secret, randomness, n, k)


def shamir_reconstruct(shares: Tensor, indices: Tensor) -> Tensor:
    """Reconstruct a secret from k Shamir shares.

    Parameters
    ----------
    shares : Tensor
        k shares as a 2D uint8 tensor of shape (k, secret_len).
    indices : Tensor
        x-coordinates of the shares (1-indexed) as a 1D tensor.

    Returns
    -------
    Tensor
        Reconstructed secret as a 1D uint8 tensor.
    """
    return torch.ops.torchscience.shamir_reconstruct(shares, indices)
