import torch
from torch import Tensor


def additive_split(
    secret: Tensor,
    n: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Split a secret into n additive shares.

    Uses XOR-based additive sharing. All n shares are required
    to reconstruct the secret (n-of-n scheme).

    Parameters
    ----------
    secret : Tensor
        Secret bytes as a 1D uint8 tensor.
    n : int
        Number of shares to generate.
    generator : torch.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    Tensor
        Shares as a 2D uint8 tensor of shape (n, len(secret)).
    """
    if n < 2:
        raise ValueError("n must be at least 2")

    secret_len = secret.size(0)
    randomness = torch.randint(
        0,
        256,
        ((n - 1) * secret_len,),
        dtype=torch.uint8,
        device=secret.device,
        generator=generator,
    )

    return torch.ops.torchscience.additive_split(secret, randomness, n)


def additive_reconstruct(shares: Tensor) -> Tensor:
    """Reconstruct a secret from all n additive shares.

    Parameters
    ----------
    shares : Tensor
        All n shares as a 2D uint8 tensor of shape (n, secret_len).

    Returns
    -------
    Tensor
        Reconstructed secret as a 1D uint8 tensor.
    """
    return torch.ops.torchscience.additive_reconstruct(shares)
