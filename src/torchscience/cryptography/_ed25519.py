import torch
from torch import Tensor


def ed25519_keypair(seed: Tensor) -> tuple[Tensor, Tensor]:
    """Generate Ed25519 keypair from seed.

    Parameters
    ----------
    seed : Tensor
        (32,) uint8 tensor - random seed.

    Returns
    -------
    tuple[Tensor, Tensor]
        (private_key, public_key)
        - private_key: (64,) uint8 tensor (seed || public_key)
        - public_key: (32,) uint8 tensor
    """
    return torch.ops.torchscience.ed25519_keypair(seed)


def ed25519_sign(private_key: Tensor, message: Tensor) -> Tensor:
    """Sign message with Ed25519.

    Parameters
    ----------
    private_key : Tensor
        (64,) uint8 tensor - private key from ed25519_keypair.
    message : Tensor
        (n,) uint8 tensor - message bytes to sign (any length).

    Returns
    -------
    Tensor
        (64,) uint8 tensor - signature (R || S).
    """
    return torch.ops.torchscience.ed25519_sign(private_key, message)


def ed25519_verify(
    public_key: Tensor,
    message: Tensor,
    signature: Tensor,
) -> Tensor:
    """Verify Ed25519 signature.

    Parameters
    ----------
    public_key : Tensor
        (32,) uint8 tensor - public key.
    message : Tensor
        (n,) uint8 tensor - original message bytes.
    signature : Tensor
        (64,) uint8 tensor - signature to verify.

    Returns
    -------
    Tensor
        Boolean tensor (True if valid, False otherwise).
    """
    return torch.ops.torchscience.ed25519_verify(
        public_key, message, signature
    )
