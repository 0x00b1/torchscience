import torch
from torch import Tensor

from torchscience.cryptography._hkdf import hkdf_sha256
from torchscience.cryptography._x25519 import x25519


def derive_key(
    shared_secret: Tensor,
    info: Tensor | bytes,
    length: int = 32,
    salt: Tensor | None = None,
) -> Tensor:
    """Derive a cryptographic key from a shared secret using HKDF.

    Parameters
    ----------
    shared_secret : Tensor
        Shared secret as a 1D uint8 tensor (e.g., from X25519).
    info : Tensor or bytes
        Context/application-specific info for domain separation.
    length : int
        Desired key length in bytes (default 32).
    salt : Tensor, optional
        Optional salt. Defaults to zeros if not provided.

    Returns
    -------
    Tensor
        Derived key as a 1D uint8 tensor.

    Examples
    --------
    >>> shared = x25519_shared_secret(my_private, their_public)
    >>> enc_key = derive_key(shared, b"encryption", length=32)
    >>> mac_key = derive_key(shared, b"authentication", length=32)
    """
    if isinstance(info, bytes):
        info = torch.tensor(
            list(info), dtype=torch.uint8, device=shared_secret.device
        )

    if salt is None:
        salt = torch.zeros(32, dtype=torch.uint8, device=shared_secret.device)

    return hkdf_sha256(shared_secret, salt, info, length)


def x25519_shared_secret(private_key: Tensor, public_key: Tensor) -> Tensor:
    """Compute X25519 ECDH shared secret.

    Performs elliptic curve Diffie-Hellman key exchange using Curve25519.

    Parameters
    ----------
    private_key : Tensor
        Your private key as a 32-byte uint8 tensor.
    public_key : Tensor
        Their public key as a 32-byte uint8 tensor.

    Returns
    -------
    Tensor
        Shared secret as a 32-byte uint8 tensor.

    Notes
    -----
    The shared secret should be passed through a KDF (like derive_key)
    before use as an encryption key.

    Examples
    --------
    >>> alice_priv, alice_pub = x25519_keypair()
    >>> bob_priv, bob_pub = x25519_keypair()
    >>> # Both compute same shared secret
    >>> alice_shared = x25519_shared_secret(alice_priv, bob_pub)
    >>> bob_shared = x25519_shared_secret(bob_priv, alice_pub)
    >>> # alice_shared == bob_shared
    """
    return x25519(private_key, public_key)
