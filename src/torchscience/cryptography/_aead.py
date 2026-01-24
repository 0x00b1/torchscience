import torch
from torch import Tensor

from torchscience.cryptography._chacha20_poly1305 import (
    chacha20_poly1305_decrypt,
    chacha20_poly1305_encrypt,
)


def encrypt(
    plaintext: Tensor,
    key: Tensor,
    aad: Tensor | None = None,
    algorithm: str = "chacha20-poly1305",
) -> tuple[Tensor, Tensor, Tensor]:
    """Encrypt data with authenticated encryption.

    Generates a random nonce automatically for convenience.

    Parameters
    ----------
    plaintext : Tensor
        Data to encrypt as a 1D uint8 tensor.
    key : Tensor
        Encryption key (32 bytes for chacha20-poly1305).
    aad : Tensor, optional
        Additional authenticated data (not encrypted, but authenticated).
    algorithm : str
        Encryption algorithm. Currently only "chacha20-poly1305" supported.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        (ciphertext, nonce, tag) - all as uint8 tensors.
        Save the nonce and tag - they are required for decryption.
    """
    if algorithm != "chacha20-poly1305":
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Generate random 12-byte nonce
    nonce = torch.randint(0, 256, (12,), dtype=torch.uint8, device=key.device)

    ciphertext, tag = chacha20_poly1305_encrypt(plaintext, key, nonce, aad)

    return ciphertext, nonce, tag


def decrypt(
    ciphertext: Tensor,
    key: Tensor,
    nonce: Tensor,
    tag: Tensor,
    aad: Tensor | None = None,
    algorithm: str = "chacha20-poly1305",
) -> Tensor:
    """Decrypt data with authenticated encryption.

    Parameters
    ----------
    ciphertext : Tensor
        Encrypted data as a 1D uint8 tensor.
    key : Tensor
        Decryption key (must match encryption key).
    nonce : Tensor
        Nonce from encryption (12 bytes).
    tag : Tensor
        Authentication tag from encryption (16 bytes).
    aad : Tensor, optional
        Additional authenticated data (must match encryption).
    algorithm : str
        Encryption algorithm. Currently only "chacha20-poly1305" supported.

    Returns
    -------
    Tensor
        Decrypted plaintext as a 1D uint8 tensor.

    Raises
    ------
    RuntimeError
        If authentication fails (ciphertext or AAD was tampered with).
    """
    if algorithm != "chacha20-poly1305":
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return chacha20_poly1305_decrypt(ciphertext, key, nonce, tag, aad)
